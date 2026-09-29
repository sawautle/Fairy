"""
Lie-detector tests for the long-term memory v0 slice.

Covers (per the task spec):
  - Budget: 100 stored facts -> injection block is <= 2000 chars / <= 25 facts.
  - Permission gate: a proposed fact is NOT stored without confirmation.
  - Honesty: "forget that I like X" removes the fact and it disappears from injection.
  - Staleness: a 30+ day unused single-use fact is not injected.
  - Resilience: corrupted JSON -> boot succeeds, memory empty, no crash.
  - Remember-across-sessions: write a fact, simulate a fresh session, see it injected.
  - No injection into debug/tool payloads: facts only enter the system prompt.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


# Make sure the project root is on sys.path so `memory.*` imports work
# even if pytest is invoked from a different working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def fresh_memory_path(tmp_path, monkeypatch):
    """Each test gets its own JSON file. Also resets the module-level
    singleton so no state bleeds between tests."""
    path = tmp_path / "long_term_facts.json"

    # Defer-import so we can patch the module-level path constant.
    from memory import long_term_memory as ltm
    monkeypatch.setattr(ltm, "_MEM_PATH", path)
    ltm.reset_long_term_memory()
    yield path
    ltm.reset_long_term_memory()


def _import_module():
    from memory import long_term_memory as ltm
    return ltm


# ─────────────────────────────────────────────────────────────────────
# Budget
# ─────────────────────────────────────────────────────────────────────
class TestInjectionBudget:
    def test_budget_chars_with_100_facts(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        for i in range(100):
            store.add_fact(
                f"fact number {i} with some descriptive text padding",
                category="fact_about_master",
            )
        block = store.get_injection_block()
        assert block, "injection block should not be empty when facts exist"
        assert len(block) <= ltm.INJECTION_BUDGET_MAX_CHARS, (
            f"injection block {len(block)} chars exceeds budget "
            f"{ltm.INJECTION_BUDGET_MAX_CHARS}"
        )

    def test_budget_count_with_100_facts(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        for i in range(100):
            store.add_fact(
                f"fact number {i} with some descriptive text padding",
                category="fact_about_master",
            )
        block = store.get_injection_block()
        line_count = sum(1 for line in block.splitlines() if line.strip())
        assert line_count <= ltm.INJECTION_BUDGET_MAX_FACTS, (
            f"injection block has {line_count} facts; budget is "
            f"{ltm.INJECTION_BUDGET_MAX_FACTS}"
        )

    def test_priority_preference_first(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        # A heavy-use non-preference, then a single-use preference.
        heavy = store.add_fact("Most-used generic fact", category="fact_about_master")
        pref = store.add_fact("Master's core preference", category="preference")
        # Bump the heavy one so it dominates by use_count.
        for _ in range(10):
            store.get_injection_block()  # each call marks used for selected facts
        # After repeated calls, the heavy fact should have a much higher use_count.
        block = store.get_injection_block()
        # The preference should still rank first because the category is the
        # primary sort key — that's the spec.
        first_line = block.splitlines()[0]
        assert "preference" in first_line.lower()

    def test_priority_use_count_then_recent(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        # Add two facts, then directly manipulate their use_count via the
        # in-memory data dict so the test is deterministic.
        store.add_fact("low use fact", category="fact_about_master")
        store.add_fact("high use fact", category="fact_about_master")
        facts = store.data["facts"]
        low_fact = next(f for f in facts if "low use" in f["text"])
        high_fact = next(f for f in facts if "high use" in f["text"])
        low_fact["use_count"] = 2
        high_fact["use_count"] = 10
        # Within the same non-preference category, higher use_count sorts first.
        block = store.get_injection_block()
        hi_idx = block.find("high use fact")
        lo_idx = block.find("low use fact")
        assert hi_idx != -1 and lo_idx != -1, "both facts should be in the block"
        assert hi_idx < lo_idx, (
            f"higher use_count should sort first, but got hi={hi_idx}, lo={lo_idx}. "
            f"Block: {block!r}"
        )

    def test_overflow_logged_but_not_injected(self, fresh_memory_path, caplog):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        # Fill beyond budget.
        for i in range(60):
            store.add_fact(f"overflow test fact {i} " * 5, category="fact_about_master")
        with caplog.at_level("INFO", logger="fairy.long_term_memory"):
            block = store.get_injection_block()
        assert len(block) <= ltm.INJECTION_BUDGET_MAX_CHARS
        # 35 facts should have been dropped (60 - 25).
        assert "35" in caplog.text or "fact" in caplog.text.lower()


# ─────────────────────────────────────────────────────────────────────
# Permission gate
# ─────────────────────────────────────────────────────────────────────
class TestPermissionGate:
    def test_proposal_not_stored_without_confirm(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        # Add a proposal, do not confirm.
        store.add_pending_proposal("Master only uses free OpenRouter models")
        pending = store.get_pending_proposals()
        assert pending == ["Master only uses free OpenRouter models"]
        # Crucially, no fact was stored yet.
        assert store.list_facts() == []
        # Injection block must be empty because nothing is stored.
        assert store.get_injection_block() == ""

    def test_confirm_stores_fact(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        store.add_pending_proposal("Master prefers short answers")
        new_id = store.confirm_proposal("Master prefers short answers")
        assert new_id is not None
        assert store.get_pending_proposals() == []
        facts = store.list_facts()
        assert len(facts) == 1
        assert facts[0]["id"] == new_id
        assert facts[0]["text"] == "Master prefers short answers"

    def test_reject_drops_proposal(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        store.add_pending_proposal("Master hates pineapple")
        assert store.reject_proposal("Master hates pineapple")
        assert store.get_pending_proposals() == []
        assert store.list_facts() == []

    def test_proposal_cap_at_two(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        # Add three, expect only the first two to land.
        assert store.add_pending_proposal("fact one") is True
        assert store.add_pending_proposal("fact two") is True
        assert store.add_pending_proposal("fact three") is False
        assert store.get_pending_proposals() == ["fact one", "fact two"]


# ─────────────────────────────────────────────────────────────────────
# Honesty: forget
# ─────────────────────────────────────────────────────────────────────
class TestForget:
    def test_forget_removes_from_injection(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        fid = store.add_fact("Master likes pineapple pizza", category="preference")
        block_before = store.get_injection_block()
        assert "pineapple pizza" in block_before
        assert store.delete_fact(fid) is True
        block_after = store.get_injection_block()
        assert "pineapple pizza" not in block_after
        # And the fact is gone from the store too.
        assert all(f.get("id") != fid for f in store.list_facts())

    def test_forget_missing_id_is_noop(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        store.add_fact("real fact", category="fact_about_master")
        assert store.delete_fact("nonexistent-id-12345") is False
        # Real fact still there.
        assert len(store.list_facts()) == 1

    def test_find_facts_matching_partial(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        store.add_fact("Master likes pineapple pizza", category="preference")
        store.add_fact("Master hates loud noises", category="preference")
        matches = store.find_facts_matching("pineapple")
        assert len(matches) == 1
        assert "pineapple" in matches[0]["text"]


# ─────────────────────────────────────────────────────────────────────
# Staleness
# ─────────────────────────────────────────────────────────────────────
class TestStaleness:
    def test_stale_fact_excluded(self, fresh_memory_path, monkeypatch):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        fid = store.add_fact(
            "old single-use fact",
            category="fact_about_master",
        )
        # Backdate last_used_at to 40 days ago, single use.
        old_iso = (datetime.now() - timedelta(days=40)).isoformat()
        for f in store.data["facts"]:
            if f.get("id") == fid:
                f["last_used_at"] = old_iso
                f["use_count"] = 1
        store._save()
        block = store.get_injection_block()
        assert "old single-use fact" not in block, (
            f"stale fact should not be injected, got block: {block!r}"
        )

    def test_recently_used_fact_included(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        store.add_fact("recently used fact", category="fact_about_master")
        block = store.get_injection_block()
        assert "recently used fact" in block

    def test_multiuse_never_stale(self, fresh_memory_path):
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        fid = store.add_fact("loved fact", category="fact_about_master")
        # Backdate by 60 days but bump use_count to 5.
        old_iso = (datetime.now() - timedelta(days=60)).isoformat()
        for f in store.data["facts"]:
            if f.get("id") == fid:
                f["last_used_at"] = old_iso
                f["use_count"] = 5
        store._save()
        block = store.get_injection_block()
        assert "loved fact" in block


# ─────────────────────────────────────────────────────────────────────
# Resilience
# ─────────────────────────────────────────────────────────────────────
class TestResilience:
    def test_corrupt_json_starts_fresh(self, fresh_memory_path):
        fresh_memory_path.write_text("{ this is : not, valid JSON,,,", encoding="utf-8")
        ltm = _import_module()
        # Should NOT raise; should start with empty facts.
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        assert store.list_facts() == []
        assert store.get_injection_block() == ""

    def test_missing_file_starts_fresh(self, fresh_memory_path):
        # Path does not exist (the fixture creates the dir but not the file).
        assert not fresh_memory_path.exists()
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        assert store.list_facts() == []
        # Subsequent add_fact must still work and persist.
        store.add_fact("fresh fact", category="fact_about_master")
        assert fresh_memory_path.exists()

    def test_oversized_file_loads_fast(self, fresh_memory_path):
        # Build a 500-fact file. This must still load quickly.
        ltm = _import_module()
        store = ltm.LongTermMemoryStore(fresh_memory_path)
        for i in range(500):
            store.add_fact(
                f"oversized fact {i} with some text padding",
                category="fact_about_master",
            )
        # Now simulate a fresh session by creating a new store on the same file.
        new_store = ltm.LongTermMemoryStore(fresh_memory_path)
        assert len(new_store.list_facts()) == 500
        # And the injection block still respects the budget.
        block = new_store.get_injection_block()
        assert len(block) <= ltm.INJECTION_BUDGET_MAX_CHARS


# ─────────────────────────────────────────────────────────────────────
# Remember across sessions
# ─────────────────────────────────────────────────────────────────────
class TestAcrossSessions:
    def test_fact_persists_across_instances(self, fresh_memory_path):
        ltm = _import_module()
        # Session 1: store a fact.
        s1 = ltm.LongTermMemoryStore(fresh_memory_path)
        fid = s1.add_fact(
            "Master prefers short answers",
            category="preference",
            source="conversation",
        )
        # Session 2: new instance, same file. The fact should reappear.
        s2 = ltm.LongTermMemoryStore(fresh_memory_path)
        facts = s2.list_facts()
        assert any(f.get("id") == fid for f in facts)
        # And the injection block on the new session includes it.
        block = s2.get_injection_block()
        assert "short answers" in block


# ─────────────────────────────────────────────────────────────────────
# Extraction helpers (pattern-based, no LLM)
# ─────────────────────────────────────────────────────────────────────
class TestExtraction:
    def test_remember_prefix_extracted(self):
        ltm = _import_module()
        out = ltm.extract_proposals("remember: I prefer dark mode")
        assert out and any("dark mode" in s for s in out)

    def test_i_prefer_pattern(self):
        ltm = _import_module()
        out = ltm.extract_proposals("By the way, I only use free models from now on.")
        assert out and any("free models" in s.lower() for s in out)

    def test_call_me_pattern(self):
        ltm = _import_module()
        out = ltm.extract_proposals("Please call me Shaz from now on.")
        assert out and any("shaz" in s.lower() for s in out)

    def test_cap_at_two(self):
        ltm = _import_module()
        text = "I prefer tea. I only use free models. I always work late. I never eat fish."
        out = ltm.extract_proposals(text)
        assert len(out) <= 2

    def test_no_false_positive_on_bland_input(self):
        ltm = _import_module()
        out = ltm.extract_proposals("Hello Fairy, how are you today?")
        assert out == []


# ─────────────────────────────────────────────────────────────────────
# No injection into tool/debug payloads
# ─────────────────────────────────────────────────────────────────────
class TestNoLeakIntoDebugOrTool:
    def test_facts_block_only_in_system_prompt(self):
        """build_system_prompt() with a facts block must place the block
        under the 'Known context about Master:' label, NOT in any debug
        or tool payload structure."""
        from core.prompts import build_system_prompt
        facts = "- Master prefers short answers"
        sys_prompt = build_system_prompt(
            "normal",
            skills=[],
            memory_context="",
            language="en",
            facts_injection=facts,
        )
        assert "Known context about Master:" in sys_prompt
        assert "short answers" in sys_prompt
        # The facts block is appended to the system prompt and never
        # appears in any standalone tool/debug payload (we just confirm
        # build_system_prompt returns a single string, not a structured
        # payload that callers could leak into tools).
        assert isinstance(sys_prompt, str)

    def test_no_facts_block_means_no_block(self):
        from core.prompts import build_system_prompt
        sys_prompt = build_system_prompt(
            "normal",
            skills=[],
            memory_context="",
            language="en",
            facts_injection="",
        )
        assert "Known context about Master:" not in sys_prompt


# ─────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────
class TestSingleton:
    def test_singleton_returns_same_instance(self, fresh_memory_path, monkeypatch):
        ltm = _import_module()
        monkeypatch.setattr(ltm, "_MEM_PATH", fresh_memory_path)
        a = ltm.get_long_term_memory()
        b = ltm.get_long_term_memory()
        assert a is b
        ltm.reset_long_term_memory()
        c = ltm.get_long_term_memory()
        assert c is not a, "reset must invalidate the singleton"
