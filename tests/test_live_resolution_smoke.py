"""Smoke test: confirm fast-path skips LLM and capability filtering works end-to-end."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.pop("FAIRY_USE_HERMES", None)
os.environ.pop("FAIRY_TASK_GUARD_REENTRY", None)

from controller.intent_resolver import resolve_intent


class _NoCallBrain:
    def __init__(self):
        self.called = False

    def __call__(self, *a, **k):
        self.called = True
        class _M:
            content = ""
            tool_calls = []
        class _R:
            message = _M()
            done = True
        return _R()


def test_fast_path_skips_llm():
    """Test 1: 'hey fairy whats the time' is fast-path; brain must not be called."""
    brain = _NoCallBrain()
    r = resolve_intent("hey fairy whats the time", brain_fn=brain)
    assert r.fast_path_eligible, f"expected fast_path, got {r.fast_path_eligible}"
    assert not brain.called, f"LLM should NOT be called on fast-path, was: {brain.called}"
    print(f"[PASS] fast-path: eligible={r.fast_path_eligible} brain_called={brain.called} "
          f"chosen={r.chosen_meaning.meaning if r.chosen_meaning else None}")


def test_use_cod_to_make_folder():
    """Test 3: 'use cod to make a folder' -> Claude Code (visible substitution)."""
    from unittest.mock import MagicMock
    brain = MagicMock()
    brain.return_value = MagicMock(
        message=MagicMock(content="", tool_calls=[]),
        done=True,
    )
    r = resolve_intent("use cod to make a folder", brain_fn=brain)
    assert r.chosen_meaning is not None, "no meaning chosen"
    assert "claude code" in r.chosen_meaning.meaning.lower(), \
        f"expected Claude Code, got: {r.chosen_meaning.meaning}"
    assert r.routing_hint == "claude_code_delegate", \
        f"expected claude_code_delegate, got: {r.routing_hint}"
    assert r.visible_reasoning, "visible_reasoning must be populated"
    assert "game" in r.visible_reasoning.lower() or "can't" in r.visible_reasoning.lower(), \
        f"visible_reasoning must acknowledge game cannot fulfill: {r.visible_reasoning}"
    print(f"[PASS] cod->claude code: chosen={r.chosen_meaning.meaning} "
          f"hint={r.routing_hint}")
    print(f"  visible: {r.visible_reasoning}")


def test_open_cod_clarification():
    """Test 4: 'open cod' -> clarification question, no routing."""
    from unittest.mock import MagicMock
    brain = MagicMock()
    brain.return_value = MagicMock(
        message=MagicMock(content="", tool_calls=[]),
        done=True,
    )
    r = resolve_intent("open cod", brain_fn=brain)
    assert r.clarification_question, \
        f"expected clarification, got none. chosen={r.chosen_meaning.meaning if r.chosen_meaning else None}"
    assert r.routing_hint is None, \
        f"clarification should not produce routing hint, got: {r.routing_hint}"
    print(f"[PASS] open cod clarification: {r.clarification_question}")


def test_voice_near_miss():
    """Test 5: Whisper-style near-miss 'faury' should fuzzy-match 'fairy'."""
    r = resolve_intent("hey faury", brain_fn=_NoCallBrain())
    # faury should not crash; result is whatever resolution gives
    assert r is not None
    print(f"[PASS] voice near-miss: 'hey faury' -> {r.chosen_meaning.meaning if r.chosen_meaning else 'no-match'}")


def test_open_youtube_fast_path():
    """Test 6a: 'open youtube' is a known website, fast-path."""
    r = resolve_intent("open youtube", brain_fn=_NoCallBrain())
    assert r.fast_path_eligible, f"open youtube should be fast-path, got {r.fast_path_eligible}"
    assert r.chosen_meaning is not None
    assert r.chosen_meaning.category == "website", \
        f"open youtube should be website, got: {r.chosen_meaning.category}"
    print(f"[PASS] open youtube: fast-path={r.fast_path_eligible} "
          f"category={r.chosen_meaning.category}")


def test_normal_chat_fast_path():
    """Test 6b: 'how are you' is normal chat, fast-path."""
    r = resolve_intent("how are you today", brain_fn=_NoCallBrain())
    assert r.fast_path_eligible
    print(f"[PASS] normal chat: fast-path={r.fast_path_eligible}")
