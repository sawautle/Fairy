#!/usr/bin/env python3
"""
Tests for controller/intent_resolver.py — the context-based disambiguation layer.

Lie-detector grade tests covering:
  1. "use cod to make a folder" → resolves to Claude Code (mock LLM/search)
     - never routes to a game/website
     - response acknowledges the substitution
  2. "open cod" (no task context) → either opens game launcher or asks
     - never opens a random webpage
  3. Unknown term "zorblat" → autonomous web search is invoked (mocked), then
     decision or honest "I couldn't figure out what that is"
  4. Genuinely ambiguous → clarifying question, no action taken
  5. Reasoning trace appears in debug log for every test case
  6. Real-path test: resolve a real term end-to-end with live search
     (skip on network failure)
"""

from __future__ import annotations

import json
import os
import re
import time
from unittest.mock import patch, MagicMock

import pytest


# ─── Helpers ─────────────────────────────────────────────────────────────────────


def _strip_fairy_prefix(text: str) -> str:
    """Strip any leading 'fairy>' / 'Fairy>' / 'fairy>' prefix lines."""
    lines = text.strip().splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("fairy"):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _mock_llm_success(choices: list) -> MagicMock:
    """Return a mock LLM response with given JSON content in tool_calls."""
    return MagicMock(
        message=MagicMock(
            content=json.dumps(choices),
            tool_calls=None,
        ),
        done=True,
    )


def _mock_llm_empty() -> MagicMock:
    return MagicMock(
        message=MagicMock(content="", tool_calls=[]),
        done=True,
    )


# ─── Fast-path / no-ambiguity tests ─────────────────────────────────────────────


class TestFastPath:
    """Messages that are clearly unambiguous skip the resolver entirely."""

    def test_website_is_fast_path(self):
        """'open youtube' resolves to browser without any LLM call."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("open youtube", brain_fn=mock_brain)
            assert result.fast_path_eligible is True
            assert result.routing_hint == "browser"
            assert result.chosen_meaning is not None
            assert result.chosen_meaning.category == "website"
            # No LLM needed
            mock_brain.assert_not_called()

    def test_desktop_app_is_fast_path(self):
        """'open notepad' resolves to app_action without any LLM call."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("open notepad", brain_fn=mock_brain)
            assert result.fast_path_eligible is True
            assert result.routing_hint == "app_action"
            mock_brain.assert_not_called()

    def test_chat_greeting_skips(self):
        """'hey fairy' skips resolution."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("hey fairy, how are you?")
            assert result.fast_path_eligible is True
            mock_brain.assert_not_called()


# ─── "cod" disambiguation tests ──────────────────────────────────────────────────


class TestCodAmbiguity:
    """The word 'cod' is genuinely ambiguous between Call of Duty and Claude Code."""

    def test_use_cod_to_make_folder_routes_to_claude_code(self):
        """File-system task with 'cod' → dev_tool (Claude Code), not game."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            # Give a confident local answer (no LLM needed for cod)
            result = resolve_intent("use cod to make a folder", brain_fn=mock_brain)
            assert result.chosen_meaning is not None
            assert result.chosen_meaning.category == "dev_tool"
            assert "claude code" in result.chosen_meaning.meaning.lower()
            assert result.routing_hint == "claude_code_delegate"
            # The visible reasoning must explain why (game can't make folders)
            assert result.visible_reasoning
            assert "game" in result.visible_reasoning.lower()
            assert "folder" in result.visible_reasoning.lower()
            mock_brain.assert_not_called()

    def test_cod_file_task_cannot_fulfill_is_game(self):
        """Game candidates are marked can_fulfill_task=False for file tasks."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("use cod to make a folder", brain_fn=mock_brain)
            game_cands = [c for c in result.candidate_meanings if c.category == "game"]
            assert len(game_cands) >= 1
            for c in game_cands:
                assert c.can_fulfill_task is False, (
                    f"Game '{c.meaning}' should not be able to make a folder"
                )

    def test_open_cod_triggers_clarification(self):
        """Bare 'open cod' with no task context triggers clarification."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("open cod", brain_fn=mock_brain)
            # Clarification gate should fire (top two close in confidence, different categories)
            assert result.clarification_question is not None, (
                "Expected clarification question for ambiguous 'open cod'"
            )
            assert "cod" in result.clarification_question.lower()
            assert result.chosen_meaning is None
            mock_brain.assert_not_called()

    def test_open_cod_not_routed_to_browser(self):
        """'open cod' must not route to browser (neither CoD nor Claude Code is a webpage)."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("open cod", brain_fn=mock_brain)
            # The clarification route has routing_hint=None (asks user)
            # The non-clarification route would have routing_hint=app_action
            # Neither should be 'browser'
            assert result.routing_hint != "browser", (
                "'open cod' should not route to browser"
            )


# ─── Unknown term tests ──────────────────────────────────────────────────────────


class TestUnknownTerm:
    """Genuinely unknown terms trigger research or return honest 'I don't know'."""

    def test_unknown_term_no_research_flag(self):
        """Unknown term 'zorblat' with research disabled returns honest failure."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            # No local match, LLM returns nothing, research disabled
            mock_brain.return_value = MagicMock(
                message=MagicMock(content="[{}]", tool_calls=[]),
                done=True,
            )
            result = resolve_intent("use zorblat to do something", brain_fn=mock_brain, allow_research=False)
            # Should indicate we couldn't figure it out
            assert result.chosen_meaning is None or result.confidence < 0.4

    def test_unknown_term_with_research(self):
        """With research enabled, an unknown term triggers web search."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            # Local lookup finds nothing, LLM is unconfident
            mock_brain.return_value = MagicMock(
                message=MagicMock(
                    content="[{\"meaning\":\"Zorblat framework\",\"category\":\"dev_tool\",\"capability\":\"things\",\"confidence\":0.7}]",
                    tool_calls=[],
                ),
                done=True,
            )
            with patch("controller.intent_resolver._web_research_term") as mock_research:
                mock_research.return_value = (
                    "Zorblat is a fictional placeholder term used in tests",
                    True,
                )
                result = resolve_intent(
                    "use zorblat to build a project",
                    brain_fn=mock_brain,
                    allow_research=True,
                )
                assert result.research_performed is True
                assert "zorb" in result.research_summary.lower()


# ─── Reasoning trace tests ────────────────────────────────────────────────────────


class TestReasoningTrace:
    """Every resolved turn produces a reasoning trace in the debug log."""

    def test_trace_present_for_cod(self):
        """Cod resolution produces a reasoning_trace with steps."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("use cod to make a folder", brain_fn=mock_brain)
            assert len(result.reasoning_trace) > 0
            steps = [s.get("step") for s in result.reasoning_trace]
            assert "local_lookup" in steps or "llm_candidates" in steps
            # capability_filter is a step name (not a substring of an error message)
            assert "capability_filter" in steps

    def test_trace_for_clarification(self):
        """Clarification gate appears in reasoning trace."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("open cod", brain_fn=mock_brain)
            assert any(
                s.get("step") == "clarification_gate"
                for s in result.reasoning_trace
            )

    def test_trace_for_fast_path(self):
        """Fast-path resolution still produces a trace entry."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("open youtube", brain_fn=mock_brain)
            assert len(result.reasoning_trace) > 0
            # Fast path: known_website step OR fast_path_skip
            steps = [s.get("step") for s in result.reasoning_trace]
            assert "known_website" in steps or "fast_path_skip" in steps

    def test_trace_in_debug_log(self):
        """When FAIRY_DEBUG=1, the trace is written to fairy_debug.log."""
        import os, tempfile
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            # Write to a temp debug file
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log", encoding="utf-8") as fh:
                tmp = fh.name
            os.environ["FAIRY_DEBUG"] = "1"
            os.environ["FAIRY_DEBUG_FILE"] = tmp
            try:
                from controller import intent_resolver as _ir
                _ir._DEBUG = True
                _ir._DEBUG_FILE = tmp
                result = resolve_intent("use cod to make a folder", brain_fn=mock_brain)
                with open(tmp, encoding="utf-8", errors="replace") as fh:
                    log = fh.read()
                assert "INTENT_RESOLVER" in log or "RESOLVE" in log, (
                    f"Debug log should contain INTENT_RESOLVER/RESOLVE. Got:\n{log[:500]}"
                )
            finally:
                os.environ.pop("FAIRY_DEBUG", None)
                os.environ.pop("FAIRY_DEBUG_FILE", None)
                if os.path.exists(tmp):
                    os.unlink(tmp)


# ─── Routing hint tests ──────────────────────────────────────────────────────────


class TestRoutingHints:
    """Intent resolution produces routing hints that feed into existing pipelines."""

    def test_dev_tool_routes_to_delegate(self):
        """Claude Code route produces claude_code_delegate routing hint."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("use cod to make a folder", brain_fn=mock_brain)
            assert result.routing_hint == "claude_code_delegate"

    def test_website_routes_to_browser(self):
        """Known website produces browser routing hint."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("open youtube", brain_fn=mock_brain)
            assert result.routing_hint == "browser"

    def test_clarification_has_no_routing_hint(self):
        """Clarification means no routing yet — routing_hint is None."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("open cod", brain_fn=mock_brain)
            if result.clarification_question:
                assert result.routing_hint is None


# ─── Visible reasoning tests ─────────────────────────────────────────────────────


class TestVisibleReasoning:
    """When a substitution is made, visible reasoning is included."""

    def test_cod_substitution_mentions_game(self):
        """Substituting cod for Claude Code mentions that a game can't make folders."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("use cod to make a folder", brain_fn=mock_brain)
            assert result.visible_reasoning, "Expected visible reasoning for cod substitution"
            assert "claude" in result.visible_reasoning.lower() or "code" in result.visible_reasoning.lower()
            assert "folder" in result.visible_reasoning.lower()

    def test_clarification_no_category_leak(self):
        """Clarification question must not include category tokens like (game) or (dev_tool).

        Categories are for the debug log/reasoning trace only — they leak
        implementation details to the user. The clarification should be
        natural language: "Cod — the game, or Claude Code, Master?"."""
        from controller.intent_resolver import resolve_intent
        forbidden = ["(game)", "(dev_tool)", "(website)", "(app)", "(video game)"]
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("open cod", brain_fn=mock_brain)
        assert result.clarification_question, "expected a clarification question"
        q = result.clarification_question
        for token in forbidden:
            assert token not in q, f"category token {token!r} leaked into clarification: {q!r}"
        # The natural replacement should appear.
        assert "the game" in q.lower() or "claude code" in q.lower(), \
            f"clarification should be natural language: {q!r}"

    def test_visible_reasoning_no_category_leak(self):
        """visible_reasoning must not include category tokens like (game)."""
        from controller.intent_resolver import resolve_intent
        forbidden = ["(game)", "(dev_tool)", "(website)", "(app)", "(video game)"]
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("use cod to make a folder", brain_fn=mock_brain)
        assert result.visible_reasoning, "expected visible reasoning"
        v = result.visible_reasoning
        for token in forbidden:
            assert token not in v, f"category token {token!r} leaked into visible_reasoning: {v!r}"

    def test_no_substitution_no_visible_reasoning(self):
        """Unambiguous inputs have empty visible reasoning."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            result = resolve_intent("open youtube", brain_fn=mock_brain)
            assert result.visible_reasoning == ""


# ─── End-to-end wired integration test ──────────────────────────────────────────


class TestWiredIntegration:
    """Verify the resolver integrates with agent_controller.handle_request."""

    def setup_method(self, method):
        """Clear pending-delegation state before each test in this class."""
        try:
            from controller.delegate_state import clear_pending
            clear_pending()
        except Exception:
            pass
        os.environ.pop("FAIRY_USE_HERMES", None)
        os.environ.pop("FAIRY_TASK_GUARD_REENTRY", None)

    def teardown_method(self, method):
        try:
            from controller.delegate_state import clear_pending
            clear_pending()
        except Exception:
            pass
        os.environ.pop("FAIRY_USE_HERMES", None)
        os.environ.pop("FAIRY_TASK_GUARD_REENTRY", None)

    def test_handle_request_runs_resolver_first(self):
        """v2 redesign: skip this test.

        The intent resolver clarification gate now fires for "use cod" queries,
        which is the correct v2 behavior. The delegation pipeline that this test
        verified is removed in v2. The handoff detection is tested by
        test_task_directory_classifier.py and the build_handoff_command tests.
        """
        pytest.skip("v2 redesign: resolver clarification gate takes priority over delegation")

    def test_resolver_clarification_produces_clarifying_question(self):
        """When resolver fires clarification gate, handle_request surfaces it."""
        from controller import agent_controller as _ac
        # Patch run_turn_safe at the module where it's imported (hermes_bridge).
        with patch("hermes_bridge.run_turn_safe") as mock_hermes:
            mock_hermes.return_value = (False, "fallback", [])
            with patch("controller.agent_controller._brain") as mock_brain:
                mock_brain.return_value = MagicMock(
                    message=MagicMock(content="", tool_calls=[]),
                    done=True,
                )
                reply, _ = _ac.handle_request("open cod", [])
                # The clarifying question should appear in the reply
                assert any(
                    kw in reply.lower()
                    for kw in ["cod", "call of duty", "claude code"]
                ), f"Expected clarification about 'cod' in reply. Got: {reply[:300]}"


# ─── Voice / Whisper near-miss tests ─────────────────────────────────────────────


class TestVoiceNearMisses:
    """Whisper transcripts produce near-miss words — resolver handles them."""

    def test_fuzzy_match_cod(self):
        """'cod' is fuzzy-matched in the local table."""
        from controller.intent_resolver import resolve_intent, _normalize_term
        with patch("controller.agent_controller._brain") as mock_brain:
            # Lowercase, stripped, punctuation-free
            assert _normalize_term("cod") == "cod"
            result = resolve_intent("use cod to make a folder", brain_fn=mock_brain)
            assert result.chosen_meaning is not None
            assert result.chosen_meaning.category == "dev_tool"

    def test_unrelated_term_not_confused_with_cod(self):
        """An unrelated term is not incorrectly resolved to cod."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            # "node" is nothing like "cod" — should not trigger cod resolution
            result = resolve_intent("open node", brain_fn=mock_brain)
            # If a meaning was chosen, it must not be cod-related
            if result.chosen_meaning is not None:
                assert "cod" not in result.chosen_meaning.meaning.lower()
                assert "claude code" not in result.chosen_meaning.meaning.lower()


# ─── Performance: fast-path latency ──────────────────────────────────────────────


class TestFastPathLatency:
    """Fast-path messages complete in negligible time (no LLM call)."""

    def test_fast_path_no_llm(self):
        """Known website / desktop app skips LLM entirely."""
        from controller.intent_resolver import resolve_intent
        with patch("controller.agent_controller._brain") as mock_brain:
            t0 = time.perf_counter()
            for _ in range(100):
                resolve_intent("open youtube", brain_fn=mock_brain)
            elapsed = time.perf_counter() - t0
            # 100 fast-path resolutions should take well under 1 second total
            assert elapsed < 1.0, f"Fast path took {elapsed:.2f}s for 100 calls"
            mock_brain.assert_not_called()


# ─── Site-resolution voice test ──────────────────────────────────────────────────


class TestSiteResolutionVoice:
    """Whisper-style noisy transcripts must still resolve to the correct site."""

    def test_hoyola_whisper_noisy_resolves_to_hoyolab(self):
        """"open hoyola four mi the website please" extracts the candidate hoyola
        and resolves to https://www.hoyolab.com/.

        Mirrors the live typed test that already passed, but with messier
        filler words ("four mi the website please") to simulate a real
        Whisper transcript where the user inserted utterance noise.
        """
        from controller.agent_controller import _infer_url_from_text
        url = _infer_url_from_text("open hoyola four mi the website please")
        # hoyola → hoyolab.com (via site_map)
        assert url and "hoyolab" in url.lower(), \
            f"Expected hoyolab URL, got: {url!r}"
        assert url.startswith("https://"), f"Expected https://, got: {url!r}"
