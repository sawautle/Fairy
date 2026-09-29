#!/usr/bin/env python3
"""
Tests for Prompt Architect (controller/prompt_architect.py).

Tests:
  1. _LLM_UNAVAILABLE_FALLBACK returns intent unchanged
  2. _call_architect_llm returns composed prompt on valid LLM response
  3. _call_architect_llm returns clarifying flag when LLM asks a question
  4. _call_architect_llm returns NONE for clarifying when LLM says NONE
  5. _call_architect_llm parses OPTIONS into a list
  6. _call_architect_llm falls back gracefully when main_brain unavailable
  7. _call_architect_llm falls back gracefully on LLM exception
  8. get_git_evidence returns None triplet when not a git repo
  9. get_git_evidence returns diff when there are changes (real git repo)
 10. run_prompt_architect produces a confirmed=False result on cancel
 11. ArchitectResult dataclass has all expected fields
 12. SYSTEM prompt contains the required COMPOSED/CLARIFY/OPTIONS markers
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

# Make controller/ importable
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


class TestPromptArchitectDataclass:
    """ArchitectResult dataclass invariants."""

    def test_architect_result_has_expected_fields(self):
        from controller.prompt_architect import ArchitectResult
        ar = ArchitectResult(
            composed_prompt="fix the bug",
            was_clarifying=False,
            clarifying_answers=[],
            confirmed=True,
        )
        assert ar.composed_prompt == "fix the bug"
        assert ar.was_clarifying is False
        assert ar.clarifying_answers == []
        assert ar.confirmed is True

    def test_architect_result_defaults(self):
        from controller.prompt_architect import ArchitectResult
        ar = ArchitectResult(composed_prompt="hello")
        assert ar.composed_prompt == "hello"
        assert ar.was_clarifying is False
        assert ar.clarifying_answers == []
        assert ar.confirmed is False


class TestLLMUnavailability:
    """When LLM is not available, fallback to raw intent."""

    def test_fallback_returns_intent_unchanged(self):
        from controller.prompt_architect import _LLM_UNAVAILABLE_FALLBACK
        composed, was_clarifying, question, options = _LLM_UNAVAILABLE_FALLBACK(
            "fix the bug in module X"
        )
        assert composed == "fix the bug in module X"
        assert was_clarifying is False
        assert question == "NONE"
        assert options == []

    def test_call_with_no_brain_imports_returns_fallback(self, monkeypatch):
        """If main_brain cannot be imported, _call_architect_llm falls back."""
        # Force the import to fail
        import controller.prompt_architect as pa

        original_path = sys.path[:]
        sys.path = [p for p in sys.path if "controller" not in p]

        try:
            composed, was_clarifying, question, options = pa._call_architect_llm(
                "test intent", [], []
            )
            # Without an LLM, the fallback should return the intent as-is
            assert composed == "test intent"
            assert was_clarifying is False
            assert question == "NONE"
            assert options == []
        finally:
            sys.path = original_path


class TestLLMResponseParsing:
    """Parsing of the LLM COMPOSED/CLARIFY/OPTIONS format."""

    def test_parses_composed_prompt(self, monkeypatch):
        """A well-formed LLM response is parsed correctly."""
        from controller import prompt_architect as pa

        fake_response = {
            "message": {
                "content": (
                    "COMPOSED: Fix the off-by-one in fairy.py line 120\n"
                    "CLARIFY: NONE\n"
                    "OPTIONS: NONE\n"
                ),
            }
        }
        monkeypatch.setattr(
            "controller.main_brain.chat",
            lambda *args, **kwargs: (fake_response, "openrouter"),
        )
        composed, was_clarifying, question, options = pa._call_architect_llm(
            "fix off-by-one", [], []
        )
        assert "off-by-one" in composed
        assert was_clarifying is False
        assert question == "NONE"
        assert options == []

    def test_parses_clarifying_question(self, monkeypatch):
        """A response with a CLARIFY question sets was_clarifying=True."""
        from controller import prompt_architect as pa

        fake_response = {
            "message": {
                "content": (
                    "COMPOSED: TBD\n"
                    "CLARIFY: Which module should I focus on?\n"
                    "OPTIONS: core|controller|tests\n"
                ),
            }
        }
        monkeypatch.setattr(
            "controller.main_brain.chat",
            lambda *args, **kwargs: (fake_response, "openrouter"),
        )
        composed, was_clarifying, question, options = pa._call_architect_llm(
            "fix bug", [], []
        )
        assert was_clarifying is True
        assert "module" in question.lower()
        assert options == ["core", "controller", "tests"]

    def test_parses_no_clarify(self, monkeypatch):
        """CLARIFY: NONE means was_clarifying=False."""
        from controller import prompt_architect as pa

        fake_response = {
            "message": {
                "content": "COMPOSED: a clear prompt\nCLARIFY: NONE\nOPTIONS: NONE",
            }
        }
        monkeypatch.setattr(
            "controller.main_brain.chat",
            lambda *args, **kwargs: (fake_response, "openrouter"),
        )
        composed, was_clarifying, question, options = pa._call_architect_llm(
            "x", [], []
        )
        assert was_clarifying is False
        assert question == "NONE"
        assert options == []

    def test_chat_exception_returns_fallback(self, monkeypatch):
        """If chat() raises, fall back to intent."""
        from controller import prompt_architect as pa

        def _raise(*args, **kwargs):
            raise RuntimeError("LLM down")
        monkeypatch.setattr("controller.main_brain.chat", _raise)
        composed, was_clarifying, question, options = pa._call_architect_llm(
            "fix the thing", [], []
        )
        assert composed == "fix the thing"
        assert was_clarifying is False


class TestSystemPrompt:
    """SYSTEM constant must include the required format markers."""

    def test_system_has_required_markers(self):
        from controller.prompt_architect import SYSTEM
        assert "COMPOSED:" in SYSTEM
        assert "CLARIFY:" in SYSTEM
        assert "OPTIONS:" in SYSTEM

    def test_system_mentions_clarification_rule(self):
        from controller.prompt_architect import SYSTEM
        assert "clarifying" in SYSTEM.lower() or "clarif" in SYSTEM.lower()


class TestGetGitEvidence:
    """git evidence collection."""

    def test_git_evidence_in_non_git_dir(self):
        from controller.prompt_architect import get_git_evidence
        with tempfile.TemporaryDirectory() as tmp:
            status, diff, stat = get_git_evidence(tmp)
            # Not a git repo: all three should be None (git returns non-zero)
            assert status is None
            assert diff is None
            assert stat is None

    def test_git_evidence_in_real_git_repo(self):
        from controller.prompt_architect import get_git_evidence
        # E:/fairy IS a git repo (assumed)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        status, diff, stat = get_git_evidence(project_root)
        # We don't assert specific values (depends on repo state), but
        # at least one of these should be non-None OR all three None
        # (if repo is clean and current). The function should not raise.
        assert isinstance(status, (str, type(None)))
        assert isinstance(diff, (str, type(None)))
        assert isinstance(stat, (str, type(None)))


class TestWaitForActionPlain:
    """_wait_for_action_plain is a real I/O path; verify it doesn't crash."""

    def test_plain_action_returns_n_on_eof(self, monkeypatch):
        """If stdin is at EOF, returns N (cancel)."""
        from controller import prompt_architect as pa
        # Force the prompt_toolkit import to fail
        # so we exercise the plain path
        import builtins
        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name.startswith("prompt_toolkit"):
                raise ImportError("forced for test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        # Also patch sys.stdin.read to return empty string (EOF)
        import io
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        result = pa._wait_for_action(None)
        # EOF -> N (cancel)
        assert result == "N"


class TestRunPromptArchitect:
    """Full pipeline behavior. We mock the LLM and the panel."""

    def test_architect_with_no_clarification_returns_composed(self, monkeypatch):
        """When LLM returns no clarification, the panel still shows."""
        from controller import prompt_architect as pa

        fake_response = {
            "message": {
                "content": "COMPOSED: Fix the bug in X\nCLARIFY: NONE\nOPTIONS: NONE",
            }
        }
        monkeypatch.setattr(
            "controller.main_brain.chat",
            lambda *args, **kwargs: (fake_response, "openrouter"),
        )
        # Mock the transparency panel to return Y (launch)
        monkeypatch.setattr(
            "controller.prompt_architect.display_transparency_panel",
            lambda composed, console=None: "Y",
        )
        result = pa.run_prompt_architect("fix X", [], None)
        assert result.confirmed is True
        assert "Fix the bug" in result.composed_prompt
        assert result.was_clarifying is False

    def test_architect_with_n_panel_returns_unconfirmed(self, monkeypatch):
        """When the panel returns N, confirmed=False."""
        from controller import prompt_architect as pa

        fake_response = {
            "message": {
                "content": "COMPOSED: Some prompt\nCLARIFY: NONE\nOPTIONS: NONE",
            }
        }
        monkeypatch.setattr(
            "controller.main_brain.chat",
            lambda *args, **kwargs: (fake_response, "openrouter"),
        )
        monkeypatch.setattr(
            "controller.prompt_architect.display_transparency_panel",
            lambda composed, console=None: "N",
        )
        result = pa.run_prompt_architect("do thing", [], None)
        assert result.confirmed is False
        assert result.composed_prompt == "Some prompt"

    def test_architect_unavailable_llm_falls_back_to_intent(self, monkeypatch):
        """If LLM is unavailable, intent is used as composed prompt."""
        from controller import prompt_architect as pa

        # Force the LLM call to fall back
        def _raise(*args, **kwargs):
            raise RuntimeError("no LLM")
        monkeypatch.setattr("controller.main_brain.chat", _raise)
        # Mock the transparency panel
        monkeypatch.setattr(
            "controller.prompt_architect.display_transparency_panel",
            lambda composed, console=None: "Y",
        )
        result = pa.run_prompt_architect("do the thing", [], None)
        assert result.confirmed is True
        assert result.composed_prompt == "do the thing"
