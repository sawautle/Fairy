#!/usr/bin/env python3
"""
Tests for choice_selector (controller/choice_selector.py).

Phase 2 tests for the interactive TTY choice selector.

Tests:
  1. run_choice_selector returns None on Ctrl+C (plain fallback)
  2. run_choice_selector returns correct answer for valid keypress
  3. run_choice_selector handles empty stdin gracefully (returns None)
  4. run_choice_selector with 1 option returns immediately (no selector needed)
  5. ChoiceResult dataclass has expected fields
  6. _plain_fallback_single returns ChoiceResult on valid input
  7. _plain_fallback returns tuple of lists on valid input
"""
from __future__ import annotations

import io
import os
import sys

import pytest

# Make controller/ importable
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


class TestChoiceResultDataclass:
    """ChoiceResult dataclass invariants."""

    def test_choice_result_fields(self):
        from controller.choice_selector import ChoiceResult
        cr = ChoiceResult(selected_index=2, selected_letter="C")
        assert cr.selected_index == 2
        assert cr.selected_letter == "C"


class TestPlainFallback:
    """Plain fallback path (used when prompt_toolkit is unavailable)."""

    def test_returns_correct_answer_on_valid_input(self, monkeypatch):
        """A valid letter keypress returns the matching option."""
        from controller.choice_selector import _plain_fallback
        monkeypatch.setattr(sys, "stdin", io.StringIO("B\n"))
        result = _plain_fallback(
            "Which module?", ["core", "controller", "tests"]
        )
        assert result is not None
        answers, indices = result
        assert answers == ["controller"]
        assert indices == [1]

    def test_returns_none_on_ctrl_c(self, monkeypatch):
        """Ctrl+C (EOF) returns None."""
        from controller.choice_selector import _plain_fallback
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # EOF
        result = _plain_fallback(
            "Which module?", ["core", "controller", "tests"]
        )
        assert result is None

    def test_single_option_returns_immediately(self, monkeypatch):
        """A single-option selector auto-selects."""
        from controller.choice_selector import run_choice_selector
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        # This should return immediately with the single option
        result = run_choice_selector(
            "Any questions?", ["Proceed"]
        )
        assert result is not None
        answers, indices = result
        assert answers == ["Proceed"]
        assert indices == [0]


class TestLetterShortcuts:
    """Letter shortcuts work in the plain fallback."""

    def test_letter_a_selects_first(self, monkeypatch):
        from controller.choice_selector import _plain_fallback
        monkeypatch.setattr(sys, "stdin", io.StringIO("A\n"))
        result = _plain_fallback("Pick one", ["Option 1", "Option 2"])
        assert result is not None
        answers, indices = result
        assert answers == ["Option 1"]

    def test_letter_c_selects_third(self, monkeypatch):
        from controller.choice_selector import _plain_fallback
        monkeypatch.setattr(sys, "stdin", io.StringIO("C\n"))
        result = _plain_fallback("Pick one", ["A", "B", "C"])
        assert result is not None
        answers, indices = result
        assert answers == ["C"]


class TestPlainFallbackSingle:
    """_plain_fallback_single wraps _plain_fallback correctly."""

    def test_returns_choiceresult(self, monkeypatch):
        from controller.choice_selector import _plain_fallback_single
        monkeypatch.setattr(sys, "stdin", io.StringIO("B\n"))
        result = _plain_fallback_single("Which?", ["X", "Y", "Z"])
        assert result is not None
        assert result.selected_index == 1
        assert result.selected_letter == "B"

    def test_returns_none_on_eof(self, monkeypatch):
        from controller.choice_selector import _plain_fallback_single
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        result = _plain_fallback_single("Which?", ["X", "Y"])
        assert result is None


class TestModuleLoadsCleanly:
    """The module loads and has expected public functions."""

    def test_module_imports(self):
        from controller.choice_selector import (
            run_choice_selector,
            ChoiceResult,
            _plain_fallback,
            _plain_fallback_single,
            _interactive_select,
        )
        assert callable(run_choice_selector)
        assert callable(_plain_fallback)
