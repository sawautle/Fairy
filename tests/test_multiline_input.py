#!/usr/bin/env python3
"""
Tests for controller/terminal_input.py — multi-line terminal input
extracted from main.py so it can be tested without dragging in numpy /
ollama / discord / etc.

Covers:
  - read_line() returns a string.
  - Shift+Enter / Alt+Enter / Ctrl+J / Ctrl+L key bindings are configured.
  - continuation_prompt is set (visual affordance for multi-line).
  - multiline=True is set (Enter submits, Shift+Enter wraps).
  - Graceful fallback when prompt_toolkit is unavailable.
  - Bangla and UTF-8 characters pass through unchanged.
  - Multi-line strings (including newlines) survive the pipeline.
  - KeyboardInterrupt / EOFError propagate to caller.
  - Empty input returns empty string.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure the controller package is on sys.path.
_controller_dir = os.path.dirname(os.path.abspath(__file__))  # tests/
_project_dir = os.path.dirname(_controller_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from controller import terminal_input as ti


class TestKeyBindings:
    """Verify the key bindings are set up correctly.

    We test the key-bindings factory directly instead of building a real
    PromptSession, because PromptSession needs a TTY and the bindings
    factory is pure (takes no args, returns a KeyBindings object).
    """

    def _get_make_keybindings(self):
        """Reach into the closure of _initialize to grab _make_keybindings.
        This is the only public way to test the binding wiring without
        triggering PromptSession (which needs a TTY)."""
        from controller import terminal_input as ti
        # Trigger _initialize just enough to populate the closure.
        # We catch the TTY error by re-importing the module with a fake
        # PT to inspect the function.
        # Easiest path: re-run _initialize with a TTY stub via mocking.
        from unittest.mock import patch, MagicMock

        # Create a fake prompt_toolkit module that gives us a working
        # KeyBindings factory without needing a TTY.
        fake_kb_module = MagicMock()
        fake_kb_module.KeyBindings = MagicMock()
        fake_kb_instance = MagicMock()
        fake_kb_instance.add = MagicMock()
        fake_kb_module.KeyBindings.return_value = fake_kb_instance

        fake_pt = MagicMock()
        fake_pt.key_binding.key_bindings = fake_kb_module
        fake_pt.PromptSession = MagicMock()
        fake_pt.history.FileHistory = MagicMock()
        fake_pt.history.InMemoryHistory = MagicMock()
        fake_pt.styles.Style = MagicMock()

        # We can't easily get _make_keybindings without running
        # _initialize. Instead, use a direct test: we verify the closure
        # is correct by running _initialize and inspecting the resulting
        # PromptSession call args.
        return ti

    def test_alt_enter_and_ctrl_j_bound_in_keybindings(self):
        """We verify the binding wiring by running _initialize with a
        mock PromptSession and inspecting the key_bindings arg.
        The actual handler logic isn't tested (it's trivial — just
        insert_text("\\n")) — only that the keys are bound."""
        from controller import terminal_input as ti
        from unittest.mock import MagicMock, patch

        captured = {}

        class _FakePromptSession:
            def __init__(self, **kw):
                # Capture the key_bindings object so we can inspect it.
                captured["kb"] = kw.get("key_bindings")
                # And the multiline / wrap_lines flags.
                captured["multiline"] = kw.get("multiline")
                captured["wrap_lines"] = kw.get("wrap_lines")
                captured["continuation"] = kw.get("prompt_continuation")

        # Reset state.
        ti._PT_SESSION = None
        ti._init_done = False

        with patch("prompt_toolkit.PromptSession", _FakePromptSession):
            with patch("prompt_toolkit.history.FileHistory", MagicMock()):
                with patch("prompt_toolkit.history.InMemoryHistory", MagicMock()):
                    with patch("prompt_toolkit.styles.Style", MagicMock()):
                        # _initialize also needs KeyBindings to work.
                        # We use the real one but inspect the resulting
                        # kb via the captured kwarg.
                        from prompt_toolkit.key_binding import KeyBindings
                        ti._initialize()

        assert "kb" in captured
        kb = captured["kb"]
        assert kb is not None
        # kb is a real KeyBindings object now (we let it pass through).
        names = []
        for binding in kb._bindings:
            for k in binding.keys:
                names.append(str(k))
        # Alt+Enter is bound as escape + enter (two separate keys).
        assert "Keys.Escape" in names, (
            f"Alt+Enter should be bound; got: {names}"
        )
        # And the second key in the sequence must be Enter.
        # We check that ANY binding has (Keys.Escape, Keys.ControlM).
        found_alt_enter = False
        for binding in kb._bindings:
            keys = [str(k) for k in binding.keys]
            if "Keys.Escape" in keys and "Keys.ControlM" in keys:
                found_alt_enter = True
                break
        assert found_alt_enter, (
            f"Alt+Enter (Escape+Enter) sequence should be bound; got: {names}"
        )
        # Ctrl+J
        assert "Keys.ControlJ" in names, (
            f"Ctrl+J should be bound; got: {names}"
        )
        # Ctrl+L
        assert "Keys.ControlL" in names, (
            f"Ctrl+L should be bound; got: {names}"
        )

    def test_session_uses_multiline(self):
        from controller import terminal_input as ti
        captured = {}

        class _FakePromptSession:
            def __init__(self, **kw):
                captured["multiline"] = kw.get("multiline")
                captured["wrap_lines"] = kw.get("wrap_lines")
                captured["continuation"] = kw.get("prompt_continuation")

        ti._PT_SESSION = None
        ti._init_done = False
        with patch("prompt_toolkit.PromptSession", _FakePromptSession):
            with patch("prompt_toolkit.history.FileHistory", MagicMock()):
                with patch("prompt_toolkit.history.InMemoryHistory", MagicMock()):
                    with patch("prompt_toolkit.styles.Style", MagicMock()):
                        ti._initialize()

        assert captured["multiline"] is True, (
            "PromptSession must be multiline=True so Enter submits and "
            "Alt+Enter wraps"
        )
        assert captured["wrap_lines"] is True
        assert captured["continuation"] is not None

    def test_continuation_prompt_is_distinct_from_main(self):
        """The continuation prompt (shown when buffer spans multiple lines)
        must be visibly different from the main prompt — that's the
        visual affordance the Master relies on."""
        from controller import terminal_input as ti
        assert ti._continuation_prompt is not None
        assert ti._continuation_prompt != ti._main_prompt
        # Continuation is the visible "  … " affordance, different from
        # the main "you› " prompt.
        assert ti._main_prompt.strip() != ti._continuation_prompt.strip()
        # Continuation should NOT be empty — that's the whole point of
        # the affordance.
        assert ti._continuation_prompt.strip() != ""


class TestReadLineBehavior:
    """Test read_line() with mocked PromptSession."""

    def test_returns_string(self):
        from controller import terminal_input as ti
        fake_session = MagicMock()
        fake_session.prompt.return_value = "hello world"
        with patch.object(ti, "_PT_SESSION", fake_session):
            result = ti.read_line()
        assert isinstance(result, str)
        assert result == "hello world"

    def test_strips_trailing_newline_only(self):
        """Internal newlines (from Shift+Enter) must survive."""
        from controller import terminal_input as ti
        fake_session = MagicMock()
        # Simulate multiline buffer content with a trailing newline.
        fake_session.prompt.return_value = "hello\nworld\n"
        with patch.object(ti, "_PT_SESSION", fake_session):
            result = ti.read_line()
        # Should strip only trailing newline, not internal ones.
        assert result == "hello\nworld"
        assert "\n" in result  # internal newline preserved

    def test_no_trailing_newline_unchanged(self):
        """If there's no trailing newline (e.g. user pressed Enter on
        an empty line), the text is unchanged."""
        from controller import terminal_input as ti
        fake_session = MagicMock()
        fake_session.prompt.return_value = "no trailing"
        with patch.object(ti, "_PT_SESSION", fake_session):
            result = ti.read_line()
        assert result == "no trailing"

    def test_bangla_passthrough(self):
        """Bangla characters must survive the pipeline unchanged."""
        from controller import terminal_input as ti
        bangla_text = "আমি কিভাবে করব?"
        fake_session = MagicMock()
        fake_session.prompt.return_value = bangla_text
        with patch.object(ti, "_PT_SESSION", fake_session):
            result = ti.read_line()
        assert result == bangla_text

    def test_multiline_bangla_passthrough(self):
        """Multi-line Bangla text survives the pipeline."""
        from controller import terminal_input as ti
        multiline_bangla = "আমি কিভাবে করব?\nতুমি কি জানো?"
        fake_session = MagicMock()
        fake_session.prompt.return_value = multiline_bangla
        with patch.object(ti, "_PT_SESSION", fake_session):
            result = ti.read_line()
        assert result == multiline_bangla

    def test_emoji_and_utf8_passthrough(self):
        """Emoji and other UTF-8 characters survive unchanged."""
        from controller import terminal_input as ti
        emoji_text = "Show me 🧚✨ fairy sprites and 日本!"
        fake_session = MagicMock()
        fake_session.prompt.return_value = emoji_text
        with patch.object(ti, "_PT_SESSION", fake_session):
            result = ti.read_line()
        assert result == emoji_text

    def test_arabic_and_korean_passthrough(self):
        """Mixed RTL and CJK characters survive unchanged."""
        from controller import terminal_input as ti
        mixed = "مرحبا 你好 🇧🇩"
        fake_session = MagicMock()
        fake_session.prompt.return_value = mixed
        with patch.object(ti, "_PT_SESSION", fake_session):
            result = ti.read_line()
        assert result == mixed

    def test_empty_input_returns_empty_string(self):
        """If user just hits Enter, result is '' (not None, not crash)."""
        from controller import terminal_input as ti
        fake_session = MagicMock()
        fake_session.prompt.return_value = ""
        with patch.object(ti, "_PT_SESSION", fake_session):
            result = ti.read_line()
        assert result == ""

    def test_ctrl_c_propagates_keyboard_interrupt(self):
        from controller import terminal_input as ti
        fake_session = MagicMock()
        fake_session.prompt.side_effect = KeyboardInterrupt()
        with patch.object(ti, "_PT_SESSION", fake_session):
            with pytest.raises(KeyboardInterrupt):
                ti.read_line()

    def test_ctrl_d_propagates_eoferror(self):
        from controller import terminal_input as ti
        fake_session = MagicMock()
        fake_session.prompt.side_effect = EOFError()
        with patch.object(ti, "_PT_SESSION", fake_session):
            with pytest.raises(EOFError):
                ti.read_line()


class TestFallbackMode:
    """When prompt_toolkit is unavailable, read_line falls back to plain
    input() and logs a warning to stderr (one-time)."""

    def test_fallback_returns_input(self):
        from controller import terminal_input as ti
        # Reset warning state.
        ti.read_line._warned = False  # type: ignore
        with patch.object(ti, "_PT_SESSION", None):
            with patch("builtins.input", return_value="fallback text") as mock_in:
                with patch("sys.stderr"):
                    result = ti.read_line()
        assert result == "fallback text"

    def test_fallback_uses_custom_prompt(self):
        from controller import terminal_input as ti
        ti.read_line._warned = True  # skip warning
        with patch.object(ti, "_PT_SESSION", None):
            with patch("builtins.input", return_value="x") as mock_in:
                with patch("sys.stderr"):
                    ti.read_line("custom> ")
        assert mock_in.called
        assert mock_in.call_args[0][0] == "custom> "

    def test_fallback_warns_once(self, capsys):
        """The fallback warning is printed only on the first call.
        Subsequent calls use plain input() without repeating the warning.

        Uses pytest's capsys fixture to capture stderr."""
        from controller import terminal_input as ti
        # Reset state — del the attribute so hasattr() returns False again,
        # which is what the code checks (not == False).
        try:
            del ti.read_line._warned
        except AttributeError:
            pass
        ti._init_done = False
        ti._PT_SESSION = None

        with patch.object(ti, "_initialize", lambda: None):
            with patch("builtins.input", return_value="x"):
                # First call — warning printed to stderr.
                ti.read_line()
                captured1 = capsys.readouterr()
                # Second call — no warning.
                ti.read_line()
                captured2 = capsys.readouterr()

        assert "prompt_toolkit" in captured1.err.lower(), (
            f"First call should print the prompt_toolkit warning to stderr; "
            f"got stderr: {captured1.err!r}"
        )
        assert "prompt_toolkit" not in captured2.err.lower(), (
            f"Second call should NOT print the warning; "
            f"got stderr: {captured2.err!r}"
        )


class TestIntegrationWithMainLoop:
    """Verify the read_line integration into the main REPL loop logic —
    the same way main.py uses it. We don't actually run main.main()
    because that triggers VRAM orchestration; we just verify the call
    signature works."""

    def test_read_line_signature(self):
        """read_line() takes an optional prompt and returns a string."""
        from controller import terminal_input as ti
        fake_session = MagicMock()
        fake_session.prompt.return_value = "ok"
        with patch.object(ti, "_PT_SESSION", fake_session):
            result = ti.read_line()
        assert isinstance(result, str)
        assert result == "ok"
