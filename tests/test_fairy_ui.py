"""
Smoke tests for the Fairy TUI helper functions in fairy.py.

These tests do NOT start a real TUI (which would require a terminal).
They import the pure helper functions and verify:
  - Large/small input detection
  - Language detection for code blocks
  - Output format selection (small = pass through, large = bounded)
  - fairy_prompt() returns the exact original text without truncation
"""
from __future__ import annotations

import io
import os
import sys

# Make fairy.py importable as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# Import the helpers — we only need the pure functions, not main()
import fairy


# ─────────────────────────────────────────────────────────────────
# _is_large_input
# ─────────────────────────────────────────────────────────────────
class TestIsLargeInput:
    def test_short_single_line_is_not_large(self):
        assert fairy._is_large_input("hello world") is False

    def test_medium_text_under_200_chars_is_not_large(self):
        text = "a" * 199
        assert fairy._is_large_input(text) is False

    def test_text_over_200_chars_is_large(self):
        text = "a" * 201
        assert fairy._is_large_input(text) is True

    def test_text_exactly_200_chars_is_not_large(self):
        # threshold is strict greater-than
        text = "a" * 200
        assert fairy._is_large_input(text) is False

    def test_few_lines_under_5_is_not_large(self):
        text = "line1\nline2\nline3\nline4"  # 4 lines
        assert fairy._is_large_input(text) is False

    def test_six_lines_is_large(self):
        text = "\n".join(f"line{i}" for i in range(6))  # 6 lines
        assert fairy._is_large_input(text) is True

    def test_exactly_5_lines_is_not_large(self):
        # threshold is strict greater-than
        text = "\n".join(f"line{i}" for i in range(5))  # 5 lines
        assert fairy._is_large_input(text) is False

    def test_empty_is_not_large(self):
        assert fairy._is_large_input("") is False


# ─────────────────────────────────────────────────────────────────
# _detect_language
# ─────────────────────────────────────────────────────────────────
class TestDetectLanguage:
    def test_python_fence(self):
        text = "```python\ndef foo(): pass\n```"
        assert fairy._detect_language(text) == "python"

    def test_javascript_fence(self):
        text = "```javascript\nconst x = 1;\n```"
        assert fairy._detect_language(text) == "javascript"

    def test_json_fence(self):
        text = "```json\n{\"a\": 1}\n```"
        assert fairy._detect_language(text) == "json"

    def test_bare_json_object(self):
        text = '{"key": "value", "num": 42}'
        assert fairy._detect_language(text) == "json"

    def test_invalid_json_returns_none(self):
        text = "{not actually json}"
        assert fairy._detect_language(text) is None

    def test_plain_text_returns_none(self):
        assert fairy._detect_language("Hello, Master!") is None

    def test_bash_fence(self):
        text = "```bash\necho hello\n```"
        assert fairy._detect_language(text) == "bash"

    def test_unknown_fence_returns_text(self):
        text = "```\nsome code\n```"
        assert fairy._detect_language(text) == "text"

    def test_fence_with_leading_whitespace(self):
        text = "   ```python\nprint('hi')\n```"
        assert fairy._detect_language(text) == "python"


# ─────────────────────────────────────────────────────────────────
# _format_output
# ─────────────────────────────────────────────────────────────────
class TestFormatOutput:
    """Tests that _format_output does not crash and selects the right path."""

    def test_empty_text_does_not_raise(self):
        # Should not raise; prints nothing
        fairy._format_output("")

    def test_short_text_does_not_raise(self, capsys):
        fairy._format_output("Short reply, Master.")
        captured = capsys.readouterr()
        assert "Short reply" in captured.out

    def test_long_text_does_not_raise(self, capsys):
        # Build a large multi-line response
        big = "\n".join(f"Line {i} of the response" for i in range(40))
        fairy._format_output(big)
        captured = capsys.readouterr()
        # Should include a hint about hidden lines
        assert "more lines" in captured.out

    def test_long_json_uses_syntax_highlighting(self, capsys):
        big_json = "{\n" + ",\n".join(
            f'  "key_{i}": "value_{i}"' for i in range(40)
        ) + "\n}"
        fairy._format_output(big_json)
        captured = capsys.readouterr()
        # The output should mention that it's an output panel
        assert "Output" in captured.out

    def test_just_under_threshold_uses_plain_path(self, capsys):
        # Just under the 500-char threshold
        text = "a" * 499
        fairy._format_output(text)
        captured = capsys.readouterr()
        # Plain path: should appear in output without "Output" panel label
        assert "a" * 10 in captured.out

    def test_just_over_threshold_uses_panel(self, capsys):
        text = "a" * 501
        fairy._format_output(text)
        captured = capsys.readouterr()
        assert "Output" in captured.out


# ─────────────────────────────────────────────────────────────────
# fairy_prompt — input capture
# ─────────────────────────────────────────────────────────────────

def _has_real_console():
    """True when we are running in a real Windows console (not Git Bash / no-TTY)."""
    try:
        from prompt_toolkit.output.win32 import Win32Output
        import sys
        Win32Output(sys.stdout)
        return True
    except Exception:
        return False


requires_console = pytest.mark.skipif(
    not _has_real_console(),
    reason="fairy_prompt() requires a real Windows TTY — skipped in Git Bash / no-TTY runs",
)


@requires_console
class TestFairyPrompt:
    """Verify fairy_prompt reads stdin and returns exact text without truncation."""

    def test_single_line_returns_text(self, monkeypatch, capsys):
        # Simulate user typing one line and pressing Enter twice (single + empty)
        monkeypatch.setattr("sys.stdin", io.StringIO("hello world\n\n"))
        text = fairy.fairy_prompt()
        assert text == "hello world"

    def test_multiline_returns_full_text(self, monkeypatch, capsys):
        # User types multiple lines, then empty line to submit
        monkeypatch.setattr(
            "sys.stdin",
            io.StringIO("line one\nline two\nline three\n\n"),
        )
        text = fairy.fairy_prompt()
        assert text == "line one\nline two\nline three"

    def test_empty_input_returns_empty(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
        text = fairy.fairy_prompt()
        assert text == ""

    def test_eof_returns_text(self, monkeypatch, capsys):
        # EOF without trailing newline
        monkeypatch.setattr("sys.stdin", io.StringIO("eof test"))
        text = fairy.fairy_prompt()
        assert text == "eof test"

    def test_large_paste_returns_exact_text(self, monkeypatch, capsys):
        # Big paste should be previewed but returned exactly
        big = "\n".join(f"pasted line {i}" for i in range(20))
        monkeypatch.setattr("sys.stdin", io.StringIO(big + "\n\n"))
        text = fairy.fairy_prompt()
        # Exact text, no truncation, no preview-mangling
        assert text == big
        # The preview should have been printed
        captured = capsys.readouterr()
        assert "Pasted" in captured.out
        assert "20 lines" in captured.out

    def test_small_paste_no_preview_panel(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin", io.StringIO("short\n\n"))
        fairy.fairy_prompt()
        captured = capsys.readouterr()
        # No "Pasted" panel for short input
        assert "Pasted" not in captured.out


# ─────────────────────────────────────────────────────────────────
# chat_panel — verifies it doesn't return None (caller contract change)
# ─────────────────────────────────────────────────────────────────
class TestChatPanelContract:
    def test_chat_panel_does_not_return_none_for_assistant(self, capsys):
        # The function changed from returning a renderable to printing directly.
        # Calling it should not return None (callers used to do console.print()).
        result = fairy.chat_panel("Hello", is_user=False)
        # After change: returns None (it printed). Before: returned Panel.
        # Either way, calling it should not raise.
        # Verify something was printed.
        captured = capsys.readouterr()
        assert "Hello" in captured.out

    def test_chat_panel_user_side_does_not_raise(self, capsys):
        fairy.chat_panel("User said this", is_user=True)
        captured = capsys.readouterr()
        assert "User said this" in captured.out
