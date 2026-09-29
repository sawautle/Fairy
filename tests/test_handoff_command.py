"""
test_handoff_command.py

Tests for the v2 handoff command construction in controller/claude_code_delegate.py.

The handoff is the structural guarantee that Fairy's spawn does NOT bypass
Claude Code's own permission system. These tests pin the exact argv that
build_handoff_command() returns and verify it never contains any of the
forbidden flags.

Forbidden flags (must NEVER appear in the returned argv):
  - -p, --print              → would skip the interactive TUI
  - --output-format json     → would turn it into a one-shot subprocess
  - --dangerously-skip-permissions
  - --allow-anything         → would let Claude Code run without prompts
  - --no-input               → would prevent user interaction
  - --non-interactive        → would disable Claude Code's prompt UI
"""
from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_THIS)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


# ─────────────────────────────────────────────────────────────────
# 1. build_handoff_command()
# ─────────────────────────────────────────────────────────────────
class TestBuildHandoffCommand:
    """build_handoff_command returns the exact argv for the interactive spawn."""

    def test_returns_list_of_two(self):
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("fix the bug")
        assert isinstance(argv, list)
        assert len(argv) == 2

    def test_first_arg_is_claude(self):
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("fix the bug")
        assert argv[0] == "claude"

    def test_second_arg_is_task_string(self):
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("fix the bug")
        assert argv[1] == "fix the bug"

    def test_task_with_whitespace_is_stripped(self):
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("  fix the bug  ")
        assert argv[1] == "fix the bug"

    def test_multiline_task_preserved(self):
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("fix the bug\nin agent_controller.py")
        assert argv[1] == "fix the bug\nin agent_controller.py"

    def test_empty_task_raises(self):
        from controller.claude_code_delegate import build_handoff_command
        try:
            build_handoff_command("")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_whitespace_only_task_raises(self):
        from controller.claude_code_delegate import build_handoff_command
        try:
            build_handoff_command("   \n\t  ")
            assert False, "expected ValueError"
        except ValueError:
            pass


# ─────────────────────────────────────────────────────────────────
# 2. Forbidden flags — the critical safety guarantee
# ─────────────────────────────────────────────────────────────────
class TestForbiddenFlags:
    """The argv from build_handoff_command must NEVER contain forbidden flags."""

    def test_no_dash_p_flag(self):
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("test")
        assert "-p" not in argv
        assert "--print" not in argv

    def test_no_output_format_flag(self):
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("test")
        assert "--output-format" not in argv
        # Also check inline form
        for arg in argv:
            assert not arg.startswith("--output-format=")

    def test_no_dangerously_skip_permissions(self):
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("test")
        assert "--dangerously-skip-permissions" not in argv
        for arg in argv:
            assert not arg.startswith("--dangerously-skip-permissions=")

    def test_no_allow_anything(self):
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("test")
        assert "--allow-anything" not in argv

    def test_no_no_input(self):
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("test")
        assert "--no-input" not in argv

    def test_no_non_interactive(self):
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("test")
        assert "--non-interactive" not in argv

    def test_only_two_args(self):
        """No extra flags, no -p, no --output-format, no bypass flags."""
        from controller.claude_code_delegate import build_handoff_command
        argv = build_handoff_command("test")
        # Just ["claude", "task"] — that's it.
        assert len(argv) == 2


# ─────────────────────────────────────────────────────────────────
# 3. is_handoff_safe_command() — defensive runtime check
# ─────────────────────────────────────────────────────────────────
class TestIsHandoffSafeCommand:
    """is_handoff_safe_command is the runtime safety guard."""

    def test_safe_command_returns_true(self):
        from controller.claude_code_delegate import (
            build_handoff_command,
            is_handoff_safe_command,
        )
        argv = build_handoff_command("fix the bug")
        safe, reason = is_handoff_safe_command(argv)
        assert safe is True
        assert reason == "ok"

    def test_empty_argv_rejected(self):
        from controller.claude_code_delegate import is_handoff_safe_command
        safe, reason = is_handoff_safe_command([])
        assert safe is False
        assert "empty" in reason.lower()

    def test_dash_p_flag_rejected(self):
        from controller.claude_code_delegate import is_handoff_safe_command
        safe, reason = is_handoff_safe_command(["claude", "-p", "fix the bug"])
        assert safe is False
        assert "-p" in reason

    def test_output_format_flag_rejected(self):
        from controller.claude_code_delegate import is_handoff_safe_command
        safe, reason = is_handoff_safe_command(
            ["claude", "fix", "--output-format", "json"]
        )
        assert safe is False
        assert "--output-format" in reason

    def test_dangerously_skip_permissions_rejected(self):
        from controller.claude_code_delegate import is_handoff_safe_command
        safe, reason = is_handoff_safe_command(
            ["claude", "--dangerously-skip-permissions", "fix"]
        )
        assert safe is False
        assert "dangerously-skip-permissions" in reason

    def test_allow_anything_rejected(self):
        from controller.claude_code_delegate import is_handoff_safe_command
        safe, reason = is_handoff_safe_command(
            ["claude", "--allow-anything", "fix"]
        )
        assert safe is False
        assert "allow-anything" in reason

    def test_no_input_rejected(self):
        from controller.claude_code_delegate import is_handoff_safe_command
        safe, reason = is_handoff_safe_command(
            ["claude", "--no-input", "fix"]
        )
        assert safe is False
        assert "--no-input" in reason

    def test_non_interactive_rejected(self):
        from controller.claude_code_delegate import is_handoff_safe_command
        safe, reason = is_handoff_safe_command(
            ["claude", "--non-interactive", "fix"]
        )
        assert safe is False
        assert "--non-interactive" in reason

    def test_inline_dash_p_equals_rejected(self):
        """--print= or -p= forms must also be caught."""
        from controller.claude_code_delegate import is_handoff_safe_command
        # --print= is a hypothetical form; the test is defensive.
        safe, reason = is_handoff_safe_command(["claude", "--print=compact", "fix"])
        # Either we catch it explicitly, or the binary path is non-claude.
        # The catch is `arg.startswith(bad + "=")` so this is caught.
        # But Claude Code's --print= form is unusual; verify the safety check
        # doesn't have a false negative for it.
        assert safe is False or "claude" in reason.lower()

    def test_non_claude_binary_rejected(self):
        from controller.claude_code_delegate import is_handoff_safe_command
        safe, reason = is_handoff_safe_command(["python", "fix the bug"])
        assert safe is False
        assert "claude" in reason.lower()

    def test_absolute_claude_path_accepted(self):
        from controller.claude_code_delegate import is_handoff_safe_command
        # Allow /usr/bin/claude or C:/Program Files/...
        safe, _ = is_handoff_safe_command(["/usr/local/bin/claude", "fix"])
        assert safe is True
        safe2, _ = is_handoff_safe_command(["C:\\Users\\me\\claude.exe", "fix"])
        assert safe2 is True
        safe3, _ = is_handoff_safe_command(["C:\\Users\\me\\claude.cmd", "fix"])
        assert safe3 is True


# ─────────────────────────────────────────────────────────────────
# 4. Explicit handoff trigger detection
# ─────────────────────────────────────────────────────────────────
class TestExplicitHandoffTrigger:
    """detect_repository_task's explicit-claude pattern is the handoff trigger."""

    def test_using_claude_triggers(self):
        from controller.claude_code_delegate import detect_repository_task
        assert detect_repository_task("fix the bug using claude") is True

    def test_use_claude_to_triggers(self):
        from controller.claude_code_delegate import detect_repository_task
        assert detect_repository_task("use claude to fix the bug") is True

    def test_use_claude_code_triggers(self):
        from controller.claude_code_delegate import detect_repository_task
        assert detect_repository_task("use claude code for this") is True

    def test_via_claude_triggers(self):
        from controller.claude_code_delegate import detect_repository_task
        assert detect_repository_task("via claude, organize my photos") is True

    def test_have_claude_triggers(self):
        from controller.claude_code_delegate import detect_repository_task
        assert detect_repository_task("have claude make a folder") is True

    def test_normal_chat_does_not_trigger(self):
        from controller.claude_code_delegate import detect_repository_task
        assert detect_repository_task("hello there") is False

    def test_repo_keyword_still_triggers(self):
        """Heuristic path: repo keywords without explicit claude mention."""
        from controller.claude_code_delegate import detect_repository_task
        # Heuristic detection — Claude Code is the right tool for this task.
        assert detect_repository_task("fix the bug in fairy") is True
