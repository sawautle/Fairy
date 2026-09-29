#!/usr/bin/env python3
"""
Tests for Hermes as the authoritative agent for action requests.

Verifies:
  1. Ambiguous "open X" requests (could be app or website) route to Hermes.
  2. Known websites still use the fast browser path.
  3. Application launch requests reach Hermes for reasoning.
  4. Hermes has guidance on app vs website distinction in ephemeral prompt.
  5. Fast path for deterministic actions (math, time, system) still works.
  6. Hermes graceful fallback when unavailable.
  7. Tool verification: successful tool execution reports success.
  8. Tool verification: failed tool execution reports failure honestly.
"""
from __future__ import annotations

import pytest


class TestHermesRoutingForActions:
    """Tests for Hermes as the authoritative agent for action requests."""

    # ── 1. Ambiguous "open X" requests route to Hermes ────────────────────────

    def test_open_calculator_is_app_action(self):
        """'open calculator' should be classified as 'app_action' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open calculator")
        assert result == "app_action", (
            f"Expected 'app_action' for 'open calculator', got '{result}'. "
            "Known desktop apps should route to Hermes for action."
        )

    def test_open_notepad_is_app_action(self):
        """'open notepad' should be classified as 'app_action' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open notepad")
        assert result == "app_action", (
            f"Expected 'app_action' for 'open notepad', got '{result}'. "
            "Known desktop apps should route to Hermes for action."
        )

    def test_open_steam_is_app_action(self):
        """'open steam' should be classified as 'app_action' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open steam")
        assert result == "app_action", (
            f"Expected 'app_action' for 'open steam', got '{result}'. "
            "Known desktop apps should route to Hermes for action."
        )

    def test_open_discord_is_browser(self):
        """'open discord' should be classified as 'browser' (discord.com is a known website)."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open discord")
        assert result == "browser", (
            f"Expected 'browser' for 'open discord', got '{result}'. "
            "Discord.com is a known website."
        )

    def test_open_wordpad_is_app_action(self):
        """'open wordpad' should be classified as 'app_action' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open wordpad")
        assert result == "app_action", (
            f"Expected 'app_action' for 'open wordpad', got '{result}'"
        )

    def test_open_vlc_is_app_action(self):
        """'open vlc' should be classified as 'app_action' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open vlc")
        assert result == "app_action", (
            f"Expected 'app_action' for 'open vlc', got '{result}'"
        )

    # ── 2. Known websites still use fast browser path ─────────────────────────

    def test_open_youtube_is_browser(self):
        """'open youtube' should be classified as 'browser' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open youtube")
        assert result == "browser", (
            f"Expected 'browser' for 'open youtube', got '{result}'. "
            "YouTube is a known website and should use fast browser path."
        )

    def test_open_reddit_is_browser(self):
        """'open reddit' should be classified as 'browser' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open reddit")
        assert result == "browser", (
            f"Expected 'browser' for 'open reddit', got '{result}'"
        )

    def test_open_github_is_browser(self):
        """'open github' should be classified as 'browser' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open github")
        assert result == "browser", (
            f"Expected 'browser' for 'open github', got '{result}'"
        )

    # ── New action verbs: start, launch, run, close ──────────────────────────

    def test_start_steam_is_app_action(self):
        """'start steam' should be classified as 'app_action'."""
        from controller import agent_controller as ac

        result = ac._fast_intent("start steam")
        assert result == "app_action", (
            f"Expected 'app_action' for 'start steam', got '{result}'"
        )

    def test_launch_discord_is_app_action(self):
        """'launch discord' should be classified as 'app_action'."""
        from controller import agent_controller as ac

        result = ac._fast_intent("launch discord")
        assert result == "app_action", (
            f"Expected 'app_action' for 'launch discord', got '{result}'"
        )

    def test_run_notepad_is_app_action(self):
        """'run notepad' should be classified as 'app_action'."""
        from controller import agent_controller as ac

        result = ac._fast_intent("run notepad")
        assert result == "app_action", (
            f"Expected 'app_action' for 'run notepad', got '{result}'"
        )

    def test_close_chrome_is_app_action(self):
        """'close chrome' should be classified as 'app_action'."""
        from controller import agent_controller as ac

        result = ac._fast_intent("close chrome")
        assert result == "app_action", (
            f"Expected 'app_action' for 'close chrome', got '{result}'"
        )

    def test_kill_steam_is_app_action(self):
        """'kill steam' should be classified as 'app_action'."""
        from controller import agent_controller as ac

        result = ac._fast_intent("kill steam")
        assert result == "app_action", (
            f"Expected 'app_action' for 'kill steam', got '{result}'"
        )

    def test_open_google_is_browser(self):
        """'open google' should be classified as 'browser' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open google")
        assert result == "browser", (
            f"Expected 'browser' for 'open google', got '{result}'"
        )

    def test_open_http_url_is_browser(self):
        """'open https://...' should be classified as 'browser' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open https://example.com")
        assert result == "browser", (
            f"Expected 'browser' for explicit URL, got '{result}'"
        )

    def test_open_www_url_is_browser(self):
        """'open www.example.com' should be classified as 'browser' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("open www.example.com")
        assert result == "browser", (
            f"Expected 'browser' for www URL, got '{result}'"
        )

    # ── 3. Other action requests still route correctly ─────────────────────────

    def test_go_to_youtube_is_browser(self):
        """'go to youtube' should be classified as 'browser' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("go to youtube")
        assert result == "browser", (
            f"Expected 'browser' for 'go to youtube', got '{result}'"
        )

    def test_navigate_to_reddit_is_browser(self):
        """'navigate to reddit' should be classified as 'browser' intent."""
        from controller import agent_controller as ac

        result = ac._fast_intent("navigate to reddit")
        assert result == "browser", (
            f"Expected 'browser' for 'navigate to reddit', got '{result}'"
        )

    # ── 4. Fast path intents still work correctly ─────────────────────────────

    def test_what_time_is_time_intent(self):
        """'what time is it' should use the time fast path."""
        from controller import agent_controller as ac

        result = ac._fast_intent("what time is it")
        assert result == "time", (
            f"Expected 'time' for 'what time is it', got '{result}'"
        )

    def test_system_monitor_is_system_intent(self):
        """'show my cpu usage' should use the system fast path."""
        from controller import agent_controller as ac

        result = ac._fast_intent("show my cpu usage")
        assert result == "system", (
            f"Expected 'system' for 'show my cpu usage', got '{result}'"
        )


class TestKnownWebsitesSet:
    """Tests for the _KNOWN_WEBSITES set."""

    def test_youtube_in_known_websites(self):
        """YouTube should be in the known websites set."""
        from controller import agent_controller as ac

        assert "youtube" in ac._KNOWN_WEBSITES, (
            "YouTube should be a known website"
        )

    def test_calculator_not_in_known_websites(self):
        """Calculator should NOT be in the known websites set."""
        from controller import agent_controller as ac

        assert "calculator" not in ac._KNOWN_WEBSITES, (
            "Calculator should be ambiguous, not a known website"
        )

    def test_steam_not_in_known_websites(self):
        """Steam should NOT be in the known websites set."""
        from controller import agent_controller as ac

        assert "steam" not in ac._KNOWN_WEBSITES, (
            "Steam should be ambiguous, not a known website"
        )


class TestHermesFallback:
    """Tests for Hermes graceful fallback when unavailable."""

    def test_ambiguous_falls_back_to_browser_in_code(self):
        """Verify the fallback code path exists for ambiguous intents."""
        # Read the actual source file to verify the changes
        import os
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "controller", "agent_controller.py"
        )
        with open(source_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()

        assert 'fast_intent == "ambiguous"' in source or "fast_intent == 'ambiguous'" in source, (
            "handle_request should have fallback logic for ambiguous intents"
        )
        assert "AMBIGUOUS_FALLBACK" in source or "hermes_unavailable" in source, (
            "handle_request should fall back to browser when Hermes fails"
        )

    def test_fast_deterministic_paths_not_affected(self):
        """Math and unit conversion should not be affected by Hermes routing."""
        from controller import agent_controller as ac

        # Math
        result = ac._try_fast_math("what is 5 plus 3")
        assert result is not None, "Math should still work"

        # Unit conversion
        result = ac._try_unit_conversion("convert 10 km to miles")
        assert result is not None, "Unit conversion should still work"


class TestToolVerification:
    """Tests for tool result verification behavior."""

    def test_looks_like_failure_detects_error_status(self):
        """_looks_like_failure should detect error status in tool results."""
        from controller import agent_controller as ac

        # Error status
        assert ac._looks_like_failure("test", {"status": "error"}) is True

        # Error in content
        assert ac._looks_like_failure("test", {"error": "something failed"}) is True

        # Ok status should not be failure
        assert ac._looks_like_failure("test", {"status": "ok"}) is False
        assert ac._looks_like_failure("test", {"ok": True}) is False

    def test_looks_like_failure_detects_failure_hints(self):
        """_looks_like_failure should detect failure hints in string results."""
        from controller import agent_controller as ac

        # Contains "error"
        assert ac._looks_like_failure("test", "there was an error") is True

        # Contains "failed"
        assert ac._looks_like_failure("test", "operation failed") is True

        # Contains "timeout"
        assert ac._looks_like_failure("test", "connection timeout") is True

        # Success message should not be failure
        assert ac._looks_like_failure("test", "operation completed successfully") is False
        assert ac._looks_like_failure("test", "calculator opened") is False


class TestDispatchToolRemap:
    """Tests to ensure dispatch tool remap still works for planner/legacy cases."""

    def test_open_app_remap_to_computer_control(self, monkeypatch):
        """_dispatch_tool should remap open_app to computer_control."""
        from controller import agent_controller as ac

        captured = {}

        def mock_computer_control(args):
            captured["args"] = args
            return '{"status": "ok"}'

        monkeypatch.setattr(ac, "computer_control", mock_computer_control)

        result = ac._dispatch_tool("open_app", {"value": "calculator"})
        assert captured.get("args") == {"action": "open", "value": "calculator"}

    def test_open_url_remap_to_computer_control(self, monkeypatch):
        """_dispatch_tool should remap open_url to computer_control."""
        from controller import agent_controller as ac

        captured = {}

        def mock_computer_control(args):
            captured["args"] = args
            return '{"status": "ok"}'

        monkeypatch.setattr(ac, "computer_control", mock_computer_control)

        result = ac._dispatch_tool("open_url", {"value": "https://youtube.com"})
        assert captured.get("args") == {"action": "open", "value": "https://youtube.com"}

    def test_close_app_remap_to_computer_control(self, monkeypatch):
        """_dispatch_tool should remap close_app to computer_control close_app action."""
        from controller import agent_controller as ac

        captured = {}

        def mock_computer_control(args):
            captured["args"] = args
            return '{"status": "ok", "terminated": 1}'

        monkeypatch.setattr(ac, "computer_control", mock_computer_control)

        result = ac._dispatch_tool("close_app", {"value": "chrome"})
        assert captured.get("args") == {"action": "close_app", "value": "chrome"}, (
            f"Expected action=close_app, value=chrome; got {captured.get('args')}"
        )

    def test_kill_app_remap_to_computer_control(self, monkeypatch):
        """_dispatch_tool should remap kill_app to computer_control close_app action."""
        from controller import agent_controller as ac

        captured = {}

        def mock_computer_control(args):
            captured["args"] = args
            return '{"status": "ok", "terminated": 1}'

        monkeypatch.setattr(ac, "computer_control", mock_computer_control)

        result = ac._dispatch_tool("kill_app", {"value": "notepad"})
        assert captured.get("args") == {"action": "close_app", "value": "notepad"}


class TestHermesBridgePersona:
    """Tests for Hermes bridge persona content."""

    def test_hermes_bridge_has_app_launch_guidance(self):
        """Hermes bridge ephemeral prompt should include app launch guidance."""
        from hermes_bridge import _FAIRY_EPHEMERAL_SYSTEM_PROMPT

        # Check for key guidance in the prompt
        assert "open calculator" in _FAIRY_EPHEMERAL_SYSTEM_PROMPT.lower() or \
               "terminal tool" in _FAIRY_EPHEMERAL_SYSTEM_PROMPT.lower(), (
            "Hermes should have guidance on how to launch apps"
        )

        # Check for verification guidance
        assert "verify" in _FAIRY_EPHEMERAL_SYSTEM_PROMPT.lower() or \
               "ALWAYS" in _FAIRY_EPHEMERAL_SYSTEM_PROMPT, (
            "Hermes should be told to verify actions"
        )

    def test_hermes_bridge_lists_known_apps(self):
        """Hermes bridge ephemeral prompt should list known Windows apps."""
        from hermes_bridge import _FAIRY_EPHEMERAL_SYSTEM_PROMPT

        # Check that some known apps are mentioned
        known_apps = ["calculator", "notepad", "steam", "discord", "vscode"]
        has_apps = any(app in _FAIRY_EPHEMERAL_SYSTEM_PROMPT.lower() for app in known_apps)
        assert has_apps, (
            "Hermes should be told about known Windows apps to help with app launch requests"
        )


class TestAppActionExtraction:
    """Tests for _extract_app_action_target()."""

    def test_open_calculator(self):
        """Extract 'calculator' from 'open calculator'."""
        from controller import agent_controller as ac
        result = ac._extract_app_action_target("open calculator")
        assert result == ("open", "calculator")

    def test_start_steam_app(self):
        """Extract 'steam' from 'start steam app'."""
        from controller import agent_controller as ac
        result = ac._extract_app_action_target("start steam app")
        assert result == ("start", "steam")

    def test_launch_discord(self):
        """Extract 'discord' from 'launch discord'."""
        from controller import agent_controller as ac
        result = ac._extract_app_action_target("launch discord")
        assert result == ("launch", "discord")

    def test_close_chrome(self):
        """Extract 'chrome' from 'close chrome'."""
        from controller import agent_controller as ac
        result = ac._extract_app_action_target("close chrome")
        assert result == ("close", "chrome")

    def test_kill_notepad_please(self):
        """Extract 'notepad' from 'kill notepad please'."""
        from controller import agent_controller as ac
        result = ac._extract_app_action_target("kill notepad please")
        assert result == ("kill", "notepad")

    def test_no_action_verb_returns_none(self):
        """Return None for non-action text."""
        from controller import agent_controller as ac
        result = ac._extract_app_action_target("what time is it")
        assert result is None


class TestAppActionFallback:
    """Tests for _try_app_action_fallback()."""

    def test_app_action_fallback_launch(self, monkeypatch):
        """Verify _try_app_action_fallback calls computer_control and verifies result."""
        from controller import agent_controller as ac

        def mock_computer_control(args):
            return '{"status": "ok", "opened": "calculator", "type": "app"}'

        monkeypatch.setattr(ac, "computer_control", mock_computer_control)
        # Mock the verify to avoid actually checking processes
        monkeypatch.setattr(ac, "_verify_app_launch",
                           lambda target, verb, timeout_seconds=3.0:
                           (True, f"{target} opened."))

        result = ac._try_app_action_fallback("open calculator")
        assert result is not None
        assert "calculator" in result.lower()

    def test_app_action_fallback_close_chrome(self, monkeypatch):
        """Verify close action routes to close_app."""
        # Patch where _try_app_action_fallback imports it (inside the function)
        import skills.computer_control as cc
        captured = {}

        def mock_computer_control(args):
            captured["args"] = args
            return '{"status": "ok", "terminated": 1}'

        monkeypatch.setattr(cc, "computer_control", mock_computer_control)
        from controller import agent_controller as ac
        monkeypatch.setattr(ac, "_verify_app_launch",
                           lambda target, verb, timeout_seconds=3.0:
                           (True, f"{target} closed."))

        result = ac._try_app_action_fallback("close chrome")
        assert captured.get("args", {}).get("action") == "close_app"
        assert captured.get("args", {}).get("value") == "chrome"

    def test_app_action_fallback_handles_failure(self, monkeypatch):
        """Verify failed app launch is reported as failure."""
        from controller import agent_controller as ac

        def mock_computer_control(args):
            return '{"status": "error", "message": "file not found"}'

        monkeypatch.setattr(ac, "computer_control", mock_computer_control)

        result = ac._try_app_action_fallback("open nonexistent_app_xyz")
        assert result is not None
        assert "couldn" in result.lower() or "failed" in result.lower() or "error" in result.lower()

    def test_app_action_fallback_handles_no_action(self):
        """Return None for non-action text."""
        from controller import agent_controller as ac
        result = ac._try_app_action_fallback("hello there")
        assert result is None


class TestComputerControlClose:
    """Tests for the new close_application function in computer_control."""

    def test_close_application_empty_target(self):
        """close_application should return error for empty target."""
        from skills.computer_control import close_application
        import json
        result = json.loads(close_application(""))
        assert result["status"] == "error"

    def test_close_application_with_target(self, monkeypatch):
        """close_application with a target should attempt termination."""
        from skills import computer_control as cc

        # Mock psutil to return no matching process
        class FakeProc:
            def __init__(self, name):
                self.info = {"name": name}
            def terminate(self):
                pass
            def wait(self, timeout=2):
                pass

        class FakePsutil:
            @staticmethod
            def process_iter(attrs):
                # Return empty — no processes
                return []

        monkeypatch.setattr(cc, "psutil", FakePsutil, raising=False)

        result = cc.close_application("nonexistent_app_xyz")
        import json
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed.get("terminated", 0) == 0


class TestKnownDesktopApps:
    """Tests for the _KNOWN_DESKTOP_APPS set."""

    def test_calculator_in_known_desktop_apps(self):
        """Calculator should be in _KNOWN_DESKTOP_APPS."""
        from controller import agent_controller as ac
        assert "calculator" in ac._KNOWN_DESKTOP_APPS

    def test_steam_in_known_desktop_apps(self):
        """Steam should be in _KNOWN_DESKTOP_APPS."""
        from controller import agent_controller as ac
        assert "steam" in ac._KNOWN_DESKTOP_APPS

    def test_vscode_in_known_desktop_apps(self):
        """VSCode should be in _KNOWN_DESKTOP_APPS."""
        from controller import agent_controller as ac
        assert "vscode" in ac._KNOWN_DESKTOP_APPS


@pytest.mark.skip(reason="removed in v2 handoff redesign — delegation pipeline replaced by direct handoff")
class TestClaudeCodeDelegation:
    """Tests for Claude Code delegation in claude_code_delegate module."""

    def test_detect_repository_task_positive(self):
        """Should detect repository tasks."""
        from controller.claude_code_delegate import detect_repository_task

        positive_cases = [
            "fix this bug in Fairy",
            "run the tests",
            "run the tests and fix the failures",
            "refactor this module",
            "add a feature to Fairy",
            "review the whole project",
            "debug this repository",
            "why is this broken",
            "fix the tests",
            "agent_controller is broken",
        ]
        for text in positive_cases:
            assert detect_repository_task(text) is True, f"Should detect: {text}"

    def test_detect_repository_task_negative(self):
        """Should NOT detect normal conversation as repository tasks."""
        from controller.claude_code_delegate import detect_repository_task

        negative_cases = [
            "hello there",
            "what time is it",
            "open calculator",
            "open youtube",
            "what is the weather today",
            "tell me a joke",
            "what is the meaning of life",
            "set a reminder for 5pm",
        ]
        for text in negative_cases:
            assert detect_repository_task(text) is False, f"Should NOT detect: {text}"

    def test_delegate_requires_permission(self):
        """delegate_to_claude_code should require permission by default."""
        from controller.claude_code_delegate import delegate_to_claude_code
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = delegate_to_claude_code(
                project_root=tmpdir,
                task="fix the bug",
                permission_granted=False,
            )
            assert result["executed"] is False
            assert result["status"] == "permission_required"
            assert "prompt" in result
            assert "yes/no" in result["prompt"].lower() or "may i" in result["prompt"].lower()

    def test_delegate_rejects_invalid_project(self):
        """delegate_to_claude_code should reject invalid project root."""
        from controller.claude_code_delegate import delegate_to_claude_code

        # Empty path
        result = delegate_to_claude_code(
            project_root="",
            task="test",
            permission_granted=True,
        )
        assert result["executed"] is False
        assert result["status"] == "invalid_project"

        # Non-existent path
        result = delegate_to_claude_code(
            project_root="C:\\nonexistent\\path\\that\\does\\not\\exist",
            task="test",
            permission_granted=True,
        )
        assert result["executed"] is False
        assert result["status"] == "invalid_project"

    def test_delegate_with_permission_no_claude(self, monkeypatch):
        """With permission but no claude binary, should report not installed."""
        from controller import claude_code_delegate as ccd
        import tempfile

        monkeypatch.setattr(ccd, "_find_claude_binary", lambda: None)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = ccd.delegate_to_claude_code(
                project_root=tmpdir,
                task="fix bug",
                permission_granted=True,
            )
            assert result["executed"] is False
            assert result["status"] == "claude_not_installed"

    def test_delegate_with_fake_claude_success(self, monkeypatch, tmp_path):
        """With a fake claude binary, should run it and report success."""
        from controller import claude_code_delegate as ccd

        # Create a fake claude script that exits 0 with output
        fake_claude = tmp_path / "claude.cmd"
        fake_claude.write_text("@echo off\necho Fixed the bug.\nexit /b 0\n")
        monkeypatch.setattr(ccd, "_find_claude_binary", lambda: str(fake_claude))

        # Use the tmp_path as project_root
        result = ccd.delegate_to_claude_code(
            project_root=str(tmp_path),
            task="fix bug",
            permission_granted=True,
        )
        assert result["executed"] is True
        assert result["status"] == "ok"
        assert "Fixed" in result.get("stdout", "") or result.get("exit_code") == 0

    def test_delegate_with_fake_claude_failure(self, monkeypatch, tmp_path):
        """With a fake claude binary that fails, should report failure."""
        from controller import claude_code_delegate as ccd

        # Create a fake claude script that exits 1
        fake_claude = tmp_path / "claude.cmd"
        fake_claude.write_text("@echo off\necho FAILED\necho error info 1>&2\nexit /b 1\n")
        monkeypatch.setattr(ccd, "_find_claude_binary", lambda: str(fake_claude))

        result = ccd.delegate_to_claude_code(
            project_root=str(tmp_path),
            task="fix bug",
            permission_granted=True,
        )
        assert result["executed"] is True
        assert result["status"] == "error"
        assert result.get("exit_code") == 1

    def test_format_report_permission_required(self):
        """format_report should return the permission prompt when not executed."""
        from controller.claude_code_delegate import format_report

        result = {
            "executed": False,
            "status": "permission_required",
            "prompt": "Master, may I have access?",
        }
        formatted = format_report(result)
        assert "Master" in formatted or "may i" in formatted.lower()

    def test_format_report_claude_not_installed(self):
        """format_report should report clearly when Claude Code isn't installed."""
        from controller.claude_code_delegate import format_report

        result = {
            "executed": False,
            "status": "claude_not_installed",
            "message": "Claude Code CLI not found.",
        }
        formatted = format_report(result)
        assert "not found" in formatted.lower() or "install" in formatted.lower()

    def test_format_report_success(self):
        """format_report should report success without fabricating output."""
        from controller.claude_code_delegate import format_report

        result = {
            "executed": True,
            "status": "ok",
            "message": "Claude Code completed in 5.0s.",
            "stdout": "Fixed the bug in module X.",
            "stderr": "",
            "exit_code": 0,
            "duration_seconds": 5.0,
        }
        formatted = format_report(result)
        assert "Fixed the bug" in formatted
        assert "5.0" in formatted or "5 " in formatted

    def test_format_report_failure_honest(self):
        """format_report should report failure honestly — not claim success."""
        from controller.claude_code_delegate import format_report

        result = {
            "executed": True,
            "status": "error",
            "message": "Claude Code failed with exit code 1",
            "stdout": "",
            "stderr": "permission denied",
            "exit_code": 1,
            "duration_seconds": 1.0,
        }
        formatted = format_report(result)
        assert "failed" in formatted.lower()
        # MUST NOT say it succeeded
        assert "fixed" not in formatted.lower() or "failed" in formatted.lower()

    def test_request_permission_includes_path(self):
        """request_permission should mention the project path."""
        from controller.claude_code_delegate import request_permission

        prompt = request_permission("E:\\fairy", "fix the bug")
        assert "E:\\fairy" in prompt
        assert "fix the bug" in prompt
        assert "May I" in prompt or "may i" in prompt.lower()
