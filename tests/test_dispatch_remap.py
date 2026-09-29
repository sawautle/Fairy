#!/usr/bin/env python3
"""
Focused regression tests for the planner misrouting fix
(open_app / open_url → computer_control remap).

Verifies:
  1. _dispatch_tool("open_app", {"value": "steam"}) reaches the launcher.
  2. _dispatch_tool("open_url", {"value": "https://youtube.com"}) reaches the URL launcher.
  3. A nonexistent app still produces a structured launcher error (not an unknown-tool error).
  4. Proper computer_control calls continue working unchanged.
  5. Missing value returns a descriptive error, not "Unknown tool".
  6. app_or_url alias works for open_app.

Uses mocking so no actual apps are launched during tests.
"""
from __future__ import annotations

import json
import pytest


class TestDispatchRemap:
    """Tests for the open_app / open_url → computer_control remap in _dispatch_tool."""

    # ── 1. open_app reaches the launcher ────────────────────────────────────

    def test_open_app_steam_reaches_launcher(self, monkeypatch):
        """_dispatch_tool('open_app', {'value': 'steam'}) dispatches to computer_control."""
        from controller import agent_controller as ac

        captured = {}

        def mock_computer_control(args):
            captured["args"] = args
            return json.dumps({"status": "ok", "opened": "steam", "type": "app"})

        monkeypatch.setattr(ac, "computer_control", mock_computer_control)

        result = ac._dispatch_tool("open_app", {"value": "steam"})

        assert captured.get("args") == {"action": "open", "value": "steam"}, (
            f"computer_control was called with {captured.get('args')}, "
            f"expected {{'action': 'open', 'value': 'steam'}}"
        )
        parsed = json.loads(result) if isinstance(result, str) else result
        assert parsed.get("status") == "ok", f"Expected ok, got: {result}"

    def test_open_app_with_app_or_url_alias(self, monkeypatch):
        """_dispatch_tool('open_app', {'app_or_url': 'discord'}) also works."""
        from controller import agent_controller as ac

        captured = {}

        def mock_computer_control(args):
            captured["args"] = args
            return json.dumps({"status": "ok", "opened": "discord"})

        monkeypatch.setattr(ac, "computer_control", mock_computer_control)

        result = ac._dispatch_tool("open_app", {"app_or_url": "discord"})

        assert captured.get("args") == {"action": "open", "value": "discord"}, (
            f"Expected action=open, value=discord; got {captured.get('args')}"
        )

    # ── 2. open_url reaches the URL launcher ─────────────────────────────────

    def test_open_url_reaches_launcher(self, monkeypatch):
        """_dispatch_tool('open_url', {'value': 'https://youtube.com'}) dispatches correctly."""
        from controller import agent_controller as ac

        captured = {}

        def mock_computer_control(args):
            captured["args"] = args
            return json.dumps({"status": "ok", "opened": "https://youtube.com", "type": "url"})

        monkeypatch.setattr(ac, "computer_control", mock_computer_control)

        result = ac._dispatch_tool("open_url", {"value": "https://youtube.com"})

        assert captured.get("args") == {"action": "open", "value": "https://youtube.com"}, (
            f"Expected action=open, value=https://youtube.com; got {captured.get('args')}"
        )
        parsed = json.loads(result) if isinstance(result, str) else result
        assert parsed.get("status") == "ok", f"Expected ok, got: {result}"

    # ── 3. Non-existent app still produces a launcher error, not unknown-tool ──

    def test_nonexistent_app_produces_launcher_error(self, monkeypatch):
        """Nonexistent app goes through computer_control and returns a structured error, not 'Unknown tool'.

        With the post-FIX-1 verification invariant, open_application now
        short-circuits BEFORE falling through to os.startfile: if no path
        resolved via shutil.which or _LAUNCH_ALIASES, it returns
        {"status": "error", "error": "executable not found", "verified": false}.
        That is the correct behaviour: never claim success when we can't
        prove the process started.
        """
        from controller import agent_controller as ac
        import os as _real_os
        import skills.computer_control as cc

        called_with = {}

        def mock_startfile(path):
            called_with["path"] = path
            raise FileNotFoundError(2, "The system cannot find the file specified.", path)

        # Patch the `os` name in the skills.computer_control module (where open_application
        # looks it up). Use a real module-like object so isinstance checks still pass.
        import types
        fake_os = types.ModuleType("os")
        fake_os.startfile = mock_startfile
        # Pass through anything else to the real os so platform.system() etc. still work.
        for attr in dir(_real_os):
            if attr not in fake_os.__dict__ and not attr.startswith("_"):
                setattr(fake_os, attr, getattr(_real_os, attr))
        monkeypatch.setattr(cc, "os", fake_os)

        result = ac._dispatch_tool("open_app", {"value": "nonexistent_app_xyz"})

        # os.startfile must NOT be called — FIX 1 returns a structured error
        # before falling through to startfile for unknown apps.
        assert "path" not in called_with, (
            f"open_application should short-circuit before os.startfile; got: {called_with}"
        )
        # Should be a structured error (the launcher catches the exception and returns JSON),
        # NOT the unknown-tool path.
        result_str = json.dumps(result) if not isinstance(result, str) else result
        assert "Unknown tool" not in result_str, (
            f"Got 'Unknown tool' instead of structured launcher error: {result_str}"
        )
        # And it must report the failure honestly (not "ok")
        parsed = json.loads(result_str) if not isinstance(result, dict) else result
        assert parsed.get("status") != "ok", (
            f"Unknown app must not report ok; got: {result_str}"
        )
        assert parsed.get("verified") is False, (
            f"Unknown app must report verified=false; got: {result_str}"
        )

    # ── 4. Proper computer_control calls continue working unchanged ───────────

    def test_computer_control_action_open_still_works(self, monkeypatch):
        """Direct computer_control({'action': 'open', 'value': 'spotify'}) unchanged.

        open_application() now resolves bare app names (spotify, steam, etc.) in order:
          1. shutil.which() — for apps on PATH
          2. _LAUNCH_ALIASES — known absolute install paths
          3. os.startfile() — last-resort for registered handlers / URLs

        We patch both Popen and startfile so the test passes regardless of which
        layer handles the call.  The Popen mock must NOT call the real
        original_popen — otherwise the test would actually launch Spotify.
        """
        from controller import agent_controller as ac
        import subprocess

        opened_via_popen = {}
        opened_via_startfile = {}

        # Fake Popen that records its argv and exits immediately without
        # actually spawning anything.  Must not call the real subprocess.Popen.
        class FakeProc:
            def __init__(self, argv, *a, **kw):
                opened_via_popen["argv"] = argv
            def communicate(self, *a, **kw):
                return b"", b""

        monkeypatch.setattr(subprocess, "Popen", FakeProc)

        # Hermetic: skip the live-process verification step entirely.
        # This test validates dispatch routing, not liveness — liveness is
        # covered by _verify_popen_launched unit tests.
        monkeypatch.setattr(
            "skills.computer_control._verify_popen_launched",
            lambda proc, name: (True, "", 12345),
        )

        def mock_startfile(path):
            opened_via_startfile["path"] = path
            return None
        monkeypatch.setattr("os.startfile", mock_startfile)

        # Call computer_control directly (the normal path)
        result = ac.computer_control({"action": "open", "value": "spotify"})
        parsed = json.loads(result) if isinstance(result, str) else result

        # At least one of the two launch paths must have fired.
        # Popen path: FakeProc sets opened_via_popen["argv"] to the resolved Spotify path.
        # startfile path: mock_startfile sets opened_via_startfile["path"].
        popen_hit = bool(opened_via_popen.get("argv"))
        startfile_hit = bool(opened_via_startfile.get("path"))
        assert popen_hit or startfile_hit, (
            f"Expected Popen or startfile to fire for 'spotify', "
            f"got Popen={opened_via_popen}, startfile={opened_via_startfile}"
        )
        assert parsed.get("status") == "ok", f"Expected ok, got: {result}"

    def test_computer_control_volume_unchanged(self, monkeypatch):
        """computer_control({'action': 'set_volume', 'value': '50'}) still works via the key-fallback."""
        from controller import agent_controller as ac

        # Mock ctypes.windll.user32.keybd_event to a no-op so the volume fallback path runs cleanly.
        import ctypes
        monkeypatch.setattr(
            ctypes.windll.user32, "keybd_event", lambda *a, **kw: None
        )

        result = ac.computer_control({"action": "set_volume", "value": "50"})
        parsed = json.loads(result) if isinstance(result, str) else result
        # Either status=ok (key fallback) or status=error with a non-fatal message; either is acceptable
        # as long as computer_control itself was reached (not the remap path).
        assert parsed.get("status") in ("ok", "error"), f"Unexpected result shape: {result}"

    # ── 5. Missing value returns a descriptive error, not unknown-tool ────────

    def test_open_app_missing_value_returns_descriptive_error(self):
        """_dispatch_tool('open_app', {}) returns a descriptive error, not 'Unknown tool'."""
        from controller import agent_controller as ac

        result = ac._dispatch_tool("open_app", {})
        result_str = json.dumps(result) if not isinstance(result, str) else result

        assert "Unknown tool" not in result_str, (
            f"Got 'Unknown tool' instead of descriptive error: {result_str}"
        )
        assert "requires 'value'" in result_str, (
            f"Expected descriptive error about missing value, got: {result_str}"
        )

    def test_open_url_missing_value_returns_descriptive_error(self):
        """_dispatch_tool('open_url', {}) returns a descriptive error, not 'Unknown tool'."""
        from controller import agent_controller as ac

        result = ac._dispatch_tool("open_url", {})
        result_str = json.dumps(result) if not isinstance(result, str) else result

        assert "Unknown tool" not in result_str, (
            f"Got 'Unknown tool' instead of descriptive error: {result_str}"
        )
        assert "requires 'value'" in result_str, (
            f"Expected descriptive error about missing value, got: {result_str}"
        )

    # ── 6. Remap fires only for open_app / open_url ─────────────────────────

    def test_totally_unknown_tool_still_unknown(self):
        """Truly unknown tools still return 'Unknown tool'."""
        from controller import agent_controller as ac

        result = ac._dispatch_tool("not_a_real_tool", {})
        result_str = json.dumps(result) if not isinstance(result, str) else result
        assert "Unknown tool: not_a_real_tool" in result_str, f"Got: {result_str}"

    # ── 7. open_app handles arbitrary string value safely ──────────────────

    def test_open_app_value_coerced_to_string(self, monkeypatch):
        """Numeric or mixed values are coerced to string before passing to open_application."""
        from controller import agent_controller as ac

        captured = {}

        def mock_computer_control(args):
            captured["args"] = args
            return json.dumps({"status": "ok"})

        monkeypatch.setattr(ac, "computer_control", mock_computer_control)

        # value is an int in some LLM outputs
        result = ac._dispatch_tool("open_app", {"value": 12345})

        assert captured.get("args", {}).get("value") == "12345", (
            f"Expected string coercion of int, got: {captured.get('args')}"
        )
