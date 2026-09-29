#!/usr/bin/env python3
"""
Tests for the Fairyt TUI boot sequence.

Covers:
  1. Regression: no literal [dim]...[/] markup reaches stdout during boot.
     The old code wrote Rich markup strings directly to stderr with plain
     sys.stderr.write(), causing tags like "[dim]Whisper Python: …[/]"
     to appear as literal text on the terminal.
  2. BootProgress correctly accumulates entries and renders the summary.
  3. _start_whisper_server diagnostics go to boot_console (stderr), never
     to the chat console (stdout).
  4. Real-path boot (requires whisper-stt directory and whisper server):
     skips if dependencies are missing, runs if they are present.
"""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time

import pytest

# Make fairy.py importable as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fairy


# ---------------------------------------------------------------------------
# Test 1 — No literal Rich markup on stdout
#
# The old _start_whisper_server used:
#     import sys as _sys
#     _sys.stderr.write(f"[dim]Whisper Python: {python_exe}[/]\n")
#
# This wrote the literal characters "[dim]...[/]" to stderr, which showed
# up as raw markup on the terminal.  The fix routes diagnostics through
# boot_console (a Console(stderr=True) instance) which interprets the markup.
#
# We verify that stdout NEVER contains the raw "[dim]" tag string after
# importing or calling boot helpers.  stderr is allowed to contain it (it's
# the boot region); stdout is not.
# ---------------------------------------------------------------------------

class TestNoLiteralDimMarkup:
    """
    Asserts that stdout contains no literal [dim]...[/] Rich markup strings.

    The boot diagnostics are routed to boot_console (stderr).  The chat
    console (stdout) should be completely free of raw markup tokens.
    """

    def test_import_fairy_no_dim_in_stdout(self, capsys):
        """
        Simply importing fairy and calling helpers must not emit [dim] tags
        to stdout.
        """
        # The import already happened (above), but re-import is safe.
        # Force any lazy initialisation to run.
        try:
            fairy._resolve_whisper_cache("base.en")
        except Exception:
            pass  # failures are fine; we're checking stdout, not behavior

        captured = capsys.readouterr()
        assert "[dim]" not in captured.out, (
            f"stdout must not contain literal [dim] markup. Got:\n{captured.out}"
        )

    def test_boot_console_print_no_dim_on_stdout(self, capsys):
        """
        boot_console.print() writes to stderr, never stdout.
        Verify that a [dim] string passed to boot_console never leaks to stdout.
        """
        fairy.boot_console.print(fairy.Text("[dim]test line[/]", style="dim"))
        captured = capsys.readouterr()
        assert "[dim]" not in captured.out, (
            f"stdout must not contain [dim] after boot_console.print(). Got:\n{captured.out}"
        )
        # stderr is allowed to contain it
        assert "[dim]" in captured.err or captured.err == "", (
            "boot_console stderr should contain the [dim] (or be empty if no TTY)"
        )

    def test_start_whisper_server_no_dim_in_stdout(self, monkeypatch, capsys):
        """
        _start_whisper_server() must not emit literal [dim] tags to stdout,
        even when the whisper server directory is missing (an early-exit path).
        """
        # Point to a non-existent whisper directory so it exits fast
        monkeypatch.setattr(fairy, "SCRIPT_DIR", r"E:\nonexistent_fairyss")

        # Replace _is_whisper_running to always return False
        monkeypatch.setattr(fairy, "_is_whisper_running", lambda *a, **k: False)

        # Also patch _resolve_whisper_cache to avoid file-system side-effects
        monkeypatch.setattr(
            fairy, "_resolve_whisper_cache",
            lambda *a, **k: {"status": "unknown", "hf_home": "", "snapshot_dir": None, "model_size": "base.en"}
        )

        # Replace boot_console.print with a no-op to avoid Rich TTY issues in tests
        noop_prints = []
        monkeypatch.setattr(fairy.boot_console, "print", lambda *a, **k: noop_prints.append((a, k)))

        fairy._start_whisper_server()

        captured = capsys.readouterr()
        assert "[dim]" not in captured.out, (
            f"_start_whisper_server must not emit [dim] markup to stdout. Got:\n{captured.out}"
        )
        assert "[/]" not in captured.out, (
            f"_start_whisper_server must not emit [/] markup to stdout. Got:\n{captured.out}"
        )


# ---------------------------------------------------------------------------
# Test 2 — BootProgress internal state
# ---------------------------------------------------------------------------

class TestBootProgressState:
    """BootProgress must accumulate entries and track active state correctly."""

    def test_initial_state_inactive(self):
        bp = fairy.BootProgress()
        assert bp._active is True, "BootProgress starts in active state"
        assert bp._entries == {}
        assert bp._lines_rendered == 0
        assert bp._header_printed is False

    def test_finish_records_entry(self):
        bp = fairy.BootProgress()
        bp.finish("Hermes", detail="ready")
        assert bp._entries["Hermes"] == ("ok", "ready")

    def test_fail_records_entry(self):
        bp = fairy.BootProgress()
        bp.fail("Whisper STT", reason="spawn failed")
        assert bp._entries["Whisper STT"] == ("fail", "spawn failed")

    def test_skip_records_entry(self):
        bp = fairy.BootProgress()
        bp.skip("Discord", reason="--no-discord")
        assert bp._entries["Discord"] == ("skip", "--no-discord")

    def test_done_marks_inactive(self):
        bp = fairy.BootProgress()
        bp.finish("Hermes", detail="ok")
        bp.done()
        assert bp._active is False, "done() must set _active = False"

    def test_finish_after_done_is_noop(self):
        """finish() after done() must not raise or print."""
        bp = fairy.BootProgress()
        bp.finish("Hermes", detail="ok")
        bp.done()
        # Must not raise
        bp.finish("Hermes", detail="ok")
        # State should be unchanged (entry already "ok")
        assert bp._entries["Hermes"] == ("ok", "ok")

    def test_component_order_respected(self):
        """finish() should store entries in component order for predictable summary."""
        bp = fairy.BootProgress()
        bp.finish("Discord", detail="ready")
        bp.finish("Whisper STT", detail="3.1s")
        bp.finish("Hermes", detail="ready")
        ordered = bp._ordered()
        assert ordered == ["Hermes", "Whisper STT", "Discord"], (
            f"Expected Hermes/Whisper/Discord order, got {ordered}"
        )

    def test_icon_map(self):
        bp = fairy.BootProgress()
        assert bp._icon("ok") == "✓"
        assert bp._icon("fail") == "✗"
        assert bp._icon("skip") == "○"
        assert bp._icon("starting") == "…"
        assert bp._icon("unknown") == "?"

    def test_summary_contains_all_finished(self):
        fairy._boot_start_time = time.time()
        bp = fairy.BootProgress()
        bp.finish("Hermes", detail="ready")
        bp.finish("Whisper STT", detail="2.1s")
        summary = str(bp._build_summary())
        assert "Hermes" in summary
        assert "Whisper STT" in summary
        assert "FAIRY ready" in summary

    def test_boot_elapsed_is_a_small_duration_not_a_timestamp(self):
        """
        Regression: _boot_start_time used to be left at 0.0, so
        elapsed = time.time() - 0.0 printed an absolute Unix timestamp
        (~1.79 billion) instead of a real boot duration.
        Also verify that a non-zero start time produces a small elapsed.
        """
        fairy._boot_start_time = time.time() - 5.0  # simulate 5 s ago
        bp = fairy.BootProgress()
        bp.finish("Hermes", detail="ok")
        summary = str(bp._build_summary())
        # The elapsed suffix should be a small float, e.g. "5.0s" — never a
        # value over 1 000 000 (Unix timestamp range).
        import re
        m = re.search(r"\((\d+\.?\d*)s\)", summary)
        assert m is not None, f"Could not find elapsed suffix in summary: {summary!r}"
        elapsed = float(m.group(1))
        assert elapsed < 120, (
            f"elapsed={elapsed} looks like a Unix timestamp, not a boot duration. "
            f"Full summary: {summary!r}"
        )
        assert elapsed >= 4.5, (
            f"elapsed={elapsed} is suspiciously small (expected ~5s from test setup). "
            f"Full summary: {summary!r}"
        )

    def test_summary_gracefully_handles_zero_boot_start_time(self):
        """
        Defensive guard: if _boot_start_time is ever 0.0 (shouldn't happen
        after the fix), elapsed must not become an enormous number.
        """
        fairy._boot_start_time = 0.0
        bp = fairy.BootProgress()
        bp.finish("Hermes", detail="ok")
        summary = str(bp._build_summary())
        import re
        m = re.search(r"\((\d+\.?\d*)s\)", summary)
        assert m is not None
        elapsed = float(m.group(1))
        assert elapsed == 0.0, (
            f"elapsed should be 0.0 when _boot_start_time is 0.0, got {elapsed}"
        )


# ---------------------------------------------------------------------------
# Test 2b — User message renders in panel, not as bare text, during boot
# ---------------------------------------------------------------------------

class TestUserMessagePanelDuringBoot:
    """
    Regression: when the user types and submits a message during the boot
    phase, the message must appear EXACTLY ONCE, inside the standard You
    panel — never as raw unstyled text.

    The boot thread runs _run_boot() (daemon) while the main loop is at
    fairy_prompt().  We simulate this by:
      1. Setting _boot_start_time so the summary would show real elapsed time.
      2. Patching _get_input_session so fairy_prompt() returns "hello".
      3. Calling fairy.main() logic up to the point where a user message
         is processed through _process_turn.

    We capture stdout and verify the You panel appears exactly once with
    the message text inside it, and no bare-text echo of "hello".
    """

    def test_user_message_appears_in_panel_not_as_bare_text(self, monkeypatch, capsys):
        """
        A submitted user message must render in the You panel and NOT as
        raw text outside the panel.
        """
        # Simulate a 3-second-old boot so elapsed-time computation is valid.
        fairy._boot_start_time = time.time() - 3.0

        # Patch the input session so fairy_prompt() immediately returns.
        class _FakeSession:
            def prompt(self, *a, **k):
                return "hello"

        # We can't run the full main() loop (it blocks forever), so we call
        # _process_turn directly — the same function the main loop uses.
        # Set up a minimal ConversationState.
        from fairy import ConversationState, _process_turn
        state = ConversationState()

        # Build a minimal args object (only the attributes _process_turn uses).
        class _FakeArgs:
            no_tts = True   # avoid side-effects

        # Capture what _process_turn(announce_user=True) prints.
        capsys.readouterr()   # clear
        _process_turn("hello", state, _FakeArgs(), announce_user=True)
        captured = capsys.readouterr()

        # The message must appear inside the You panel (has "You" title).
        assert "You" in captured.out, (
            f"User message should appear in a panel with 'You' title. "
            f"stdout: {captured.out!r}"
        )
        assert "hello" in captured.out, (
            f"User message text 'hello' must appear in stdout. "
            f"stdout: {captured.out!r}"
        )
        # Count occurrences of "hello" — should appear exactly once (inside the
        # panel), not twice (bare text + panel).
        bare_count = captured.out.count("hello")
        assert bare_count == 1, (
            f"'hello' appeared {bare_count} times — expected exactly 1. "
            f"Any bare-text echo means it rendered outside the panel. "
            f"stdout: {captured.out!r}"
        )

    def test_process_turn_with_announce_false_does_not_print_user_text(self, monkeypatch, capsys):
        """
        Verify that _process_turn with announce_user=False does NOT print
        the user's text (the panel is the responsibility of the caller).
        This documents the contract change: the main loop now passes
        announce_user=True so the panel IS rendered.
        """
        fairy._boot_start_time = time.time()

        state = fairy.ConversationState()

        class _FakeArgs:
            no_tts = True

        capsys.readouterr()
        # Patch agent_controller.handle_request so we don't need a real LLM.
        from unittest.mock import patch
        with patch.object(fairy.agent_controller, "handle_request",
                         return_value=("mock reply", [])):
            fairy._process_turn("typed_message", state, _FakeArgs(), announce_user=False)
        captured = capsys.readouterr()

        # With announce_user=False, the user's text must NOT appear in stdout.
        # (The panel path is not taken; the caller is responsible for it.)
        assert "typed_message" not in captured.out, (
            f"With announce_user=False, the user text should not be printed. "
            f"stdout: {captured.out!r}"
        )

    def test_process_turn_with_announce_true_prints_user_text_in_panel(self, monkeypatch, capsys):
        """
        When announce_user=True, the user's text must appear in the You panel
        (not as bare text) — exactly once.
        """
        fairy._boot_start_time = time.time()

        state = fairy.ConversationState()

        class _FakeArgs:
            no_tts = True

        from unittest.mock import patch
        with patch.object(fairy.agent_controller, "handle_request",
                         return_value=("mock reply", [])):
            capsys.readouterr()
            fairy._process_turn("typed_during_boot", state, _FakeArgs(), announce_user=True)
        captured = capsys.readouterr()

        assert "You" in captured.out, "You panel title must appear"
        assert "typed_during_boot" in captured.out
        assert captured.out.count("typed_during_boot") == 1, (
            f"Expected exactly 1 occurrence of 'typed_during_boot', "
            f"got {captured.out.count('typed_during_boot')} in: {captured.out!r}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Real-path boot test (integration — skips if dependencies missing)
# ---------------------------------------------------------------------------

class TestWhisperRealBoot:
    """
    Starts the whisper server subprocess and verifies it opens port 9000.

    This is the closest we can get to a "live boot" test without actually
    starting the full TTY.  Skips if the whisper-stt directory is missing.
    """

    @pytest.mark.requires_network
    def test_whisper_server_starts_and_opens_port(self, monkeypatch):
        """
        The whisper server must open port 9000 within 120 seconds.
        On a warm boot (model already cached) this should be < 10 s.
        """
        import socket
        whisper_dir = r"E:\whisper-stt"
        if not os.path.isdir(whisper_dir):
            pytest.skip(f"whisper-stt directory not found: {whisper_dir}")

        venv_python = os.path.join(whisper_dir, ".venv", "Scripts", "python.exe")
        server_script = os.path.join(whisper_dir, "whisper_server.py")
        if not os.path.isfile(server_script):
            pytest.skip(f"whisper_server.py not found at {server_script}")

        # Kill any existing server so the test is deterministic
        try:
            with socket.create_connection(("127.0.0.1", 9000), timeout=0.5):
                pytest.skip("Whisper server already running on port 9000 — cannot test clean start")
        except OSError:
            pass  # Port is free — proceed

        log_file = os.path.join(whisper_dir, "test_whisper.log")
        try:
            open(log_file, "w").close()
        except Exception as exc:
            pytest.skip(f"Cannot create log file: {exc}")

        env = os.environ.copy()
        env["WHISPER_MODEL"] = "base.en"
        env["WHISPER_DEVICE"] = "cpu"
        env["WHISPER_COMPUTE_TYPE"] = "int8"
        env["WHISPER_ENHANCE"] = "0"

        python_exe = venv_python if os.path.isfile(venv_python) else sys.executable
        proc = subprocess.Popen(
            [python_exe, "whisper_server.py"],
            cwd=whisper_dir,
            env=env,
            stdout=open(log_file, "w", encoding="utf-8", errors="replace"),
            stderr=subprocess.STDOUT,
        )

        try:
            t0 = time.time()
            while time.time() - t0 < 120:
                if proc.poll() is not None:
                    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                        log = f.read()[-800:]
                    pytest.fail(f"whisper_server exited during startup. Log:\n{log}")
                try:
                    with socket.create_connection(("127.0.0.1", 9000), timeout=0.5):
                        elapsed = time.time() - t0
                        assert elapsed < 120, f"Whisper took {elapsed:.1f}s to start (expected < 120s)"
                        return  # Success
                except OSError:
                    time.sleep(0.5)
            pytest.fail("Whisper server did not open port 9000 within 120s")
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Test 4 — Discord bot spawn (real-path, skips if script missing)
# ---------------------------------------------------------------------------

class TestDiscordRealBoot:
    @pytest.mark.requires_network
    def test_discord_bot_starts_without_immediate_crash(self, monkeypatch):
        """
        _start_discord_bot() must not exit immediately (exit code != None
        within 0.5s) when the bot script is present.
        """
        script = os.path.join(fairy.SCRIPT_DIR, "controller", "discord_bot.py")
        if not os.path.isfile(script):
            pytest.skip(f"discord_bot.py not found: {script}")

        # Quick smoke: spawn, wait 0.5s, check it hasn't exited
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0

        proc = subprocess.Popen(
            [sys.executable, script],
            cwd=fairy.SCRIPT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=si,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        time.sleep(0.5)
        poll = proc.poll()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        assert poll is None, (
            f"Discord bot exited immediately (code {poll}). "
            "Check discord_bot.log for details."
        )
