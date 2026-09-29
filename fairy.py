#!/usr/bin/env python3
"""
Fairy TUI — Rich terminal interface.
BARGE-IN: Say the wake word while Fairy is speaking to interrupt her.
"""
import os
import sys
import argparse
import inspect
import signal
import threading
import subprocess
import socket
import time
import random
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.rule import Rule
from rich.syntax import Syntax

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory, FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PTStyle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import agent_controller, tts, ascii_logo, quips, voice_listener
from controller.agent_controller import _log_chat_turn  # FIX 3: chat-path logging
from controller.quips import (
    BRAIN_CORE, BRAIN_HERMES, BRAIN_CLAUDE, BRAIN_WEB,
    BRAIN_COMPUTER, BRAIN_FILE, SUBSYSTEM_CYCLE,
    SPARKLE_SET, SPINNER_FRAMES, random_quip, next_subsystem,
)

console = Console()
# A SEPARATE console routed to stderr so boot diagnostics never interleave
# with chat output on stdout. Used by BootProgress to render an isolated
# "transient" region that can be cleared without disturbing the prompt.
boot_console = Console(stderr=True)
_running = True
_listener = None
_whisper_proc = None
_whisper_started_by_us = False
_whisper_ready = False          # True once the server is confirmed up on port 9000
_whisper_startup_error = None   # Error string if startup failed
_discord_proc = None
_discord_started_by_us = False

# Protects BootProgress._entries against concurrent access from watcher threads.
_boot_lock = threading.Lock()
_boot_start_time: float = 0.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Fairy's color palette — matches her blue/white Phaethon theme ─────
FAIRY_BLUE = "#3d7dfd"    # the deep powerline-badge blue
FAIRY_LIGHT = "#7ec8ff"   # her lighter accent blue
FAIRY_WHITE = "#f2f8ff"   # near-white highlight
LIME_ACCENT = "#a6e22e"
SOFT_PURPLE = "#b48ead"
DIM_PURPLE = "#6b5b95"
_SPARKLES = ["✦", "❄", "✧", "❆", "✶"]  # rotated for small cute touches

# ─── Personality/animation constants ──────────────────────────────────
THINKING_FACT_DELAY = 4.0  # seconds before showing a thinking fact during long turns
GREETING_DOT_CYCLE_MS = 120  # ms per frame for header dot animation
GREETING_DOT_CYCLES = 3  # full color cycles at startup
TYPEWRITER_DELAY = 0.015  # seconds per character for greeting reveal

# ─── UI display thresholds ──────────────────────────────────────────
_LARGE_INPUT_CHARS = 200   # chars; pastes above this are previewed
_LARGE_INPUT_LINES = 5     # lines; multi-line pastes above this are previewed
_LARGE_OUTPUT_CHARS = 500  # chars; replies above this are bounded
_LARGE_OUTPUT_LINES = 20   # lines; replies above this are bounded
_PREVIEW_LINES = 12        # lines shown in bounded output panel
_MAX_PANEL_WIDTH = 80      # max width for output panels


_SUPPORTS_STATUS_CB = "on_status" in inspect.signature(agent_controller.handle_request).parameters


# ──────────────────────────────────────────────────────────────────────
# Boot progress renderer — lives in its own stderr-backed region so it
# never clobbers the prompt or the chat scrollback on stdout.
#
# Usage:
#   bp = BootProgress()          # no __enter__/__exit__ needed
#   bp.start("Whisper STT", "warming up")
#   bp.finish("Whisper STT", "3.1s")
#   bp.done()                    # clears stderr region, prints summary to stdout
#
# Design notes:
#   - All boot output goes to boot_console (stderr). The chat console
#     (stdout) is NEVER touched during boot — so `console.clear()` is never
#     called on the prompt/chat scrollback. The user can start typing
#     immediately; their input is buffered by prompt_toolkit and is
#     unaffected by the boot region's clear.
#   - The "clear" is implemented as a sequence of ANSI clear-line + cursor-up
#     escape codes emitted to stderr only. The terminal applies them in-place
#     without disturbing the stdout scrollback.
# ──────────────────────────────────────────────────────────────────────


def _clear_boot_region(n_lines: int) -> None:
    """Erase the last n_lines printed to boot_console (stderr).

    Uses ANSI cursor-up + erase-line.  No-op if stderr isn't a TTY.
    """
    if n_lines <= 0:
        return
    try:
        if not boot_console.is_terminal:
            return
    except Exception:
        return
    try:
        # Move cursor up n lines, then erase each line entirely.
        seq = ""
        for _ in range(n_lines):
            seq += "\x1b[1A"   # cursor up one line
            seq += "\x1b[2K"   # erase entire line
        boot_console.file.write(seq)
        boot_console.file.flush()
    except Exception:
        pass  # Never let a cosmetic clear raise into the boot path


class BootProgress:
    """
    Accumulates boot-component status, rendering each update as a line in the
    boot region (stderr).  Call .done() at the end of the boot phase to
    clear the region and print a single summary line to the chat console.
    """

    _COMPONENT_ORDER = ["Hermes", "Whisper STT", "Discord"]

    def __init__(self):
        # name -> (status, detail_or_reason)
        self._entries: dict[str, tuple[str, str | None]] = {}
        self._lines_rendered: int = 0
        self._header_printed: bool = False
        # Set to False by done() so late-arriving watcher callbacks know not to
        # try refreshing the already-cleared boot region.
        self._active: bool = True

    def _icon(self, status: str) -> str:
        return {"ok": "✓", "fail": "✗", "skip": "○", "starting": "…"}.get(status, "?")

    def _format_line(self, name: str, status: str, detail: str | None) -> Text:
        icon = self._icon(status)
        if status == "ok":
            text = Text()
            text.append("  ", style="dim")
            text.append(icon, style=f"bold {FAIRY_LIGHT}")
            text.append(f" {name}", style=f"bold {FAIRY_LIGHT}")
        elif status == "fail":
            text = Text()
            text.append("  ", style="dim")
            text.append(icon, style="bold #ff6666")
            text.append(f" {name}", style="bold #ff6666")
        elif status == "skip":
            text = Text()
            text.append("  ", style="dim")
            text.append(icon, style="dim")
            text.append(f" {name}", style="dim")
        else:  # starting
            text = Text()
            text.append("  ", style="dim")
            text.append(icon, style="dim")
            text.append(f" {name}", style="dim")
        if detail:
            text.append(f"  ({detail})", style="dim")
        return text

    def _print_header(self) -> None:
        if not self._header_printed:
            boot_console.print(Text("⚙  Booting Fairy…", style="dim"))
            self._header_printed = True

    def start(self, name: str, detail: str | None = None) -> None:
        """Record that a component has begun booting."""
        with _boot_lock:
            self._entries[name] = ("starting", detail)
        if not self._active:
            return
        self._print_header()
        self._reprint()

    def finish(self, name: str, detail: str | None = None) -> None:
        """Record a successful finish."""
        with _boot_lock:
            self._entries[name] = ("ok", detail)
        if not self._active:
            return
        self._print_header()
        self._reprint()

    def fail(self, name: str, reason: str) -> None:
        """Record a failure."""
        with _boot_lock:
            self._entries[name] = ("fail", reason)
        if not self._active:
            return
        self._print_header()
        self._reprint()

    def skip(self, name: str, reason: str) -> None:
        """Record that a component was skipped."""
        with _boot_lock:
            self._entries[name] = ("skip", reason)
        if not self._active:
            return
        self._print_header()
        self._reprint()

    def _ordered(self) -> list[str]:
        with _boot_lock:
            in_order = [n for n in self._COMPONENT_ORDER if n in self._entries]
            others = [n for n in self._entries if n not in self._COMPONENT_ORDER]
            return in_order + others

    def _reprint(self) -> None:
        """Re-render the boot region in place: clear old lines, print new."""
        ordered = self._ordered()
        _clear_boot_region(self._lines_rendered)
        for name in ordered:
            status, detail = self._entries[name]
            boot_console.print(self._format_line(name, status, detail))
        self._lines_rendered = len(ordered) + (1 if self._header_printed else 0)

    def _build_summary(self) -> Text:
        """Build the one-line summary for the chat console."""
        parts: list[Text] = []
        for name in self._COMPONENT_ORDER:
            if name not in self._entries:
                continue
            status, detail = self._entries[name]
            icon = self._icon(status)
            if status == "ok":
                seg = Text(f"{name} {icon}")
                if detail:
                    seg.append(f" ({detail})")
                parts.append(seg)
            elif status == "fail":
                seg = Text(f"{name} {icon} ({detail})", style="#ff6666")
                parts.append(seg)
            elif status == "skip":
                parts.append(Text(f"{name} {icon}", style="dim"))
        elapsed = time.time() - _boot_start_time
        if _boot_start_time <= 0.0:
            elapsed = 0.0
        result = Text("✦ FAIRY ready — ")
        result.append("  ".join(str(p) for p in parts))
        result.append(f"  ({elapsed:.1f}s)")
        return result

    def done(self) -> None:
        """Clear the stderr boot region and print one summary to stdout."""
        total_lines = self._lines_rendered
        # Mark the region inactive so late callbacks (e.g. whisper watcher)
        # don't try to redraw an already-cleared region.
        self._active = False
        # Clear the stderr region
        _clear_boot_region(total_lines)
        # Print the summary to the chat console (stdout)
        try:
            console.print(self._build_summary())
        except Exception:
            # Never let the summary fail to print
            try:
                parts = []
                for name in self._COMPONENT_ORDER:
                    if name not in self._entries:
                        continue
                    status, detail = self._entries[name]
                    icon = self._icon(status)
                    if status == "ok":
                        s = f"{name} {icon}"
                        if detail:
                            s += f" ({detail})"
                        parts.append(s)
                    elif status == "fail":
                        parts.append(f"{name} {icon} ({detail})")
                    elif status == "skip":
                        parts.append(f"{name} {icon}")
                elapsed = time.time() - _boot_start_time
                if _boot_start_time <= 0.0:
                    elapsed = 0.0
                print("✦ FAIRY ready — " + "  ".join(parts) + f"  ({elapsed:.1f}s)")
            except Exception:
                pass


def _is_whisper_running(host="127.0.0.1", port=9000, timeout=0.1):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _resolve_whisper_cache(model_size: str = "base.en") -> dict:
    """
    Resolve where the Whisper model lives on disk (or whether it must be
    downloaded). Pure path-resolution — does NOT hit the network.

    Returns a dict with:
      status: "cached" | "missing" | "unknown"
      hf_home: str   (resolved HF_HOME)
      snapshot_dir: str | None
      model_size: str
    """
    import os as _os
    hf_home = _os.environ.get("HF_HOME") or _os.path.join(
        _os.path.expanduser("~"), ".cache", "huggingface"
    )
    repo_id = f"Systran/faster-whisper-{model_size}"
    snapshots = _os.path.join(
        hf_home, "hub", f"models--{repo_id.replace('/', '--')}", "snapshots"
    )
    snap_dir = None
    if _os.path.isdir(snapshots):
        entries = [
            _os.path.join(snapshots, d)
            for d in _os.listdir(snapshots)
            if _os.path.isdir(_os.path.join(snapshots, d))
        ]
        if entries:
            snap_dir = entries[0]
    if snap_dir and _os.path.isfile(_os.path.join(snap_dir, "model.bin")):
        return {
            "status": "cached",
            "hf_home": hf_home,
            "snapshot_dir": snap_dir,
            "model_size": model_size,
        }
    return {
        "status": "missing" if _os.path.isdir(hf_home) else "unknown",
        "hf_home": hf_home,
        "snapshot_dir": snap_dir,
        "model_size": model_size,
    }


def _start_whisper_server(boot_progress: "BootProgress | None" = None):
    """
    Spawn the whisper_server.py subprocess and return IMMEDIATELY without
    waiting for the model to load. Boot must not block on STT.

    Background worker thread monitors subprocess + port. Sets
    _whisper_ready = True when the server is reachable, or
    _whisper_startup_error on failure. UI (PTT / status) reads these.

    If a BootProgress is passed, the boot region is updated in place
    (starting → ok / fail). Diagnostics (Python, working dir, cache probe)
    are written to the boot region via boot_console — NEVER to the chat
    console, so the prompt scrollback is preserved.
    """
    global _whisper_proc, _whisper_started_by_us
    global _whisper_ready, _whisper_startup_error

    # If port is already open, server is live — nothing to do
    if _is_whisper_running():
        _whisper_ready = True
        _whisper_started_by_us = False
        if boot_progress is not None:
            boot_progress.finish("Whisper STT", detail="already running")
        return True, "Whisper server already running"

    WHISPER_DIR = r"E:\whisper-stt"
    WHISPER_VENV_PYTHON = os.path.join(WHISPER_DIR, ".venv", "Scripts", "python.exe")
    LOG_FILE = r"E:\fairy\whisper_server.log"

    env = os.environ.copy()
    env["WHISPER_MODEL"] = "base.en"
    env["WHISPER_DEVICE"] = "cpu"
    env["WHISPER_COMPUTE_TYPE"] = "int8"
    env["WHISPER_ENHANCE"] = "0"

    try:
        open(LOG_FILE, "w").close()
    except Exception:
        pass

    # Use the whisper-stt venv Python directly so faster-whisper is available.
    # shutil.which("python") returns the Windows Store launcher stub
    # (C:\Users\...\python.EXE) on most Windows installs, which doesn't have
    # faster-whisper and causes the server to crash silently during startup.
    if os.path.isfile(WHISPER_VENV_PYTHON):
        python_exe = WHISPER_VENV_PYTHON
    else:
        import shutil
        python_exe = shutil.which("python") or "python"

    # Diagnostic lines — tell the user where the model is coming from
    # (cached vs. would need to download). All written to the boot
    # console (stderr) so the chat console is never polluted.
    if boot_progress is not None:
        boot_progress.start("Whisper STT", detail=f"warming up (cwd {os.path.basename(WHISPER_DIR)})")
    else:
        boot_console.print(Text(f"  Whisper Python: {python_exe}", style="dim"))
        boot_console.print(Text(f"  Working dir: {WHISPER_DIR}", style="dim"))

    try:
        cache = _resolve_whisper_cache("base.en")
        if cache["status"] == "cached":
            cache_line = f"model 'base.en' cached"
        elif cache["status"] == "missing":
            cache_line = f"model 'base.en' not in cache (will download)"
        else:
            cache_line = f"model 'base.en' cache status: {cache['status']}"
        boot_console.print(Text(f"    {cache_line}", style="dim"))
    except Exception as exc:
        boot_console.print(Text(f"    (cache probe failed: {exc})", style="dim"))

    _whisper_ready = False
    _whisper_startup_error = None

    try:
        # Use a real file for stdout/stderr — pipe-based capture blocks
        # the parent on Windows when the child writes a lot of text.
        # The watcher tails the log if startup fails.
        _log_fp = open(LOG_FILE, "w", encoding="utf-8", errors="replace")
        _whisper_proc = subprocess.Popen(
            [python_exe, "whisper_server.py"],
            cwd=WHISPER_DIR,
            env=env,
            stdout=_log_fp,
            stderr=subprocess.STDOUT,
        )
        _whisper_started_by_us = True
    except Exception as exc:
        _whisper_startup_error = str(exc)
        if boot_progress is not None:
            boot_progress.fail("Whisper STT", reason=f"spawn failed: {exc}")
        else:
            boot_console.print(Text(f"  ✗ Failed to spawn whisper server: {exc}", style="bold #ff6666"))
        return False, str(exc)

    # Background watcher — the ONLY thing that knows whether the server
    # ever came up. UI polls _whisper_ready / _whisper_startup_error.
    def _whisper_watcher():
        global _whisper_ready, _whisper_startup_error
        proc = _whisper_proc
        if proc is None:
            return
        t0 = time.time()
        while True:
            # Process died before opening the port
            if proc.poll() is not None:
                # Read whatever the server wrote to the log
                try:
                    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                        out = f.read()
                except Exception:
                    out = ""
                _whisper_startup_error = (
                    f"Server exited with code {proc.returncode} after "
                    f"{time.time() - t0:.1f}s"
                )
                # Update the boot region.  If boot already completed and
                # this is a later failure, also surface a brief error on
                # the chat console.
                if boot_progress is not None and boot_progress._active:
                    boot_progress.fail("Whisper STT", reason=_whisper_startup_error)
                else:
                    try:
                        console.print(
                            f"[red][Fairy] ✘ Whisper startup failed "
                            f"({_whisper_startup_error})[/]"
                        )
                        if out:
                            console.print(f"[dim]{out.strip()[-600:]}[/]")
                    except Exception:
                        pass
                return

            if _is_whisper_running():
                _whisper_ready = True
                elapsed = time.time() - t0
                # Update the boot region in place. The chat console
                # NEVER sees this line — the boot region's own summary
                # will mention "Whisper ✓ (Xs)" instead.
                if boot_progress is not None and boot_progress._active:
                    boot_progress.finish("Whisper STT", detail=f"{elapsed:.1f}s")
                # No "Whisper ready" line printed on success — per the
                # user's spec, only failures should print to the chat
                # console. Background success updates the boot summary
                # silently.
                return

            # Hard cap at 120s to avoid infinite spin
            if time.time() - t0 > 120:
                try:
                    proc.kill()
                except Exception:
                    pass
                _whisper_startup_error = "Whisper server didn't open port 9000 within 120s"
                if boot_progress is not None and boot_progress._active:
                    boot_progress.fail("Whisper STT", reason=_whisper_startup_error)
                else:
                    try:
                        console.print(f"[red][Fairy] {_whisper_startup_error}[/]")
                    except Exception:
                        pass
                return
            time.sleep(0.5)

    threading.Thread(target=_whisper_watcher, daemon=True).start()
    return True, "Whisper server starting in background"


def _stop_whisper_server():
    global _whisper_proc, _whisper_started_by_us
    if _whisper_started_by_us and _whisper_proc is not None:
        try:
            _whisper_proc.terminate()
            _whisper_proc.wait(timeout=5)
        except Exception:
            try:
                _whisper_proc.kill()
            except Exception:
                pass
        _whisper_proc = None
        _whisper_started_by_us = False


def _is_discord_bot_running():
    return _discord_proc is not None and _discord_proc.poll() is None


def _start_discord_bot(boot_progress: "BootProgress | None" = None):
    """
    Spawn the Discord bot subprocess. Returns immediately without blocking.

    If a BootProgress is passed, updates it in place (starting → ok / fail).
    Diagnostics go to the boot console (stderr) so the chat console is clean.
    """
    global _discord_proc, _discord_started_by_us

    if _is_discord_bot_running():
        if boot_progress is not None:
            boot_progress.finish("Discord", detail="already running")
        return True, "Discord bot already running"

    DISCORD_BOT_PATH = os.path.join(SCRIPT_DIR, "controller", "discord_bot.py")
    LOG_FILE = os.path.join(SCRIPT_DIR, "discord_bot.log")

    if not os.path.isfile(DISCORD_BOT_PATH):
        if boot_progress is not None:
            boot_progress.fail("Discord", reason=f"script not found")
        return False, f"Not found: {DISCORD_BOT_PATH}"

    try:
        open(LOG_FILE, "w").close()
    except Exception:
        pass

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0

    if boot_progress is not None:
        boot_progress.start("Discord", detail="connecting…")
    else:
        boot_console.print(
            Text(f"  {random.choice(_SPARKLES)} Summoning the Discord bot…", style=f"dim")
        )

    try:
        _discord_proc = subprocess.Popen(
            [sys.executable, DISCORD_BOT_PATH],
            cwd=SCRIPT_DIR,
            stdout=open(LOG_FILE, "a"),
            stderr=subprocess.STDOUT,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        # No blocking sleep here — Discord takes a few seconds to connect.
        # Poll once after a brief moment to catch immediate crashes.
        time.sleep(0.3)
        if _discord_proc.poll() is not None:
            try:
                with open(LOG_FILE, "r") as f:
                    log = f.read()[-1200:]
            except Exception:
                log = "(could not read log)"
            if boot_progress is not None:
                boot_progress.fail("Discord", reason="exited immediately")
            return False, f"Discord bot exited immediately. Log:\n{log}"

        _discord_started_by_us = True
        if boot_progress is not None:
            boot_progress.finish("Discord", detail="ready")
        return True, f"Discord bot started (log: {LOG_FILE})"
    except Exception as exc:
        if boot_progress is not None:
            boot_progress.fail("Discord", reason=str(exc))
        return False, str(exc)


def _stop_discord_bot():
    global _discord_proc, _discord_started_by_us
    if _discord_started_by_us and _discord_proc is not None:
        try:
            _discord_proc.terminate()
            _discord_proc.wait(timeout=5)
        except Exception:
            try:
                _discord_proc.kill()
            except Exception:
                pass
        _discord_proc = None
        _discord_started_by_us = False


def _stop_tts():
    try:
        tts.stop()
    except AttributeError:
        try:
            tts.stop_server()
        except Exception:
            pass


def _sigint_handler(signum, frame):
    global _running
    console.print("\n[dim][Fairy] Going dark...[/]")
    _running = False
    if _listener is not None:
        _listener.stop()
    _stop_whisper_server()
    _stop_discord_bot()
    _stop_tts()
    sys.exit(0)


signal.signal(signal.SIGINT, _sigint_handler)


def get_greeting():
    hour = datetime.now().hour
    if 5 <= hour < 12:
        greetings = [
            "Good morning, Master. The coffee's hot and I'm ready.",
            "Rise and shine, Master! Another day of brilliance awaits.",
            "Morning, Master! Hope you slept well.",
            "Good morning! Let's make today legendary.",
        ]
    elif 12 <= hour < 17:
        greetings = [
            "Good afternoon, Master. What genius idea shall we tackle next?",
            "Afternoon, Master! Ready to conquer the world?",
            "Good afternoon! I hope your day is going splendidly.",
            "Hey Master, afternoon vibes! What's the plan?",
        ]
    elif 17 <= hour < 22:
        greetings = [
            "Good evening, Master. The night is young and full of possibilities.",
            "Evening, Master! Time to unwind or get busy?",
            "Good evening! The stars are out, and so is your brilliance.",
            "Hey Master, evening approaches. Let's make it count.",
        ]
    else:
        greetings = [
            "So late night shenanigans, Master? Shouldn't you be sleeping? Or is this when the best ideas come?",
            "Late night, Master? You're a night owl, I see.",
            "Burning the midnight oil, Master? I admire your dedication.",
            "It's late, Master! But if you're up, I'm up.",
        ]
    return random.choice(greetings)


def show_header():
    console.clear()
    # The legacy "● ● ●" greeting-dots banner (animate_greeting_dots) was
    # retired in favor of the new ascii_logo.show_intro boot animation.
    # The single source of truth for the startup banner is ascii_logo.
    ascii_logo.show_intro(console)

    # ── Keyboard shortcut hints ─────────────────────────────────────────
    hint = Text.assemble(
        ("Ctrl+C", "bright_yellow"),
        ("  Cancel  ", "dim"),
        ("· ", "dim"),
        ("Ctrl+L", "bright_yellow"),
        ("  Clear  ", "dim"),
        ("· ", "dim"),
        ("↑↓", "bright_yellow"),
        ("  History (saved)  ", "dim"),
        ("· ", "dim"),
        ("Alt+Enter", "bright_yellow"),
        ("  Newline  ", "dim"),
        ("· ", "dim"),
        ("Paste", "bright_yellow"),
        ("  freely, then Enter to send", "dim"),
        ("· ", "dim"),
        ("Hold Right Ctrl", "bright_yellow"),
        ("  Talk (push-to-talk)", "dim"),
    )
    console.print(Align.center(hint))
    console.print()


def _detect_language(text: str) -> str | None:
    """Heuristically detect code language from text content."""
    stripped = text.strip()
    if stripped.startswith("```"):
        for lang in (
            "python", "javascript", "typescript", "json", "yaml",
            "bash", "sh", "shell", "html", "css", "sql",
            "rust", "go", "java", "c", "cpp", "xml",
        ):
            if stripped.lower().startswith(f"```{lang}"):
                return lang
        return "text"  # fenced block, no language specified
    if stripped.startswith("{"):
        try:
            import json
            json.loads(stripped)
            return "json"
        except Exception:
            pass
    return None


def _typewriter(text: str, style: str = "bright_white", delay: float = 0.010) -> None:
    """Cute little character-by-character reveal for short lines (greetings, quips)."""
    try:
        for ch in text:
            console.print(ch, style=style, end="")
            time.sleep(delay)
        console.print()
    except Exception:
        # Never let a cosmetic animation break the actual output
        console.print(Text(text, style=style))


def _format_output(text: str) -> None:
    """Print large output with bounded display: first N lines + count."""
    if not text:
        return

    all_lines = text.splitlines()
    total_lines = len(all_lines)
    total_chars = len(text)
    is_large = total_chars > _LARGE_OUTPUT_CHARS or total_lines > _LARGE_OUTPUT_LINES

    if not is_large:
        # Short one-liners (greetings, quips, farewells) get a cute
        # typewriter reveal; anything longer just prints instantly so it
        # never feels like it's dragging on real answers.
        if total_lines == 1 and total_chars <= 140:
            _typewriter(text, style="bright_white")
        else:
            console.print(Text(text, style="bright_white"))
        return

    # Show first _PREVIEW_LINES with an overflow indicator
    preview_lines = all_lines[:_PREVIEW_LINES]
    preview = "\n".join(preview_lines)
    overflow_lines = total_lines - _PREVIEW_LINES
    overflow_chars = total_chars - len(preview)
    preview += f"\n... ({overflow_lines} more lines, {overflow_chars} chars hidden — type !! to show full)"

    lang = _detect_language(text)
    width = min(console.width - 4, _MAX_PANEL_WIDTH)
    title = f"[bold {FAIRY_LIGHT}]Output ({total_lines} lines, {total_chars} chars)[/]"

    if lang in ("python", "javascript", "typescript", "json", "yaml",
                "bash", "sh", "html", "css", "sql", "rust", "go",
                "java", "c", "cpp", "xml"):
        syntax = Syntax(preview, lexer=lang, theme="monokai", word_wrap=True)
        panel = Panel(
            syntax,
            title=title,
            border_style=FAIRY_BLUE,
            width=width,
            height=min(total_lines + 4, console.height - 8),
        )
    else:
        panel = Panel(
            Text(preview, style="bright_white"),
            title=title,
            border_style=FAIRY_BLUE,
            width=width,
            height=min(total_lines + 4, console.height - 8),
        )

    console.print(panel)


# ──────────────────────────────────────────────────────────────────────
# Personality / animation system
#   - FairySpinner:   Rich renderable used as the spinner content
#   - SpinnerState:   thread-safe mutable state (brain, frame, fact)
#   - show_turn_flourish: small status flash at turn end
#   - typewriter_greeting: sparkle-suffixed greeting reveal
#   - animate_greeting_dots: cycling header dots at startup
#   - PromptBadge:   cycle subsystem sparkle for the input prompt
# ──────────────────────────────────────────────────────────────────────


def _detect_brain_from_input(text: str) -> str | None:
    """Heuristic: which brain mode is most likely in use, based on input."""
    if not text:
        return None
    lower = text.lower()
    if any(k in lower for k in ("hermes", "delegate", "fallback")):
        return "hermes"
    if any(k in lower for k in ("claude", "code", "coding", "program", "function", "script")):
        return "claude_code"
    if any(k in lower for k in ("search", "web", "google", "lookup", "find online")):
        return "web_search"
    if any(k in lower for k in ("computer", "click", "type into", "open", "exec", "run program")):
        return "computer_control"
    if any(k in lower for k in ("file", "write file", "read file", "edit file", "create file", "save to")):
        return "file"
    return None


def _subsystem_for_brain(brain_key: str | None):
    """Map brain key -> KaomojiSubsystem instance (or BRAIN_CORE)."""
    return {
        "hermes": BRAIN_HERMES,
        "claude_code": BRAIN_CLAUDE,
        "web_search": BRAIN_WEB,
        "computer_control": BRAIN_COMPUTER,
        "file": BRAIN_FILE,
    }.get(brain_key, BRAIN_CORE)


class SpinnerState:
    """Thread-safe mutable state for the animated spinner."""

    def __init__(self):
        self.brain_key: str | None = None
        self.custom_line: str | None = None
        self.start_time: float = 0.0
        self.show_thinking_fact: bool = False
        self.thinking_fact: str = ""
        self._lock = threading.Lock()

    def set_brain(self, brain_key: str | None, custom_line: str | None = None) -> None:
        with self._lock:
            self.brain_key = brain_key
            self.custom_line = custom_line

    def set_custom_line(self, custom_line: str | None) -> None:
        with self._lock:
            self.custom_line = custom_line

    def reset(self) -> None:
        with self._lock:
            self.start_time = time.time()
            self.show_thinking_fact = False
            self.thinking_fact = ""
            self.custom_line = None

    def stop(self) -> None:
        # Nothing to release; placeholder for symmetry with start/reset
        return

    def current_personality(self) -> str:
        with self._lock:
            if self.custom_line:
                return self.custom_line
        # Brain-specific lines live in quips.loading_line
        return quips.loading_line(self.brain_key)

    def current_brain_info(self) -> tuple[str, str]:
        """Return (brain_name, color_hex) for the active brain."""
        with self._lock:
            key = self.brain_key
        if key and key in quips.BRAIN_LOADING_LINES:
            name, color, _ = quips.BRAIN_LOADING_LINES[key]
            return name, color
        return "BRAIN.CORE", FAIRY_BLUE

    def maybe_show_thinking_fact(self) -> None:
        """If running > THINKING_FACT_DELAY and not yet shown, pick a fact."""
        with self._lock:
            elapsed = time.time() - self.start_time
            if elapsed >= THINKING_FACT_DELAY and not self.show_thinking_fact:
                self.show_thinking_fact = True
                self.thinking_fact = random_quip()


def _render_spinner_line(state: SpinnerState) -> str:
    """Compose one frame of spinner text for use with console.status()."""
    frame = random.choice(SPINNER_FRAMES) if SPINNER_FRAMES else "✦"
    sparkle = random.choice(SPARKLE_SET) if SPARKLE_SET else "✦"
    brain_name, _ = state.current_brain_info()
    personality = state.current_personality()
    badge = f"[{brain_name}]"
    line = f"{sparkle} {frame} {personality}  {badge}"
    state.maybe_show_thinking_fact()
    if state.show_thinking_fact and state.thinking_fact:
        line += f"  💭 {state.thinking_fact}"
    return line


def show_turn_flourish(tools_dispatched: int, outcome: str, exception_msg: str | None = None) -> None:
    """Flash a tiny colored status line on turn completion."""
    if tools_dispatched <= 0 and outcome == "success":
        return
    try:
        if outcome == "success":
            msg = "✦ done, effortlessly"
            style = f"bold {LIME_ACCENT}"
        elif outcome == "warning":
            msg = "⚠ something exploded, I survived, you're welcome"
            style = "bold #f0a500"
        else:
            msg = "⚠ something exploded, I survived, you're welcome"
            if exception_msg:
                msg += f"  ({exception_msg[:60]})"
            style = "bold #ff4444"
        console.print(Text(msg, style=style), justify="right")
    except Exception:
        # Never let a flourish break the actual output
        pass


def typewriter_greeting(text: str, delay: float = TYPEWRITER_DELAY) -> None:
    """Typewriter-reveal the greeting, then append 2–3 sparkles."""
    try:
        from rich.console import Console as _C
    except Exception:
        _C = None  # type: ignore
    try:
        for ch in text:
            console.print(ch, style=FAIRY_WHITE, end="")
            time.sleep(delay)
        sparkle_count = random.randint(2, 3)
        sparkles = "".join(random.choice(SPARKLE_SET) for _ in range(sparkle_count))
        console.print(" " + sparkles, style=LIME_ACCENT)
        console.print()
    except Exception:
        # If anything goes wrong (non-TTY, etc.), fall back to plain print
        console.print(Text(text + " ✦", style=FAIRY_WHITE))


def animate_greeting_dots(cycles: int = GREETING_DOT_CYCLES) -> None:
    """Animate the header dots cycling through colors at startup."""
    if not console.is_terminal:
        return
    colors = [FAIRY_BLUE, FAIRY_LIGHT, LIME_ACCENT, SOFT_PURPLE, FAIRY_WHITE]
    dots = "● ● ●"
    try:
        for _ in range(cycles):
            for color in colors:
                console.print(Align.center(Text(dots, style=f"bold {color}")), end="\r")
                time.sleep(GREETING_DOT_CYCLE_MS / 1000)
        # Final state — settle on fairy blue
        console.print(Align.center(Text(dots, style=f"bold {FAIRY_BLUE}")))
    except Exception:
        # Animation is purely cosmetic; never crash on it
        console.print(Align.center(Text(dots, style=f"bold {FAIRY_BLUE}")))


class PromptBadge:
    """Manages the input prompt badge with cycling sparkles and subsystem kaomoji."""

    def __init__(self):
        self.turn_count = 0

    def render(self) -> str:
        """Return a small badge string for use with prompt_toolkit fragments."""
        self.turn_count += 1
        subsystem = next_subsystem()
        sparkle = SPARKLE_SET[(self.turn_count - 1) % len(SPARKLE_SET)] if SPARKLE_SET else "✦"
        kaomoji = subsystem.pick_kaomoji_only()
        # Plain-text badge; the styling is done via _prompt_fragments() segments
        return f"{sparkle} {kaomoji} fairy"


_prompt_badge = PromptBadge()


def _show_quick_reaction(state: SpinnerState) -> None:
    """For very fast turns, print a quick kaomoji reaction beneath the reply."""
    try:
        subsystem = _subsystem_for_brain(state.brain_key)
        kaomoji = subsystem.pick_kaomoji_only()
        console.print(Text(f"  {kaomoji}", style=f"dim {subsystem.color_hex}"))
    except Exception:
        pass


def _show_full_response(state: "ConversationState") -> None:
    """Re-display the complete last response, untruncated, for debugging."""
    text = state.last_full_reply
    if not text:
        console.print("[dim][Fairy] No full response stored yet.[/]")
        return

    lang = _detect_language(text)
    if lang in ("python", "javascript", "typescript", "json", "yaml",
                "bash", "sh", "html", "css", "sql", "rust", "go",
                "java", "c", "cpp", "xml"):
        body = Syntax(text, lexer=lang, theme="monokai", word_wrap=True)
    else:
        body = Text(text, style="bright_white")

    console.print(Panel(
        body,
        title=f"[bold {FAIRY_LIGHT}]Full response ({len(text)} chars, {len(text.splitlines())} lines)[/]",
        border_style=FAIRY_BLUE,
        width=min(console.width - 4, _MAX_PANEL_WIDTH),
    ))

    # Also save to a file so you can open/copy it reliably on Windows
    try:
        path = os.path.join(SCRIPT_DIR, "fairy_last_response.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        console.print(f"[dim]Saved to: {path}[/]")
    except OSError as exc:
        console.print(f"[dim](Could not save to file: {exc})[/]")


def _is_large_input(text: str) -> bool:
    """Return True if text is considered a large/paste input."""
    if len(text) > _LARGE_INPUT_CHARS:
        return True
    if text.count("\n") + 1 > _LARGE_INPUT_LINES:
        return True
    return False


def _build_input_keybindings() -> KeyBindings:
    """
    Claude-Code-style input behavior:
      - Enter submits immediately (normal text = one Enter, period).
      - A terminal paste (bracketed paste) is inserted as literal text,
        newlines and all, WITHOUT being interpreted as Enter presses —
        so pasting a multi-line block never auto-submits mid-paste and
        never needs a trailing blank line to close it.
      - Alt+Enter / Ctrl+J insert a literal newline if you want to type
        a multi-line message by hand instead of pasting one.
      - Ctrl+L clears the screen without losing what you've typed.
    """
    kb = KeyBindings()

    @kb.add("escape", "enter")  # Alt+Enter
    def _insert_newline_alt(event):
        event.current_buffer.insert_text("\n")

    @kb.add("c-j")  # Ctrl+J (works even where terminals eat Alt+Enter)
    def _insert_newline_ctrlj(event):
        event.current_buffer.insert_text("\n")

    @kb.add("c-l")  # Ctrl+L
    def _clear_screen(event):
        console.clear()
        event.app.renderer.reset()

    return kb


_HISTORY_FILE = os.path.join(SCRIPT_DIR, ".fairy_history")
try:
    # Persistent across restarts — ↑/↓ still works even after you close
    # and reopen the terminal.
    _input_history = FileHistory(_HISTORY_FILE)
except Exception:
    # Falls back gracefully if the folder isn't writable for some reason.
    _input_history = InMemoryHistory()

_input_style = PTStyle.from_dict({
    "segment.badge": f"bg:{FAIRY_BLUE} fg:#ffffff bold",
    "segment.arrow": f"fg:{FAIRY_BLUE}",
    "prompt": f"fg:{FAIRY_LIGHT} bold",
})


def _prompt_fragments():
    """A small powerline-style segmented prompt in Fairy's blue/white theme."""
    return [
        ("class:segment.badge", " ✦ FAIRY "),
        ("class:segment.arrow", "❯"),
        ("class:prompt", " you  "),
    ]


_input_session = None  # lazily initialized on first fairy_prompt() call


def _get_input_session():
    """Lazily create the PromptSession — requires a real TTY, so must not run at import time."""
    global _input_session
    if _input_session is None:
        _input_session = PromptSession(
            history=_input_history,
            key_bindings=_build_input_keybindings(),
            multiline=False,   # Enter = submit; paste still inserts newlines fine
            wrap_lines=True,
            style=_input_style,
        )
    return _input_session


def fairy_prompt() -> str:
    """
    Prompt for user input.

    - A single Enter submits normal text immediately — no double-Enter.
    - Pasted multi-line content is captured as one atomic paste and only
      sent when you actually press Enter.
    - If the submitted input is large (>200 chars or >5 lines), shows a
      bounded preview Panel before it's handed off to the conversation.
    - Returns the EXACT original text (no truncation) for processing.
    """
    console.print()
    try:
        text = _get_input_session().prompt(_prompt_fragments())
    except (EOFError, KeyboardInterrupt):
        raise

    text = text.strip("\n")
    if not text.strip():
        return ""

    if _is_large_input(text):
        # Show a bounded preview of what was pasted/typed
        preview = text[:300] + ("..." if len(text) > 300 else "")
        total_lines = text.count("\n") + 1
        panel = Panel(
            Text(preview, style="dim"),
            title=f"[{FAIRY_BLUE}]Pasted {total_lines} lines, {len(text)} chars[/]",
            border_style=FAIRY_BLUE,
            width=min(console.width - 4, _MAX_PANEL_WIDTH),
        )
        console.print(panel)
        console.print()

    return text


def chat_panel(text, is_user=False):
    """Display a chat message. Short replies use Panel; large replies use _format_output."""
    if not is_user:
        # For assistant output, delegate to _format_output for proper bounding
        _format_output(text)
        return

    # For user input, always use Panel (they come from voice or confirmed input)
    panel = Panel(
        Text(text, style="bright_white"),
        title=f"[{FAIRY_BLUE}]You[/]",
        border_style=FAIRY_BLUE,
        width=min(console.width - 4, _MAX_PANEL_WIDTH),
    )
    console.print(Align.right(panel))


class ConversationState:
    def __init__(self):
        self.history = []
        self.lock = threading.Lock()
        self.interrupt_event = threading.Event()
        self.in_turn = False
        self.pending_command = None
        self.last_full_reply = None   # ← NEW: untruncated last response for !! command
        # Spinner state lives on the conversation so the personality system
        # can hold its brain/fact state across turns
        self.spinner_state = SpinnerState()
        # Phase 1: background task spawned during this turn.
        # Consumed by _process_turn after handle_request() returns.
        # This is the ONLY way the live panel gets a task_id — the
        # registry must contain a real Popen handle; no text claim can
        # produce a panel without a process.
        self.background_task_id: str | None = None


def _handle_memory_cli(user_text: str) -> None:
    """Handle `!memory list` and `!memory forget <id|partial>` from the TUI.

    These are convenience shortcuts; the chat tool `forget_fact` and the
    in-prompt y/n flow cover the same surface for Discord and voice.
    """
    from memory.long_term_memory import get_long_term_memory
    ltm = get_long_term_memory()
    parts = user_text.strip().split(maxsplit=2)
    # parts[0] = "!memory", parts[1] = subcommand, parts[2] = arg
    sub = parts[1].lower() if len(parts) > 1 else ""
    if sub == "list":
        facts = ltm.list_facts()
        if not facts:
            console.print("[dim]No long-term facts stored.[/dim]")
            return
        console.print(f"[bold {FAIRY_BLUE}]Long-term facts[/bold {FAIRY_BLUE}]:")
        for i, f in enumerate(facts):
            cat = f.get("category", "")
            text = (f.get("text") or "").strip()
            fid = (f.get("id") or "")[:8]
            console.print(f"  {i + 1}. [{cat}] {text}  (id:{fid})")
        return
    if sub == "forget":
        if len(parts) < 3 or not parts[2].strip():
            console.print("[dim]Usage: !memory forget <id-or-partial-text>[/dim]")
            return
        needle = parts[2].strip().strip("'\"`")
        # Try exact id first, then partial text match.
        target = None
        for f in ltm.list_facts():
            if f.get("id", "").startswith(needle):
                target = f
                break
        if target is None:
            matches = ltm.find_facts_matching(needle)
            if not matches:
                console.print(f"[dim]No fact matched '{needle}'.[/dim]")
                return
            if len(matches) > 1:
                console.print(f"[dim]Multiple matches for '{needle}':[/dim]")
                for i, m in enumerate(matches):
                    console.print(f"  {i + 1}. {m.get('text')}  (id:{m.get('id', '')[:8]})")
                console.print("[dim]Be more specific.[/dim]")
                return
            target = matches[0]
        removed_text = target.get("text", "")
        ltm.delete_fact(target.get("id"))
        console.print(f"[dim]Forgotten: \"{removed_text}\"[/dim]")
        return
    console.print("[dim]Usage: !memory list | !memory forget <id-or-partial>[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Live background-task output panel
#
# Renders a Rich Live panel showing real-time stdout/stderr from a subprocess
# tracked in BackgroundTaskRegistry. Only active while the process is running.
# The panel NEVER renders without a real Popen handle — see the KeyError guard.
# ─────────────────────────────────────────────────────────────────────────────

def _show_live_task_panel(task_id: str) -> int | None:
    """
    Poll the task's output tail and render in a Rich Live panel.

    Runs until the process exits, then shows a final frozen panel with the
    exit code (green for 0, red for non-zero). Returns the exit code.

    Raises KeyError if task_id is not in the registry (no real process = no
    panel — this is the structural anti-fabrication check).
    """
    from rich.live import Live
    from core.background_tasks import _REGISTRY as _bg_reg

    handle = _bg_reg.get(task_id)  # raises KeyError if no real process

    def make_panel() -> "Panel":
        exit_code = handle.exit_code
        tail = _bg_reg.get_output_tail(task_id, n=10)
        combined = "\n".join(tail) if tail else "(waiting for output...)"
        elapsed = time.time() - handle.started_at
        body = Text(combined, style="bright_white")
        if exit_code is None:
            status_str = f"[dim]running... ({elapsed:.0f}s)[/]"
            border = FAIRY_BLUE
            title = f"[bold {FAIRY_LIGHT}]{handle.name}[/]  {status_str}"
        else:
            ok = exit_code == 0
            ec_style = LIME_ACCENT if ok else "#ff4444"
            status_str = f"[bold {ec_style}]exited (code {exit_code})[/]"
            border = LIME_ACCENT if ok else "#ff4444"
            title = f"[bold {FAIRY_LIGHT}]{handle.name}[/]  {status_str}"
        return Panel(
            body,
            title=title,
            border_style=border,
            width=min(console.width - 4, _MAX_PANEL_WIDTH),
        )

    try:
        with Live(make_panel(), console=console, refresh_per_second=4, transient=True) as live:
            while handle.exit_code is None and handle.proc.poll() is None:
                time.sleep(0.25)
                try:
                    live.update(make_panel())
                except Exception:
                    break
            # Final update with exit code shown
            try:
                live.update(make_panel())
            except Exception:
                pass
        return handle.exit_code
    except Exception as exc:
        console.print(f"[yellow][Fairy] Background task panel error: {exc}[/]")
        return handle.exit_code


def _process_turn(user_text, state, args, announce_user):
    state.in_turn = True
    state.interrupt_event.clear()

    # Reset spinner personality state for this turn; pick an initial brain
    # from the input so the badge matches the kind of work we're starting.
    state.spinner_state.reset()
    initial_brain = _detect_brain_from_input(user_text)
    state.spinner_state.set_brain(initial_brain)
    turn_start = time.time()
    turn_exception: str | None = None
    turn_outcome = "success"
    tools_dispatched = 0

    try:
        with state.lock:
            if state.interrupt_event.is_set():
                return

            if announce_user:
                console.print()
                chat_panel(user_text, is_user=True)
                if state.interrupt_event.is_set():
                    return

            ack_text = quips.ack()
            if not args.no_tts and not state.interrupt_event.is_set():
                tts.speak_and_play(ack_text, block=False)
                if state.interrupt_event.is_set():
                    _stop_tts()
                    return

            if state.interrupt_event.is_set():
                _stop_tts()
                return

            with console.status(
                _render_spinner_line(state.spinner_state),
                spinner="dots",
                spinner_style=FAIRY_LIGHT,
            ) as status:
                def on_status(msg):
                    if state.interrupt_event.is_set():
                        return
                    # If the controller sends a generic/empty status, rotate
                    # through a thinking phrase so the spinner stays alive
                    # without revealing chain-of-thought.
                    if not msg or not msg.strip():
                        msg = quips.thinking()
                    # Status callbacks can also flip the active brain (e.g.
                    # when delegating to hermes) — detect that here.
                    brain_from_status = _detect_brain_from_input(msg)
                    if brain_from_status:
                        state.spinner_state.set_brain(brain_from_status)
                    else:
                        state.spinner_state.set_custom_line(msg)
                    status.update(
                        f"[bold {FAIRY_LIGHT}]"
                        f"{random.choice(_SPARKLES)} {msg} "
                        f"[{_brain_badge(state.spinner_state)}]"
                        f"[/]"
                    )
                    if not args.no_tts and msg and msg.strip():
                        tts.speak_and_play(msg, block=False)

                kwargs = {"on_status": on_status} if _SUPPORTS_STATUS_CB else {}
                _log_chat_turn(user_text, "fairy_tui_start")
                try:
                    reply, state.history = agent_controller.handle_request(
                        user_text, state.history, **kwargs
                    )
                except Exception as exc:
                    if state.interrupt_event.is_set():
                        return
                    err_text = str(exc).lower()
                    if any(x in err_text for x in ("cuda", "out of memory", "cudamalloc")):
                        reply = "Out of VRAM! Close other GPU apps or run with --no-tts."
                    else:
                        reply = f"Something broke: {exc}"
                    turn_outcome = "exception"
                    turn_exception = str(exc)
                    _log_chat_turn(user_text, "fairy_tui_end", error=str(exc), outcome="exception")
                else:
                    count, names = agent_controller._was_tool_dispatched()
                    tools_dispatched = count
                    _log_chat_turn(user_text, "fairy_tui_end",
                                      reply_preview=(reply or "")[:300],
                                      tools_dispatched=count,
                                      tool_names=names,
                                      outcome="success")

            # ── Claude Code handoff check (Phase 2) ──────────────────────
            # handle_request() sets _pending_handoff when the user triggers
            # a handoff (explicit "use claude", or approved gate). On handoff,
            # reply is None — suspend the TUI, spawn Claude Code interactively,
            # resume when it exits. The state.history we just got is the
            # post-turn history; the handoff replay re-prints those entries
            # on resume.
            _handoff = agent_controller.get_pending_handoff()
            if _handoff is not None and _handoff.get("action") == "handoff":
                if state.interrupt_event.is_set():
                    _stop_tts()
                    return
                # TTS: do not speak a handoff transition
                _stop_tts()
                _suspend_and_handoff(
                    task=_handoff["task"],
                    project_root=_handoff["project_root"],
                    task_category=_handoff.get("task_category", "repo"),
                    history=state.history,
                )
                return

            # ── Phase 1: drain any background task spawned during this turn ──
            # The task_id comes from ConversationState.background_task_id, which
            # is set by agent_controller.handle_request() before returning.
            # This is the ONLY way the panel gets a task_id — BackgroundTaskRegistry
            # requires a real Popen handle. A fabricated "running" claim never
            # reaches this point (KeyError at _show_live_task_panel).
            _btid = state.background_task_id
            state.background_task_id = None  # consume immediately
            if _btid:
                from core.background_tasks import _REGISTRY as _bg_reg
                try:
                    exit_code = _show_live_task_panel(_btid)
                    _bg_reg.unregister(_btid)
                    if exit_code and exit_code != 0:
                        console.print(
                            Text(f"  ⚠ background task exited with code {exit_code}",
                                 style="bold #ff4444")
                        )
                except KeyError:
                    # No real process — structurally impossible to fabricate
                    console.print(
                        Text("  [Fairy] Background task claim rejected: no process handle found",
                             style="bold #ff4444")
                    )


            if state.interrupt_event.is_set():
                _stop_tts()
                return

            state.last_full_reply = reply   # ← NEW
            console.print()
            chat_panel(reply, is_user=False)
            if not args.no_tts and not state.interrupt_event.is_set():
                tts.speak_and_play(reply, block=False)

            # ── Personality flourishes: fast-turn kaomoji or end-of-turn status ──
            elapsed = time.time() - turn_start
            if tools_dispatched > 0 or turn_outcome != "success":
                show_turn_flourish(tools_dispatched, turn_outcome, turn_exception)
            elif elapsed < 1.0 and reply and reply.strip():
                _show_quick_reaction(state.spinner_state)

    finally:
        state.spinner_state.stop()
        state.in_turn = False


# ──────────────────────────────────────────────────────────────────────
# Claude Code handoff
#
# Replaces the delegation pipeline (Phase 2). Fairy acts as a gatekeeper:
# she opens Claude Code in the terminal, the user works directly with
# Claude Code's own TUI and permission prompts, and when Claude Code
# exits Fairy resumes with a one-line summary.
#
# Terminal control:
#   - Releases prompt_toolkit's input session (sets _input_session = None)
#     so ConPTY state is cleanly released before the child starts.
#   - Clears the Rich console so Fairy's output is gone from the screen.
#   - Spawns `claude <task>` with inherited stdio — Claude Code gets
#     the full terminal with its own VT support, permission prompts, etc.
#   - After exit, clears the screen again, redraws Fairy's header, and
#     replays the last N chat lines to rebuild visible scrollback.
#   - No -p, no --output-format json, no permission-bypass flags.
# ──────────────────────────────────────────────────────────────────────
def _suspend_and_handoff(
    task: str,
    project_root: str,
    task_category: str,
    history: list,
) -> float:
    """
    Hand off to Claude Code in the terminal.

    Suspends Fairy's TUI, spawns ``claude <task>`` interactively in
    ``project_root``, waits for it to exit, then restores the TUI.

    Args:
        task:         The task description (passed as Claude Code's initial prompt).
        project_root: Working directory for the spawn.
        task_category: "repo" or "user_file" (for logging).
        history:      Fairy's current chat history (used for scrollback replay).

    Returns:
        Duration in seconds.
    """
    import controller.claude_code_delegate as ccd

    # 1. Release prompt_toolkit's input session so ConPTY state is clean.
    global _input_session
    _input_session = None

    # 2. Clear the console — remove Fairy's output from the screen.
    console.clear()

    # 3. Show the handoff splash (pure stdout, no Rich state).
    from controller.splash import render_handoff_splash
    task_summary = task[:60] + ("..." if len(task) > 60 else "")
    try:
        render_handoff_splash(task_summary, project_root)
    except Exception:
        # Splash is cosmetic — best-effort only.
        pass

    # 4. Find claude binary and build the argv.
    claude_bin = ccd.find_claude_binary()
    if claude_bin is None:
        _print_fallback_splash(project_root)
        input("\n[Press Enter to return to Fairy]")
        return 0.0

    argv = ccd.build_handoff_command(task)
    safe, reason = ccd.is_handoff_safe_command(argv)
    if not safe:
        # Defensive: structurally impossible if build_handoff_command() is correct,
        # but guard anyway so a code regression can't bypass Claude Code's prompts.
        print(f"\n[ERROR] Handoff command failed safety check: {reason}", file=sys.stderr)
        input("\n[Press Enter to return to Fairy]")
        return 0.0

    # 5. Log HANDOFF_START.
    _log_handoff("HANDOFF_START", project_root, task_category, task)

    # 6. Spawn interactively — stdin/stdout/stderr inherited so Claude Code
    #    gets the full terminal with its own VT support and permission UI.
    start_time = time.time()
    try:
        exit_code = subprocess.run(
            argv,
            cwd=project_root,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        ).returncode
    except FileNotFoundError:
        print(f"\n[ERROR] Claude Code not found at: {argv[0]}", file=sys.stderr)
        exit_code = -1
    except KeyboardInterrupt:
        # User pressed Ctrl+C while Claude Code was running.
        exit_code = -2
    duration = time.time() - start_time

    # 7. Log HANDOFF_END.
    _log_handoff(
        "HANDOFF_END",
        project_root,
        task_category,
        task,
        duration=duration,
        exit_code=exit_code,
    )

    # 8. Clear screen and restore Fairy's TUI.
    console.clear()
    show_header()

    # 9. Replay the last N history lines as plain text so scrollback shows context.
    _replay_chat_history(history, n=10)

    # 10. Print the one-line resume summary.
    _print_resume_summary(duration, exit_code)
    console.print()  # blank line before prompt

    return duration


def _log_handoff(
    event: str,
    project_root: str,
    task_category: str,
    task: str,
    duration: float | None = None,
    exit_code: int | None = None,
) -> None:
    """Append a HANDOFF_START / HANDOFF_END line to fairy_debug.log."""
    try:
        import pathlib
        debug_log = pathlib.Path(SCRIPT_DIR) / "fairy_debug.log"
        timestamp = datetime.now().isoformat(timespec="seconds")
        parts = [
            event,
            f"cwd={project_root}",
            f"category={task_category}",
            f"task_preview={task[:80]!r}",
        ]
        if duration is not None:
            parts.append(f"duration={duration:.1f}s")
        if exit_code is not None:
            parts.append(f"exit_code={exit_code}")
        line = f"{timestamp} {' | '.join(parts)}\n"
        debug_log.open("a", encoding="utf-8").write(line)
    except Exception:
        # Logging is never fatal.
        pass


def _print_fallback_splash(project_root: str) -> None:
    """Minimal fallback when claude binary is not found (stdout-only, no Rich)."""
    print()
    print("┌" + "─" * 76 + "┐")
    print(f"│  Claude Code CLI not found.                                   │")
    print(f"│  Install: https://claude.com/claude-code                     │")
    print(f"│  Or: npm install -g @anthropic-ai/claude-code                │")
    print(f"│  Working dir: {project_root:<60}│")
    print("└" + "─" * 76 + "┘")


def _replay_chat_history(history: list, n: int = 10) -> None:
    """Print the last n non-system chat entries as plain text."""
    # Filter to user/assistant turns only
    turns = [e for e in history[-n * 2 :] if isinstance(e, dict) and e.get("role") in ("user", "assistant")]
    for entry in turns:
        role = entry.get("role", "?")
        content = (entry.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            # Truncate very long user inputs for scrollback readability
            display = content[:200] + ("..." if len(content) > 200 else "")
            console.print(f"[dim]▌ {display}[/]")
        else:
            # Assistant: show first line only
            first_line = content.splitlines()[0][:200]
            console.print(f"[dim]▌ Fairy: {first_line}[/]")


def _print_resume_summary(duration: float, exit_code: int | None) -> None:
    """Print the single-line Claude Code session summary after resume."""
    if exit_code == -2:
        label = "aborted (Ctrl+C)"
    elif exit_code is None:
        label = "exited (unknown code)"
    elif exit_code == 0:
        label = f"exited cleanly"
    else:
        label = f"exited (code {exit_code})"
    summary = f"[Claude Code] session ended — {duration:.0f}s, {label}"
    console.print(Text(summary, style="dim"))


def _brain_badge(spinner_state: SpinnerState) -> str:
    """Return the brain badge text (e.g. 'BRAIN.CORE') for the current state."""
    name, _ = spinner_state.current_brain_info()
    return name


# ──────────────────────────────────────────────────────────────────────
# Terminal image-path detection (Goal 3 — Discord parity in the TUI)
#
# If the user pastes a local path that ends in a known image extension
# and exists on disk, read the bytes and prepend a vision description
# so the brain sees the picture, not just the path.  This keeps the
# terminal TUI aligned with the Discord attachment behavior.
# ──────────────────────────────────────────────────────────────────────
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def _find_image_paths(text: str) -> list[str]:
    """Return any whitespace-delimited tokens in `text` that look like
    existing image files.  Pure function — no I/O beyond os.path.exists.
    """
    out: list[str] = []
    for token in text.split():
        if token.lower().endswith(_IMAGE_EXTS):
            try:
                if os.path.isfile(token):
                    out.append(token)
            except OSError:
                continue
    return out


def _expand_image_path_in_input(text: str) -> str:
    """If the input references local image files, prepend their
    vision-pipeline description so the brain sees the picture content.

    Falls through unchanged if no image paths are present, if vision
    is unavailable, or if any error occurs — this is best-effort.
    """
    paths = _find_image_paths(text)
    if not paths:
        return text

    try:
        from controller.vision import describe_image, is_vision_capable
    except ImportError:
        return text

    if not is_vision_capable():
        return (
            f"[Vision] The current brain model doesn't support image input. "
            f"Your text is still being processed.\n\n{text}"
        )

    # The user-visible question is everything that isn't a path.
    words = text.split()
    non_path_words = [w for w in words if w not in paths]
    question = " ".join(non_path_words).strip()
    if not question:
        question = (
            "Describe this image in detail and answer any question the user "
            "is asking about it."
        )

    chunks: list[str] = []
    for idx, path in enumerate(paths, start=1):
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            chunks.append(f"[Image {idx} ({path}) read failed: {exc}]")
            continue

        ext = os.path.splitext(path)[1].lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext, "image/png")

        try:
            desc = describe_image(data, mime, question)
        except Exception as exc:
            desc = f"[Image {idx} description failed: {exc}]"

        prefix = f"Image {idx} of {len(paths)}" if len(paths) > 1 else "Image"
        chunks.append(f"[{prefix}: {path}]\n{desc}")

    return "\n\n".join(chunks) + "\n\n" + text


def _handle_noteworthy(text, state, args):
    with state.lock:
        remark = agent_controller.generate_ambient_remark(text)
        if remark:
            chat_panel(remark, is_user=False)
            if not args.no_tts:
                tts.speak_and_play(remark, block=False)


def main():
    global _listener

    parser = argparse.ArgumentParser(description="Fairy AI Assistant")
    parser.add_argument("--no-tts", action="store_true", help="Disable voice output")
    parser.add_argument("--no-voice", action="store_true", help="Disable always-listening mic input")
    parser.add_argument("--no-discord", action="store_true", help="Don't auto-start the Discord bot")
    parser.add_argument("--wake-word", action="store_true", help="Enable wake-word detection")
    parser.add_argument(
        "--probe-context",
        action="store_true",
        help="Initialize Hermes, print the resolved context_length, then exit. "
             "Used to verify the config-side num_ctx fix without entering the TUI.",
    )
    parser.add_argument(
        "--probe-turn",
        metavar="MESSAGE",
        help="Initialize Hermes and send MESSAGE through run_turn, then exit. "
             "Use to verify the context-length fix end-to-end without the TUI.",
    )
    args = parser.parse_args()

    if args.probe_context or args.probe_turn:
        # Non-interactive probe: run model selector, init Hermes, dump the
        # resolved context length, then exit. No TTS, no listener, no Discord.
        from controller.model_selector import startup_model_check
        try:
            startup_model_check()
        except Exception as exc:
            print(f"[probe-context] Model check error: {exc}", file=sys.stderr)
        try:
            from hermes_bridge import _get_cached_agent  # noqa: F401
            agent = _get_cached_agent()
            if agent is None:
                print("[probe-context] Hermes agent init returned None", file=sys.stderr)
                sys.exit(2)
            # Mirror the resolved-context-length log path from hermes_bridge
            from hermes_bridge import _log_resolved_context_length
            _log_resolved_context_length(agent)
            if args.probe_turn:
                from hermes_bridge import run_turn
                print(f"\n[probe-turn] Sending: {args.probe_turn!r}")
                reply, history = run_turn(args.probe_turn, [])
                print(f"\n[probe-turn] Reply ({len(reply)} chars):")
                print("---BEGIN REPLY---")
                print(reply)
                print("---END REPLY---")
            sys.exit(0)
        except Exception as exc:
            print(f"[probe-context] Hermes init/run failed: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(2)

    if args.wake_word:
        console.print(
            "[yellow]wake-word mode not yet implemented — using push-to-talk[/]"
        )

    # Boot clock starts here.  Earlier this was left at the module-level
    # default of 0.0, so the ✦ FAIRY ready summary printed an absolute
    # Unix timestamp (e.g. 1788334095.4s) instead of a real boot duration.
    global _boot_start_time
    _boot_start_time = time.time()

    show_header()

    greeting = get_greeting()
    # Reveal the greeting with sparkles appended — feels alive on first boot
    typewriter_greeting(greeting)
    if not args.no_tts:
        tts.speak_and_play(greeting, block=False)

    console.print(Rule(style="dim"))

    # ── Model selection: auto-detect best available ────────────────────
    try:
        from controller.model_selector import startup_model_check
        startup_model_check()
    except Exception as exc:
        console.print(f"[yellow]Model check skipped: {exc}[/]")

    state = ConversationState()

    # ── Boot region: transient, stderr-only, cleared on completion ─────────
    # Hermes / Discord / Whisper each report into this region.  The chat
    # console (stdout) is never touched during boot — so the user's prompt
    # and any text they start typing are never erased by the boot clear.
    boot_progress = BootProgress()

    def _on_directed(text):
        if state.in_turn:
            state.interrupt_event.set()
            _stop_tts()
            state.pending_command = text

            def _wait_and_restart():
                while state.in_turn:
                    time.sleep(0.05)
                state.interrupt_event.clear()
                _process_turn(text, state, args, announce_user=True)

            threading.Thread(target=_wait_and_restart, daemon=True).start()
        else:
            _process_turn(text, state, args, announce_user=True)

    def _start_voice_listener():
        if args.no_voice:
            return
        try:
            global _listener
            _listener = voice_listener.VoiceListener(
                on_directed=_on_directed,
                on_noteworthy=lambda text: _handle_noteworthy(text, state, args),
            )
            voice_listener.set_stt_ready_getter(_is_whisper_running)
            _listener.start()
        except Exception as exc:
            # Route through boot_console so the chat scrollback is untouched.
            boot_console.print(
                Text(f"  Voice listening unavailable ({exc}) — typing only for now.", style="dim")
            )
            _listener = None

    def _run_boot():
        """
        Drive the boot sequence in a background thread.  All output goes to
        boot_console (stderr) and is erased at the end.  Updates the
        BootProgress entries; the watcher threads for Whisper also call
        boot_progress.finish() / .fail() when the port opens (or the
        process dies).
        """
        # Discord
        if not args.no_discord:
            _start_discord_bot(boot_progress=boot_progress)
        else:
            boot_progress.skip("Discord", reason="--no-discord")

        # Hermes (primary agent layer).  Non-blocking — Hermes also runs
        # via OpenRouter fallback at turn time.
        boot_progress.start("Hermes", detail="checking")
        try:
            from hermes_bridge import is_hermes_ready, is_hermes_initialized
            _hermes_use = os.getenv("FAIRY_USE_HERMES", "1") == "1"
            _ready = is_hermes_ready()
            _initialized = is_hermes_initialized()
            if _hermes_use and _ready:
                boot_progress.finish("Hermes", detail="ready (primary)")
            elif _initialized:
                # Show appropriate detail based on whether Hermes is disabled
                if _hermes_use:
                    boot_progress.finish("Hermes", detail="OpenRouter fallback")
                else:
                    boot_progress.finish("Hermes", detail="disabled (FAIRY_USE_HERMES=0)")
            else:
                boot_progress.finish("Hermes", detail="lazy (on first turn)")
        except Exception as exc:
            boot_progress.fail("Hermes", reason=str(exc))

        # Whisper (only if voice not disabled)
        if not args.no_voice:
            _start_whisper_server(boot_progress=boot_progress)
        else:
            boot_progress.skip("Whisper STT", reason="--no-voice")

        # Clear the stderr region and print the final summary on stdout.
        boot_progress.done()

    # Start the voice listener eagerly so the user has push-to-talk even
    # before the boot region finalizes.
    _start_voice_listener()

    # Drive the boot sequence on a background thread so the prompt can be
    # displayed immediately and the user can start typing.  When the boot
    # thread clears its stderr region, the prompt and any typed text
    # (on stdout) are unaffected.
    boot_thread = threading.Thread(target=_run_boot, daemon=True)
    boot_thread.start()

    while _running:
        try:
            user_text = fairy_prompt()
        except (EOFError, KeyboardInterrupt):
            console.print("[dim][Fairy] Interrupted. Exiting...[/]")
            break

        if not user_text.strip():
            continue
        if user_text.lower() in ("!!", "/show"):
            _show_full_response(state)
            continue
        # ── Memory CLI: !memory list / !memory forget <id|partial> ──
        if user_text.lower().startswith("!memory"):
            _handle_memory_cli(user_text)
            continue
        if user_text.lower() in ("quit", "exit", "q", "bye"):
            farewell = "Goodbye, Master. Try not to break anything while I'm gone."
            _format_output(farewell)
            if not args.no_tts:
                tts.speak_and_play(farewell, block=True)
            break

        # ── Image path detection (Goal 3 — terminal parity) ────────────────
        # If the user pastes a path to a local image, route it through the
        # vision pipeline so the brain sees the picture, not just the path.
        user_text = _expand_image_path_in_input(user_text)

        _process_turn(user_text, state, args, announce_user=True)

        # ── Show conversation history hint if there are past turns ─────────
        turn_count = len(state.history) // 2
        if turn_count > 0:
            console.print(
                Text(f"  ↑ {turn_count} turn{'s' if turn_count != 1 else ''} above  ·  scroll to review",
                     style="dim")
            )

    if _listener is not None:
        _listener.stop()

    _stop_whisper_server()
    _stop_discord_bot()
    console.print("\n[dim][Fairy] Offline.[]")


if __name__ == "__main__":
    main()