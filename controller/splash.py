#!/usr/bin/env python3
"""
Blue splash screen for Claude ✕ Fairy interactive delegation.

Renders at runtime via pyfiglet (ansi_shadow font, verbatim 1x, no scaling,
no post-processing). Output written via sys.stdout.buffer.write() after enabling
Windows VT console mode — same technique as existing ascii_logo.py.

Design:
  - CLAUDE wordmark in ansi_shadow (49 cols × 7 rows)
  - ✕ divider
  - FAIRY wordmark in ansi_shadow (36 cols × 7 rows)
  - Blue box drawn with box-drawing characters
  - Task summary line
  - Enter prompt

The raw pyfiglet output is always generated at runtime — no literal art is
ever pasted into this file.
"""
from __future__ import annotations

import ctypes
import os
import sys

# ── VT-mode enable (Windows) ───────────────────────────────────────────────────

def _enable_vt() -> bool:
    """Enable Windows VT processing for box-drawing and block chars."""
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False


# ── Color constants (ANSI escape codes, raw) ────────────────────────────────────

_C_BLU = "\033[38;2;61;125;253m"   # FAIRY_BLUE #3d7dfd
_C_LBL = "\033[38;2;126;200;255m"   # FAIRY_LIGHT #7ec8ff
_C_WHT = "\033[1;97m"               # near-white
_C_DIM = "\033[38;5;244m"
_C_RST = "\033[0m"

# ── Wordmark generation (runtime pyfiglet — never hardcoded art) ───────────────

_WORDMARK_CLAUDE: list[str] = []
_WORDMARK_FAIRY: list[str] = []
_FIGLET_OK = False

try:
    import pyfiglet as _pyfiglet
    _fig = _pyfiglet.Figlet(font="ansi_shadow")
    _raw_claude = _fig.renderText("CLAUDE")
    _raw_fairy = _fig.renderText("FAIRY")
    _lines_c = [ln for ln in _raw_claude.split("\n")]
    _lines_f = [ln for ln in _raw_fairy.split("\n")]
    while _lines_c and not _lines_c[-1].strip():
        _lines_c.pop()
    while _lines_f and not _lines_f[-1].strip():
        _lines_f.pop()
    _WORDMARK_CLAUDE = _lines_c
    _WORDMARK_FAIRY = _lines_f
    _FIGLET_OK = True
except Exception:
    # Fallback: use a simple block-font approximation
    _WORDMARK_CLAUDE = [
        r"  ____ _        _   _   _ ____  _____ ",
        r" / ___| |      / \ | | | |  _ \| ____|",
        r"| |   | |     / _ \| | | | | | |  _|  ",
        r"| |___| |___ / ___ \ |_| | |_| | |___ ",
        r" \____|_____/_/   \_\___/|____/|_____|",
    ]
    _WORDMARK_FAIRY = [
        r"  _____ _    ___ ______   __",
        r" |  ___/ \  |_ _|  _ \ \ / /",
        r" | |_ / _ \  | || |_) \ V / ",
        r" |  _/ ___ \ | ||  _ < | |  ",
        r" |_|/_/   \_\___|_| \_\|_|  ",
    ]
    _FIGLET_OK = False


# ── Rendering ──────────────────────────────────────────────────────────────────

def render_splash(task_summary: str) -> None:
    """
    Render the full splash to stdout.

    Writes raw UTF-8 bytes to sys.stdout.buffer to bypass the cp1252 stream
    encoding on Windows. Works on any TTY without alternate screen setup.

    Args:
        task_summary: A short one-line description of the task to run.
    """
    _enable_vt()

    lines_out: list[bytes] = []

    # Helper to append a text line with color
    def out(text: str) -> None:
        lines_out.append(text.encode("utf-8"))

    def blank(n: int = 1) -> None:
        for _ in range(n):
            lines_out.append(b"")

    # ── Box ────────────────────────────────────────────────────────────────────
    # Top border
    out(f"{_C_BLU}\u250c{'─' * 76}\u2510{_C_RST}")

    # Gap row
    out(f"{_C_BLU}\u2502{_C_RST}{' ' * 76}{_C_BLU}\u2502{_C_RST}")

    # CLAUDE wordmark — color it blue, pad to center
    max_width = 76
    claude_width = max((len(ln) for ln in _WORDMARK_CLAUDE), default=49)
    claude_pad = (max_width - claude_width) // 2
    for ln in _WORDMARK_CLAUDE:
        out(f"{_C_BLU}\u2502{_C_RST}{' ' * claude_pad}{_C_BLU}{ln}{_C_RST}{' ' * (max_width - claude_pad - len(ln))}{_C_BLU}\u2502{_C_RST}")

    # Blank separator
    blank()

    # ✕ divider line — center it
    divider_text = f"  ✕  "
    pad_left = (max_width - len(divider_text)) // 2
    out(f"{_C_BLU}\u2502{_C_RST}{' ' * pad_left}{_C_BLU}{divider_text}{_C_RST}{' ' * (max_width - pad_left - len(divider_text))}{_C_BLU}\u2502{_C_RST}")

    # Blank separator
    blank()

    # FAIRY wordmark — color it blue
    fairy_width = max((len(ln) for ln in _WORDMARK_FAIRY), default=36)
    fairy_pad = (max_width - fairy_width) // 2
    for ln in _WORDMARK_FAIRY:
        out(f"{_C_BLU}\u2502{_C_RST}{' ' * fairy_pad}{_C_BLU}{ln}{_C_RST}{' ' * (max_width - fairy_pad - len(ln))}{_C_BLU}\u2502{_C_RST}")

    # Blank separator
    blank()

    # CLAUDE ✕ FAIRY header line
    header = "CLAUDE  ✕  FAIRY"
    pad = (max_width - len(header)) // 2
    out(f"{_C_BLU}\u2502{_C_RST}{' ' * pad}{_C_BLU}{_C_WHT}  {header}  {_C_RST}{' ' * (max_width - pad - len(header))}{_C_BLU}\u2502{_C_RST}")

    # Blank separator
    blank()

    # Task summary line (dimmed, centered)
    if task_summary:
        summary_display = f"Task: {task_summary[:60]}"
        pad = (max_width - len(summary_display)) // 2
        out(f"{_C_BLU}\u2502{_C_RST}{' ' * pad}{_C_DIM}{summary_display}{_C_RST}{' ' * (max_width - pad - len(summary_display))}{_C_BLU}\u2502{_C_RST}")

    # Bottom gap + border
    blank()
    out(f"{_C_BLU}\u2514{'─' * 76}\u2518{_C_RST}")

    # Flush all at once
    for line_bytes in lines_out:
        sys.stdout.buffer.write(line_bytes + b"\n")
    sys.stdout.buffer.flush()


def render_handoff_splash(task_summary: str, project_dir: str) -> None:
    """
    Render the pre-handoff splash to stdout.

    Displays the blue FAIRY ✕ CLAUDE banner followed by the task summary
    and working directory. Designed to be printed immediately before
    spawning Claude Code in interactive mode — no alternate screen switching,
    no Rich dependencies.

    Args:
        task_summary: Short description of the task (truncated to 60 chars).
        project_dir: Absolute path of the working directory.
    """
    _enable_vt()

    max_width = 76

    # Shorten project_dir for display
    home = os.path.expanduser("~")
    try:
        rel_dir = os.path.relpath(project_dir, home)
        if not rel_dir.startswith(".."):
            display_dir = "~\\" + rel_dir if rel_dir != "." else "~"
        else:
            display_dir = project_dir
    except Exception:
        display_dir = project_dir

    lines_out: list[bytes] = []

    def out(text: str) -> None:
        lines_out.append(text.encode("utf-8"))

    def blank(n: int = 1) -> None:
        for _ in range(n):
            lines_out.append(b"")

    # Box top
    out(f"{_C_BLU}┌{'─' * 76}┐{_C_RST}")
    blank()

    # Header
    header = "FAIRY  ✕  CLAUDE"
    pad = (max_width - len(header)) // 2
    out(f"{_C_BLU}│{_C_RST}{' ' * pad}{_C_WHT}{header}{_C_RST}{' ' * (max_width - pad - len(header))}{_C_BLU}│{_C_RST}")
    blank()

    # Separator
    sub = "Opening Claude Code for you — work directly in its terminal"
    pad = (max_width - len(sub)) // 2
    out(f"{_C_BLU}│{_C_RST}{' ' * pad}{_C_DIM}{sub}{_C_RST}{' ' * (max_width - pad - len(sub))}{_C_BLU}│{_C_RST}")
    blank()

    # Divider
    out(f"{_C_BLU}│{' ' * 76}│{_C_RST}")
    blank()

    # Task
    if task_summary:
        label = "Task:"
        task_display = task_summary[:60]
        content = f"{label} {task_display}"
        pad = (max_width - len(content)) // 2
        out(f"{_C_BLU}│{_C_RST}{' ' * pad}{_C_WHT}{content}{_C_RST}{' ' * (max_width - pad - len(content))}{_C_BLU}│{_C_RST}")
    else:
        out(f"{_C_BLU}│{_C_RST}{' ' * 76}{_C_BLU}│{_C_RST}")

    blank()

    # Working dir
    dir_label = "Dir:"
    dir_display = display_dir[:60]
    content = f"{dir_label} {dir_display}"
    pad = (max_width - len(content)) // 2
    out(f"{_C_BLU}│{_C_RST}{' ' * pad}{_C_DIM}{content}{_C_RST}{' ' * (max_width - pad - len(content))}{_C_BLU}│{_C_RST}")
    blank()

    # Divider
    out(f"{_C_BLU}│{' ' * 76}│{_C_RST}")
    blank()

    # Note: Claude Code will use its own terminal interface
    note = "Claude Code will run with its own terminal interface."
    pad = (max_width - len(note)) // 2
    out(f"{_C_BLU}│{_C_RST}{' ' * pad}{_C_DIM}{note}{_C_RST}{' ' * (max_width - pad - len(note))}{_C_BLU}│{_C_RST}")
    blank()

    # Box bottom
    out(f"{_C_BLU}└{'─' * 76}┘{_C_RST}")

    for line_bytes in lines_out:
        sys.stdout.buffer.write(line_bytes + b"\n")
    sys.stdout.buffer.flush()


def render_completion(
    task_summary: str,
    exit_code: int,
    git_status: str | None,
    git_diff: str | None,
    output_tail: list[str],
) -> None:
    """
    Render the evidence-only completion panel.

    Shows only: exit code (colored green/red) + git status + git diff stat.
    Never narrates what Claude Code "did" — only what git reports.
    """
    _enable_vt()

    def out(text: str) -> None:
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))

    def blank() -> None:
        out("")

    max_width = 76

    def framed(title: str, content_lines: list[str]) -> None:
        """Render a section with box-drawing frame."""
        content_width = max_width - 4
        out(f"{_C_BLU}\u2524 {_C_WHT}{title}{_C_RST}")
        for cline in content_lines:
            display = cline[:content_width]
            out(f"{_C_BLU}\u2502  {_C_DIM}{display}{_C_RST}")
        out(f"{_C_BLU}\u251c{'─' * (content_width + 2)}\u2524{_C_RST}")

    # ── Box ────────────────────────────────────────────────────────────────────
    out(f"{_C_BLU}\u250c{'─' * 76}\u2510{_C_RST}")
    blank()

    # Header
    header = f"Claude  ✕  Fairy: Complete"
    pad = (max_width - len(header)) // 2
    out(f"{_C_BLU}\u2502{_C_RST}{' ' * pad}{_C_WHT}{header}{_C_RST}{' ' * (max_width - pad - len(header))}{_C_BLU}\u2502{_C_RST}")
    blank()

    # Exit code
    ok = exit_code == 0
    ec_color = _C_BLU if ok else "\033[38;2;255;68;68m"
    out(f"{_C_BLU}\u2502  {_C_WHT}Exit code: {ec_color}{exit_code}{_C_RST}{' ' * (max_width - 18 - len(str(exit_code)))}{_C_BLU}\u2502{_C_RST}")
    blank()

    # Git status
    if git_status:
        framed("Git status:", git_status.splitlines()[:10])
    else:
        framed("Git status:", ["no files changed"])

    # Git diff
    if git_diff:
        framed("Git diff:", git_diff.splitlines()[:10])
    else:
        framed("Git diff:", ["no changes"])

    # Output tail (if any)
    if output_tail:
        framed("Output tail:", output_tail[:5])

    blank()
    out(f"{_C_BLU}\u2514{'─' * 76}\u2518{_C_RST}")
    sys.stdout.buffer.flush()
