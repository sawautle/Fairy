"""
FAIRY animated ASCII logo — minimal title sequence.

Composition: SOLID █ block FAIRY wordmark rendered via pyfiglet
(ansi_shadow font, verbatim — no scaling, no post-processing) as the
hero element, colored electric blue.

Minimal animation (~2s):
  1. RAIN STORM  ~0.8s — heavy digital rain (▚▖▗), fades out
  2. SNAP        ~0.1s — single white flash frame
  3. TYPE-ON     ~0.8s — "F·A·I·R·Y" + tagline stamp
  4. HOLD        ~0.3s — final frame

Render path is logged so Master can see which path ran.

Safety:
  - non-TTY/piped: instant static render
  - narrow terminal: compact fallback
  - fit check: 6 rows × 36 cols minimum
"""

from __future__ import annotations

import sys
import os
import time
import threading
import shutil
import random
import select as _select
import logging

# ── Logger ────────────────────────────────────────────────────────────────────
_log = logging.getLogger("ascii_logo")
_log.setLevel(logging.INFO)
if not _log.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[ascii_logo] %(levelname)s: %(message)s"))
    _log.addHandler(_handler)

try:
    from rich.live import Live as _Live
    from rich.console import Console as _Console, Group as _Group
    from rich.text import Text as _RichText
except ImportError:
    _Live = None
    _Console = None
    _Group = None
    _RichText = None

# ── Color constants ────────────────────────────────────────────────────────────
_C_DIM  = "\033[38;5;67m"
_C_MID  = "\033[38;5;39m"
_C_HOT  = "\033[38;5;51m"
_C_WHT  = "\033[1;97m"
_C_RST  = "\033[0m"
_C_CLR  = "\033[2J\033[H"

# ── Wordmark: pyfiglet ansi_shadow VERBATIM ────────────────────────────────
# ansi_shadow: solid █ block letters with 3D shadow built in.
# Rendered EXACTLY as pyfiglet returns it — no scaling, no doubling.
# Native: 6 rows × 36 cols.

_WORDMARK_RAW: list[str] = []
_FIGLET_OK: bool = False

try:
    import pyfiglet as _pyfiglet
    _fig = _pyfiglet.Figlet(font="ansi_shadow")
    _raw = _fig.renderText("FAIRY")
    _lines = [_ for _ in _raw.split("\n")]   # all lines
    # Strip trailing blank lines only (preserve internal gaps)
    while _lines and not _lines[-1].strip():
        _lines.pop()
    _WORDMARK_RAW = _lines
    _FIGLET_OK = True
except Exception as _e:
    _log.warning("pyfiglet unavailable (%s), falling back to static", _e)
    _FIGLET_OK = False
    _WORDMARK_RAW = [
        "███████╗ █████╗ ██╗██████╗ ██╗   ██╗",
        "██╔════╝██╔══██╗██║██╔══██╗╚██╗ ██╔╝",
        "█████╗  ███████║██║██████╔╝ ╚████╔╝ ",
        "██╔══╝  ██╔══██║██║██╔══██╗  ╚██╔╝  ",
        "██║     ██║  ██║██║██║  ██║   ██║   ",
        "╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝   ╚═╝   ",
    ]

_WORD_ROWS = len(_WORDMARK_RAW)
_WORD_COLS = max((len(l) for l in _WORDMARK_RAW), default=36)

# Apply electric blue color
_WORDMARK_COLORED = [_C_MID + ln + _C_RST for ln in _WORDMARK_RAW]

if _FIGLET_OK:
    _log.info("wordmark: pyfiglet ansi_shadow verbatim, %d rows x %d cols",
              _WORD_ROWS, _WORD_COLS)
else:
    _log.warning("wordmark: fallback static")

# ── Legacy compat shims (deprecated — kept so tests don't break) ─────────────
_PIXEL_LETTERS = [[0]*5 for _ in range(7)]
_PIXEL_NAMES   = ["F", "A", "I", "R", "Y"]
_PIXEL_W = 2
_LETTER_COLS = 10
_KERN_COLS = 2
# ── End legacy ────────────────────────────────────────────────────────────────


# ── Rendering helpers ────────────────────────────────────────────────────────
def _render_wordmark(letter_bitmaps=None) -> list[str]:
    """Return the wordmark lines verbatim."""
    return list(_WORDMARK_RAW)


def _apply_color(lines: list[str], color: str) -> list[str]:
    return [color + ln + _C_RST for ln in lines]


# ── Digital rain ──────────────────────────────────────────────────────────────
_RAIN_CHARS = ["▚", "▖", "▗"]
_RAIN_BRIGHT = [_C_HOT, _C_MID, _C_DIM]


def _make_rain_frame(tick: int, width: int, height: int, density: float) -> list[str]:
    """Generate one rain frame."""
    rng = random.Random(tick * 2654435761 & 0xFFFFFFFF)
    lines_out = []
    for y in range(height):
        line = ""
        for x in range(0, width, 2):
            if rng.random() < density:
                bi = (x * 7 + tick * 13 + y) % 3
                ci = (x * 3 + tick * 5 + y) % 3
                line += _RAIN_BRIGHT[bi] + _RAIN_CHARS[ci] + _C_RST
            else:
                line += "  "
        lines_out.append(line)
    return lines_out


# ── Helpers ──────────────────────────────────────────────────────────────────
def _enable_ansi() -> bool:
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False


def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _get_clock() -> str:
    return time.strftime("%H:%M:%S")


def _fits_terminal() -> bool:
    try:
        term_cols, term_lines = shutil.get_terminal_size()
        # wordmark needs 6 rows × 36 cols; total banner ~12 rows × 36 cols
        return term_lines >= 12 and term_cols >= 36
    except Exception:
        return False


def _render_compact() -> str:
    return (
        f"{_C_MID}╭{'-' * 30}╮{_C_RST}\n"
        f"{_C_MID}│{_C_WHT}        F A I R Y        {_C_RST}{_C_MID}│{_C_RST}\n"
        f"{_C_MID}│{_C_DIM}    personal ai companion   {_C_RST}{_C_MID}│{_C_RST}\n"
        f"{_C_MID}╰{'-' * 30}╯{_C_RST}"
    )


# ── Frame builders ────────────────────────────────────────────────────────────
def _build_frame(
    frame_type: str,
    rain_tick: int = 0,
    rain_density: float = 0.7,
    terminal_width: int = 36,
    white_flash: bool = False,
    letters_shown: int = 0,
    show_tagline: bool = False,
) -> "_Group | str":
    """Build a frame. Simplified: rain / flash / type-on."""
    if _Group is None or _RichText is None:
        return _build_frame_string(
            frame_type, rain_tick, rain_density, terminal_width,
            white_flash, letters_shown, show_tagline
        )

    parts: list = []

    if frame_type == "rain":
        rain = _make_rain_frame(rain_tick, terminal_width, _WORD_ROWS, rain_density)
        for rl in rain:
            parts.append(_ansi_to_text(rl))

    elif frame_type == "flash":
        # White wordmark, 1 frame only
        for wl in _WORDMARK_COLORED:
            parts.append(_ansi_to_text(_C_WHT + wl + _C_RST))

    elif frame_type == "typeon":
        for wl in _WORDMARK_COLORED:
            parts.append(_ansi_to_text(wl))
        parts.append(_RichText(""))
        fairyx = "F·A·I·R·Y"
        revealed = fairyx[:letters_shown].center(terminal_width)
        parts.append(_ansi_to_text(_C_WHT + revealed + _C_RST))
        if show_tagline:
            parts.append(_RichText(""))
            tag = "personal ai companion".center(terminal_width)
            parts.append(_ansi_to_text(_C_DIM + tag + _C_RST))
            parts.append(_ansi_to_text(_C_MID + _get_clock().center(terminal_width) + _C_RST))

    elif frame_type == "final":
        for wl in _WORDMARK_COLORED:
            parts.append(_ansi_to_text(wl))
        parts.append(_RichText(""))
        fairyx = "F·A·I·R·Y".center(terminal_width)
        parts.append(_ansi_to_text(_C_WHT + fairyx + _C_RST))
        parts.append(_RichText(""))
        tag = "personal ai companion".center(terminal_width)
        parts.append(_ansi_to_text(_C_DIM + tag + _C_RST))
        parts.append(_ansi_to_text(_C_MID + _get_clock().center(terminal_width) + _C_RST))

    return _Group(*parts)


def _build_frame_string(
    frame_type: str,
    rain_tick: int = 0,
    rain_density: float = 0.7,
    terminal_width: int = 36,
    white_flash: bool = False,
    letters_shown: int = 0,
    show_tagline: bool = False,
) -> str:
    """Build a frame as plain string (no-Rich fallback)."""
    lines: list[str] = []

    if frame_type == "rain":
        rain = _make_rain_frame(rain_tick, terminal_width, _WORD_ROWS, rain_density)
        lines.extend(rain)

    elif frame_type == "flash":
        for wl in _WORDMARK_COLORED:
            lines.append(_C_WHT + wl + _C_RST)

    elif frame_type == "typeon":
        lines.extend(_WORDMARK_COLORED)
        lines.append("")
        fairyx = "F·A·I·R·Y"
        revealed = fairyx[:letters_shown].center(terminal_width)
        lines.append(_C_WHT + revealed + _C_RST)
        if show_tagline:
            lines.append("")
            lines.append(_C_DIM + "personal ai companion".center(terminal_width) + _C_RST)
            lines.append(_C_MID + _get_clock().center(terminal_width) + _C_RST)

    elif frame_type == "final":
        lines.extend(_WORDMARK_COLORED)
        lines.append("")
        lines.append(_C_WHT + "F·A·I·R·Y".center(terminal_width) + _C_RST)
        lines.append("")
        lines.append(_C_DIM + "personal ai companion".center(terminal_width) + _C_RST)
        lines.append(_C_MID + _get_clock().center(terminal_width) + _C_RST)

    return "\n".join(lines)


def _ansi_to_text(ansi_line: str) -> "_RichText":
    if _RichText is None:
        return ansi_line
    return _RichText.from_ansi(ansi_line)


def _centered(text: str, width: int) -> str:
    return text.center(width)


# ── Animation ─────────────────────────────────────────────────────────────────
# Phase durations (total ~2s)
_ANIM_DUR_RAIN   = 0.8   # rain storm, fades out
_ANIM_DUR_FLASH  = 0.1   # single white flash frame
_ANIM_DUR_TYPEON = 0.8   # F·A·I·R·Y + tagline stamp
_ANIM_DUR_HOLD   = 0.3   # final hold
_FPS = 30
_FRAME_TIME = 1.0 / _FPS


def _animate():
    """Run the minimal title sequence (~2s)."""
    _enable_ansi()

    try:
        term_cols = shutil.get_terminal_size().columns
    except Exception:
        term_cols = 80
    TW = min(term_cols, 120)

    done = threading.Event()

    def _key_listener():
        try:
            if os.name == "nt":
                import msvcrt
                while not done.is_set():
                    if msvcrt.kbhit():
                        msvcrt.getch()
                        done.set()
                        return
                    time.sleep(0.05)
            else:
                fd = sys.stdin.fileno()
                while not done.is_set():
                    r, _, _ = _select.select([sys.stdin], [], [], 0.05)
                    if r:
                        sys.stdin.read(1)
                        done.set()
                        return
        except Exception:
            pass

    t = threading.Thread(target=_key_listener, daemon=True)
    t.start()

    t0 = time.monotonic()
    p_rain_end  = _ANIM_DUR_RAIN
    p_flash_end  = p_rain_end + _ANIM_DUR_FLASH
    p_type_end   = p_flash_end + _ANIM_DUR_TYPEON

    rng = random.Random()

    if _Live is not None:
        with _Live(
            _build_frame("rain", rain_tick=0, rain_density=0.8, terminal_width=TW),
            console=_Console(force_terminal=True),
            refresh_per_second=_FPS,
            transient=False,
        ) as live:
            tick = 0
            while not done.is_set():
                elapsed = time.monotonic() - t0
                tick += 1

                if elapsed < p_rain_end:
                    t_phase = elapsed / _ANIM_DUR_RAIN
                    density = 0.8 * (1.0 - t_phase)   # fade out
                    frame = _build_frame(
                        "rain", rain_tick=tick, rain_density=density, terminal_width=TW
                    )

                elif elapsed < p_flash_end:
                    # Single white flash frame
                    frame = _build_frame("flash", terminal_width=TW)

                elif elapsed < p_type_end:
                    t_phase = (elapsed - p_flash_end) / _ANIM_DUR_TYPEON
                    letters = min(9, int(t_phase * 10))
                    show_tag = t_phase > 0.6
                    frame = _build_frame(
                        "typeon",
                        letters_shown=letters,
                        show_tagline=show_tag,
                        terminal_width=TW,
                    )

                else:
                    frame = _build_frame("final", terminal_width=TW)
                    if elapsed >= p_type_end + _ANIM_DUR_HOLD:
                        done.set()
                        break

                live.update(frame)
                time.sleep(_FRAME_TIME)

        done.set()
        _render_static()
        return

    # No-Rich fallback
    sys.stdout.write(_C_CLR)
    sys.stdout.flush()
    tick = 0
    while not done.is_set():
        elapsed = time.monotonic() - t0
        tick += 1
        if elapsed < p_rain_end:
            t_phase = elapsed / _ANIM_DUR_RAIN
            density = 0.8 * (1.0 - t_phase)
            frame_str = _build_frame_string("rain", rain_tick=tick, rain_density=density, terminal_width=TW)
        elif elapsed < p_flash_end:
            frame_str = _build_frame_string("flash", terminal_width=TW)
        elif elapsed < p_type_end:
            t_phase = (elapsed - p_flash_end) / _ANIM_DUR_TYPEON
            letters = min(9, int(t_phase * 10))
            show_tag = t_phase > 0.6
            frame_str = _build_frame_string("typeon", letters_shown=letters, show_tagline=show_tag, terminal_width=TW)
        else:
            frame_str = _build_frame_string("final", terminal_width=TW)
            if elapsed >= p_type_end + _ANIM_DUR_HOLD:
                done.set()
                break
        sys.stdout.write(_C_CLR + frame_str + "\n")
        sys.stdout.flush()
        time.sleep(_FRAME_TIME)

    _render_static()


# ── Static render ─────────────────────────────────────────────────────────────
def _render_static() -> list[str]:
    """Render the complete static banner."""
    _enable_ansi()
    _log.info("RENDER PATH: static")
    try:
        sys.stdout.write(_C_CLR)
        sys.stdout.flush()
    except Exception:
        pass

    try:
        term_cols = shutil.get_terminal_size().columns
    except Exception:
        term_cols = 80
    TW = min(term_cols, 120)

    lines_out: list[str] = []
    lines_out.append("")
    lines_out.extend(_WORDMARK_COLORED)
    lines_out.append("")
    lines_out.append(f"  {_C_DIM}personal ai companion{_C_RST}")
    lines_out.append(f"  {_C_MID}{_get_clock()}{_C_RST}")
    lines_out.append("")

    for ln in lines_out:
        print(ln)

    return lines_out


# ── Public API ────────────────────────────────────────────────────────────────
def show_intro(console=None, art=None, final_style: str = None, play: bool = True):
    """Show the FAIRY AI-core boot banner with minimal animation."""
    is_tty = _is_tty()
    fits   = _fits_terminal()

    if not is_tty or not play:
        _log.info("RENDER PATH: static (not TTY or play=False)")
        _render_static()
        return

    if not fits:
        _log.info("RENDER PATH: compact (terminal too small)")
        _enable_ansi()
        try:
            sys.stdout.write(_C_CLR)
            sys.stdout.flush()
        except Exception:
            pass
        print()
        print(_render_compact())
        print(f"{_C_MID}  {_get_clock()}{_C_RST}")
        print()
        return

    _log.info("RENDER PATH: full animation")
    _animate()


# ── Expose wordmark for tests / preview ───────────────────────────────────────
def get_wordmark_lines() -> list[str]:
    """Return the wordmark lines (uncolored)."""
    return _WORDMARK_RAW


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    show_intro()
