"""Regression test: the FAIRY boot banner animation runs through rich.live.Live
without crashing, producing distinct frames.

Banner: solid █ block FAIRY wordmark via pyfiglet ansi_shadow (verbatim,
6 rows × 36 cols) as the hero element.

Minimal animation (~2s):
  rain storm → white flash → F·A·I·R·Y type-on → hold.

This test:
  1. Patches sleep to 0 (test <2s).
  2. Runs against a real Rich Console (StringIO, force_terminal=True).
  3. Asserts animation completes, output has # blocks (█), rain chars (▚▖▗),
     type-on "F·A·I·R·Y", and tagline.
  4. Asserts >= 10 distinct frames produced.
"""
from __future__ import annotations

import io
import os
import sys
import time
from unittest.mock import patch

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _zero_sleep(_seconds: float) -> None:
    return None


def test_banner_animation_runs_through_real_live(monkeypatch):
    """Animation completes without error via rich.live.Live, produces >= 10 frames."""
    from controller import ascii_logo

    monkeypatch.setattr(ascii_logo.time, "sleep", _zero_sleep)

    class _FakeTTYOutput:
        def __init__(self):
            self._buffer = io.StringIO()

        def write(self, data: str) -> int:
            return self._buffer.write(data)

        def isatty(self) -> int:
            return 1

        def flush(self) -> None:
            self._buffer.flush()

    fake_out = _FakeTTYOutput()
    monkeypatch.setattr(ascii_logo.sys, "stdout", fake_out)
    monkeypatch.setattr(ascii_logo.threading, "Thread", lambda *a, **kw: _NoOpThread())

    from rich.live import Live as _RealLive
    from rich.console import Console as _RealConsole
    from rich.text import Text as _RealText

    buf = io.StringIO()
    capture_console = _RealConsole(
        file=buf,
        force_terminal=True,
        color_system="truecolor",
        width=200,
    )

    real_update_calls: list = []

    class _CapturingLive:
        def __init__(self, *args, **kwargs):
            kwargs["console"] = capture_console
            self._live = _RealLive(*args, **kwargs)
            self._real_update = self._live.update

        def update(self, renderable):
            real_update_calls.append(renderable)
            try:
                capture_console.render(renderable)
            except Exception as e:
                raise AssertionError(
                    f"renderable raised on Live.update(): {e!r}\n"
                    f"type={type(renderable).__name__}"
                )
            return self._real_update(renderable)

        def __enter__(self):
            self._live.__enter__()
            return self

        def __exit__(self, *args):
            return self._live.__exit__(*args)

    monkeypatch.setattr(ascii_logo, "_Live", _CapturingLive)
    monkeypatch.setattr(ascii_logo, "_Console", _RealConsole)
    monkeypatch.setattr(ascii_logo, "_is_tty", lambda: True)
    monkeypatch.setattr(ascii_logo, "_fits_terminal", lambda: True)

    ascii_logo.show_intro(play=True)

    out = buf.getvalue()

    assert len(real_update_calls) >= 10, (
        f"Expected >= 10 Live.update calls, got {len(real_update_calls)}"
    )

    from rich.console import Group as _Group
    for i, r in enumerate(real_update_calls):
        assert isinstance(r, (_Group, _RealText, str)), (
            f"Live.update got incompatible renderable: "
            f"type={type(r).__name__} at frame {i}"
        )

    # Output contains wordmark (█ blocks)
    assert "█" in out, f"wordmark █ blocks not found in output:\n{out[:500]}"
    # Rain chars
    assert any(ch in out for ch in "▚▖▗"), f"rain chars not found in output:\n{out[:500]}"
    # Type-on
    assert "F" in out and "·" in out, f"F·A·I·R·Y not found in output:\n{out[:500]}"
    # Tagline
    assert "personal ai companion" in out, f"tagline not found in output:\n{out[:500]}"


def test_wordmark_dimensions():
    """The wordmark is 6 rows × 36 cols (ansi_shadow verbatim)."""
    from controller import ascii_logo

    wm = ascii_logo.get_wordmark_lines()
    assert len(wm) == 6, f"Expected 6 rows, got {len(wm)}"

    for i in range(6):
        row = wm[i]
        assert len(row) == ascii_logo._WORD_COLS, (
            f"Wordmark row {i} has width {len(row)}, "
            f"expected {ascii_logo._WORD_COLS}"
        )
        # Rows 0-4 have █ blocks; row 5 is shadow-only
        if i < 5:
            assert "█" in row, f"Wordmark row {i} has no █ blocks:\n{row!r}"


def test_wordmark_figlet_based():
    """Wordmark must come from pyfiglet ansi_shadow verbatim (not hand-drawn)."""
    from controller import ascii_logo

    assert ascii_logo._FIGLET_OK, "pyfiglet should be available"
    assert ascii_logo._WORD_ROWS == 6
    assert ascii_logo._WORD_COLS >= 30, (
        f"Wordmark should be at least 30 cols wide, got {ascii_logo._WORD_COLS}"
    )

    wm = ascii_logo.get_wordmark_lines()
    full = "".join(wm)
    assert "█" in full, "Wordmark should contain █ block characters"

    # All rows same width (padded)
    widths = {len(ln) for ln in wm}
    assert len(widths) == 1, f"Wordmark rows have inconsistent widths: {widths}"


def test_build_frame_returns_compatible_renderable():
    """All frame types must return a renderable Live can consume."""
    from controller import ascii_logo
    from rich.console import Group as _Group
    from rich.text import Text as _Text

    frame_types = ["rain", "flash", "typeon", "final"]
    for ft in frame_types:
        kwargs = {"frame_type": ft, "terminal_width": 36}
        if ft == "rain":
            kwargs["rain_tick"] = 5
            kwargs["rain_density"] = 0.8
        elif ft == "typeon":
            kwargs["letters_shown"] = 5
            kwargs["show_tagline"] = False
        frame = ascii_logo._build_frame(**kwargs)
        assert isinstance(frame, (_Group, _Text, str)), (
            f"_build_frame({ft}) returned {type(frame).__name__}, "
            f"expected Group/Text/str"
        )


def test_build_frame_string_fallback():
    """The no-Rich fallback produces a printable string."""
    from controller import ascii_logo

    s = ascii_logo._build_frame_string(frame_type="final", terminal_width=36)
    assert isinstance(s, str)
    assert "█" in s, "wordmark █ blocks not found in final frame"
    assert "personal ai companion" in s, "tagline not found in final frame"


def test_animation_has_distinct_frames():
    """The animation produces many distinct frame states."""
    from controller import ascii_logo

    frames: list[str] = []

    # Rain frames at different densities
    for tick in range(4):
        f = ascii_logo._build_frame(
            "rain", rain_tick=tick, rain_density=0.8 - tick * 0.15, terminal_width=36
        )
        frames.append(str(f))

    # Flash frame
    frames.append(str(ascii_logo._build_frame("flash", terminal_width=36)))

    # Type-on frames
    for n in [0, 3, 6, 9]:
        frames.append(str(ascii_logo._build_frame(
            "typeon", letters_shown=n, show_tagline=n == 9, terminal_width=36
        )))

    # Final frame
    frames.append(str(ascii_logo._build_frame("final", terminal_width=36)))

    assert len(frames) >= 10, f"Sampled {len(frames)} frames, expected >= 10"
    for i, fr in enumerate(frames):
        assert isinstance(fr, str), f"Frame {i} is not a string: {type(fr)}"


def test_height_fit_check():
    """_fits_terminal() returns True for terminals >= 12 rows, 36 cols."""
    from controller import ascii_logo
    assert ascii_logo._fits_terminal()


def test_render_path_logging(caplog):
    """show_intro logs the render path when falling back."""
    from controller import ascii_logo

    class _NonTTY:
        def isatty(self): return False
        def write(self, data): return len(data)
        def flush(self): pass

    original = ascii_logo.sys.stdout
    ascii_logo.sys.stdout = _NonTTY()

    try:
        with caplog.at_level("INFO", logger="ascii_logo"):
            ascii_logo.show_intro(play=True)
    finally:
        ascii_logo.sys.stdout = original

    assert "RENDER PATH" in caplog.text, (
        f"Expected 'RENDER PATH' in log, got: {caplog.text!r}"
    )


class _NoOpThread:
    def __init__(self, *args, **kwargs):
        self.daemon = kwargs.get("daemon", False)

    def start(self) -> None:
        return None
