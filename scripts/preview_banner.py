#!/usr/bin/env python3
"""
Preview script for the FAIRY animated ASCII logo.

Renders snapshots of each animation phase so Master can verify without
running the full interactive boot sequence.

Usage:
    python scripts/preview_banner.py
"""
from __future__ import annotations

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONIOENCODING"] = "utf-8"

from controller.ascii_logo import (
    show_intro,
    _render_wordmark,
    _WORDMARK_COLORED,
    _build_frame,
    _make_rain_frame,
    _get_clock,
    _enable_ansi,
    _C_DIM,
    _C_MID,
    _C_WHT,
    _C_RST,
    _WORDMARK_RAW,
)
import random

SEP = "─" * 60


def render_wordmark():
    """Render the static wordmark (ansi_shadow verbatim)."""
    _enable_ansi()
    lines = ["", SEP]
    lines.append("│  WORDMARK (pyfiglet ansi_shadow verbatim, 36 cols × 6 rows)")
    lines.append(SEP)
    for wl in _WORDMARK_COLORED:
        lines.append(f"  {wl}")
    lines.append(SEP)
    lines.append("")
    return "\n".join(lines)


def render_rain_phase():
    """Render a rain storm frame (mid-density)."""
    _enable_ansi()
    rain = _make_rain_frame(tick=10, width=40, height=6, density=0.5)
    lines = ["", SEP]
    lines.append("│  PHASE 1: RAIN STORM (fades out)")
    lines.append(SEP)
    for rl in rain:
        lines.append(f"  {rl}")
    lines.append(SEP)
    lines.append("")
    return "\n".join(lines)


def render_flash_phase():
    """Render the single white flash frame."""
    _enable_ansi()
    lines = ["", SEP]
    lines.append("│  PHASE 2: WHITE FLASH (1 frame)")
    lines.append(SEP)
    for wl in _WORDMARK_COLORED:
        lines.append(f"  {_C_WHT}{wl}{_C_RST}")
    lines.append(SEP)
    lines.append("")
    return "\n".join(lines)


def render_final_frame():
    """Render the final static frame."""
    _enable_ansi()
    lines = ["", SEP]
    lines.append("│  FINAL FRAME (wordmark + type-on + tagline + clock)")
    lines.append(SEP)
    for wl in _WORDMARK_COLORED:
        lines.append(f"  {wl}")
    lines.append("")
    lines.append(f"  {_C_WHT}F·A·I·R·Y{_C_RST}")
    lines.append(f"  {_C_DIM}personal ai companion{_C_RST}")
    lines.append(f"  {_C_MID}{_get_clock()}{_C_RST}")
    lines.append(SEP)
    lines.append("")
    return "\n".join(lines)


def main():
    print(render_wordmark())
    print(render_rain_phase())
    print(render_flash_phase())
    print(render_final_frame())


if __name__ == "__main__":
    main()
