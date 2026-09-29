"""
Lightweight per-message language detection.

Uses a Unicode-range heuristic — no heavy library needed:
  Bengali block U+0980–U+09FF in the message → "bn"
  Otherwise → "en"

"mixed" is treated as "bn" (Bengali has priority when present).
"""
from __future__ import annotations

# Unicode code point ranges
_BENGALI_START = 0x0980
_BENGALI_END   = 0x09FF


def detect_language(text: str) -> str:
    """
    Detect the language of `text` using a Unicode-range heuristic.

    Returns:
        "bn" — if any character falls in the Bengali block (U+0980–U+09FF).
        "en" — otherwise (English or any non-Bengali script).

    The heuristic gives Bengali priority when mixed scripts appear, matching
    the spec: "mixed → bn".
    """
    for ch in text:
        cp = ord(ch)
        if _BENGALI_START <= cp <= _BENGALI_END:
            return "bn"
    return "en"
