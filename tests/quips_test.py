"""
Tests for the personality/animation system in controller/quips.py.

Covers:
  - KaomojiSubsystem (no-repeat until exhausted, with_parenthetical, pick_kaomoji_only)
  - random_quip()
  - loading_line() — both brain-specific and default
  - next_subsystem() / current_subsystem()
  - SPINNER_PERSONALITY_LINES — merged original + new thinking phrases
  - PRESERVED API: ack(), thinking(), retry(), fallback(), synthesize()
  - Palette constants exist and are valid hex values
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from controller import quips
from controller.quips import (
    KaomojiSubsystem,
    BRAIN_CORE, BRAIN_HERMES, BRAIN_CLAUDE, BRAIN_WEB,
    BRAIN_COMPUTER, BRAIN_FILE,
    SUBSYSTEM_CYCLE,
    SPARKLE_SET, SPINNER_FRAMES,
    SPINNER_PERSONALITY_LINES,
    QUIPS,
    BRAIN_LOADING_LINES,
    random_quip, loading_line, next_subsystem, current_subsystem,
    typewriter,
    ack, thinking, retry, fallback, synthesize,
)


# ─────────────────────────────────────────────────────────────────
# Palette constants
# ─────────────────────────────────────────────────────────────────
class TestPalette:
    def test_fairy_blue_is_valid_hex(self):
        assert quips.FAIRY_BLUE.startswith("#") and len(quips.FAIRY_BLUE) == 7

    def test_fairy_light_is_valid_hex(self):
        assert quips.FAIRY_LIGHT.startswith("#") and len(quips.FAIRY_LIGHT) == 7

    def test_fairy_white_is_valid_hex(self):
        assert quips.FAIRY_WHITE.startswith("#") and len(quips.FAIRY_WHITE) == 7

    def test_lime_accent_is_valid_hex(self):
        assert quips.LIME_ACCENT.startswith("#") and len(quips.LIME_ACCENT) == 7

    def test_soft_purple_is_valid_hex(self):
        assert quips.SOFT_PURPLE.startswith("#") and len(quips.SOFT_PURPLE) == 7


# ─────────────────────────────────────────────────────────────────
# KaomojiSubsystem — no-repeat-until-exhausted
# ─────────────────────────────────────────────────────────────────
class TestKaomojiSubsystem:
    def test_pick_returns_from_own_pool(self):
        sub = KaomojiSubsystem("TEST", "#ffffff", ("(╯°□°)╯", "┻━┻", "tableflip"))
        picks = [sub.pick() for _ in range(50)]
        for k in picks:
            assert k in ("(╯°□°)╯", "┻━┻", "tableflip")

    def test_no_repeat_until_exhausted(self):
        sub = KaomojiSubsystem("TEST", "#ffffff", ("A", "B", "C"))
        seen: set[str] = set()
        for _ in range(3):
            k = sub.pick()
            assert k not in seen, f"Got repeat {k!r} before exhaustion"
            seen.add(k)
        # 4th pick must be one of the already-seen items
        fourth = sub.pick()
        assert fourth in ("A", "B", "C")

    def test_reset_restarts_cycle(self):
        sub = KaomojiSubsystem("TEST", "#ffffff", ("X", "Y"))
        # Exhaust
        first = sub.pick()
        second = sub.pick()
        assert first != second
        # Manually reset (simulates a new session)
        sub._reset_exhaustion()
        # Both should be available again
        picks = {sub.pick() for _ in range(20)}
        assert len(picks) == 2  # both X and Y should appear

    def test_with_parenthetical_returns_string(self):
        sub = KaomojiSubsystem(
            "TEST", "#ffffff",
            ("(・ω・)",),
            parentheticals=("idle mode",),
        )
        result = sub.with_parenthetical()
        assert "(・ω・)" in result
        assert "idle mode" in result

    def test_with_parenthetical_no_parens_falls_back_to_pick(self):
        sub = KaomojiSubsystem("TEST", "#ffffff", ("(・ω・)",))
        result = sub.with_parenthetical()
        assert result == "(・ω・)"

    def test_pick_kaomoji_only(self):
        sub = KaomojiSubsystem(
            "TEST", "#ffffff",
            ("(・ω・)",),
            parentheticals=("foo",),
        )
        # pick_kaomoji_only should not include parenthetical
        assert sub.pick_kaomoji_only() == "(・ω・)"
        assert sub.pick_kaomoji_only() == "(・ω・)"

    def test_subsystem_has_name_and_color(self):
        assert BRAIN_CORE.name == "BRAIN.CORE"
        assert BRAIN_CORE.color_hex == quips.FAIRY_BLUE
        assert BRAIN_HERMES.name == "BRAIN.HERMES"
        assert BRAIN_HERMES.color_hex == quips.SOFT_PURPLE


# ─────────────────────────────────────────────────────────────────
# Pre-built subsystems exist and are reachable
# ─────────────────────────────────────────────────────────────────
class TestSubsystems:
    @pytest.mark.parametrize("sub", [
        BRAIN_CORE, BRAIN_HERMES, BRAIN_CLAUDE, BRAIN_WEB,
        BRAIN_COMPUTER, BRAIN_FILE,
    ])
    def test_subsystem_is_kaomojisubsystem(self, sub):
        assert isinstance(sub, KaomojiSubsystem)

    def test_subsystem_cycle_length(self):
        assert len(SUBSYSTEM_CYCLE) == 6

    def test_subsystem_cycle_order(self):
        assert SUBSYSTEM_CYCLE[0] is BRAIN_CORE
        assert SUBSYSTEM_CYCLE[1] is BRAIN_HERMES

    def test_all_subsystems_in_cycle(self):
        all_subs = {BRAIN_CORE, BRAIN_HERMES, BRAIN_CLAUDE, BRAIN_WEB,
                    BRAIN_COMPUTER, BRAIN_FILE}
        assert set(SUBSYSTEM_CYCLE) == all_subs


# ─────────────────────────────────────────────────────────────────
# random_quip()
# ─────────────────────────────────────────────────────────────────
class TestRandomQuip:
    def test_returns_from_quips(self):
        q = random_quip()
        assert q in QUIPS

    def test_returns_string(self):
        assert isinstance(random_quip(), str)

    def test_multiple_calls_cover_quips(self):
        seen = {random_quip() for _ in range(200)}
        # With 200 picks we should see at least half the quips
        assert len(seen) >= len(QUIPS) // 2


# ─────────────────────────────────────────────────────────────────
# loading_line()
# ─────────────────────────────────────────────────────────────────
class TestLoadingLine:
    @pytest.mark.parametrize("key", ["hermes", "claude_code", "web_search", "computer_control"])
    def test_brain_specific_returns_matching_brain(self, key):
        line = loading_line(key)
        assert isinstance(line, str)
        # The line should come from the brain-specific pool, not the
        # default personality lines — verify by intersecting with both
        # pools and ensuring the brain pool is the source.
        _, _, lines = BRAIN_LOADING_LINES[key]
        # With enough samples, at least one should come from brain pool
        seen = {loading_line(key) for _ in range(50)}
        assert seen & set(lines), f"No brain-specific lines returned for {key!r}"

    def test_none_returns_from_personality_lines(self):
        line = loading_line(None)
        assert line in SPINNER_PERSONALITY_LINES

    def test_unknown_key_falls_back_to_personality_lines(self):
        line = loading_line("totally_unknown_key_123")
        assert line in SPINNER_PERSONALITY_LINES

    def test_returns_string(self):
        for key in [None, "hermes", "web_search"]:
            assert isinstance(loading_line(key), str)


# ─────────────────────────────────────────────────────────────────
# BRAIN_LOADING_LINES mapping
# ─────────────────────────────────────────────────────────────────
class TestBrainLoadingLines:
    def test_all_expected_keys_present(self):
        expected = {"hermes", "claude_code", "web_search", "computer_control"}
        assert set(BRAIN_LOADING_LINES.keys()) == expected

    def test_each_entry_has_name_color_lines(self):
        for key, entry in BRAIN_LOADING_LINES.items():
            name, color, lines = entry
            assert isinstance(name, str)
            assert color.startswith("#")
            assert isinstance(lines, tuple)
            assert len(lines) >= 1


# ─────────────────────────────────────────────────────────────────
# next_subsystem() / current_subsystem()
# ─────────────────────────────────────────────────────────────────
class TestSubsystemCycle:
    def test_next_subsystem_returns_subsystem(self):
        sub = next_subsystem()
        assert isinstance(sub, KaomojiSubsystem)

    def test_current_subsystem_returns_subsystem(self):
        sub = current_subsystem()
        assert isinstance(sub, KaomojiSubsystem)

    def test_full_cycle_returns_all_unique(self):
        seen = set()
        for _ in range(len(SUBSYSTEM_CYCLE) + 2):
            sub = next_subsystem()
            seen.add(sub)
        # After a full cycle + 1, we've seen all subsystems
        assert len(seen) == len(SUBSYSTEM_CYCLE)

    def test_subsystem_cycle_is_list(self):
        assert isinstance(SUBSYSTEM_CYCLE, list)


# ─────────────────────────────────────────────────────────────────
# SPARKLE_SET & SPINNER_FRAMES
# ─────────────────────────────────────────────────────────────────
class TestSparklesAndFrames:
    def test_sparkle_set_has_sparkles(self):
        assert len(SPARKLE_SET) >= 4
        for s in SPARKLE_SET:
            assert isinstance(s, str)
            assert len(s) <= 3  # sparkles are small chars

    def test_spinner_frames_is_tuple(self):
        assert isinstance(SPINNER_FRAMES, tuple)
        assert len(SPINNER_FRAMES) >= 4


# ─────────────────────────────────────────────────────────────────
# SPINNER_PERSONALITY_LINES — merged from original + new
# ─────────────────────────────────────────────────────────────────
class TestSpinnerPersonalityLines:
    def test_includes_original_thinking_phrases(self):
        original_phrases = [
            "Consulting the wisdom...",
            "Pulling threads together...",
            "Waving my wand...",
            "Consulting the oracle...",
        ]
        for phrase in original_phrases:
            assert phrase in SPINNER_PERSONALITY_LINES, f"Missing: {phrase!r}"

    def test_includes_new_personality_lines(self):
        new_phrases = [
            "consulting the archives…",
            "twirling asynchronously…",
            "doing science, do not perceive me…",
        ]
        for phrase in new_phrases:
            assert phrase in SPINNER_PERSONALITY_LINES, f"Missing: {phrase!r}"

    def test_is_not_empty(self):
        assert len(SPINNER_PERSONALITY_LINES) >= 10


# ─────────────────────────────────────────────────────────────────
# QUIPS — thinking facts
# ─────────────────────────────────────────────────────────────────
class TestQuipsThinkingFacts:
    def test_quips_is_tuple(self):
        assert isinstance(QUIPS, tuple)

    def test_quips_not_empty(self):
        assert len(QUIPS) >= 5

    def test_all_quips_are_short_strings(self):
        for q in QUIPS:
            assert isinstance(q, str)
            assert len(q) <= 120  # must fit on one console line


# ─────────────────────────────────────────────────────────────────
# Preserved API — existing quips functions still work
# ─────────────────────────────────────────────────────────────────
class TestPreservedAPI:
    def test_ack_returns_str(self):
        assert isinstance(ack(), str)

    def test_ack_returns_from_pool(self):
        assert ack() in quips.ACK_PHRASES

    def test_thinking_returns_str(self):
        assert isinstance(thinking(), str)

    def test_thinking_returns_from_spinner_personality_lines(self):
        assert thinking() in SPINNER_PERSONALITY_LINES

    def test_retry_returns_from_pool(self):
        assert retry() in quips.RETRY_PHRASES

    def test_fallback_returns_from_pool(self):
        assert fallback() in quips.FALLBACK_PHRASES

    def test_synthesize_returns_from_pool(self):
        assert synthesize() in quips.SYNTHESIZE_PHRASES


# ─────────────────────────────────────────────────────────────────
# typewriter helper
# ─────────────────────────────────────────────────────────────────
class TestTypewriter:
    def test_typewriter_does_not_raise(self, capsys):
        # Should not crash even with empty string or special chars
        typewriter("")
        typewriter("hello world")
        typewriter("special chars: @#$%")
        # Nothing guaranteed in stdout; just verify it didn't raise
        captured = capsys.readouterr()
        assert isinstance(captured.out, str)
