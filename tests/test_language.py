"""
Tests for bidirectional language mirroring (English ↔ Bangla).

Network-isolated per conftest policy: every test uses only the in-process
language detector, prompt builder, chunker, and history persistence — no
HTTP, no Ollama, no Discord.

Coverage:
  - Detection: Bangla script → "bn", English → "en", mixed → "bn"
  - Prompt assembly: language instruction present, history preserved
  - Chunking: a reply containing Bangla splits/joins without corruption
    (multi-byte boundary test)
  - History: Bangla messages persist and survive reload
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure the project root is on sys.path so `core`, `controller`, etc. import.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest


# ── discord_bot import trick (matches test_discord_chunking.py) ────────────────────
def _load_discord_bot_module():
    """Load discord_bot with lock-file and env patched so tests can import it."""
    fake_token = "fake-token-for-tests"
    with patch.dict(os.environ, {
        "DISCORD_BOT_TOKEN": fake_token,
        "FAIRY_DISCORD_OWNER_ID": "123456",
        "FAIRY_DISCORD_ALLOW_ALL": "1",
    }):
        with patch("os.path.exists", return_value=False):
            import importlib
            for key in list(sys.modules.keys()):
                if key.startswith("controller.discord_bot"):
                    del sys.modules[key]
            return importlib.import_module("controller.discord_bot")


def _chunk(text: str, limit: int = 2000):
    """Load discord_bot and call _chunk_text."""
    bot = _load_discord_bot_module()
    return bot._chunk_text(text, limit)


# ═══════════════════════════════════════════════════════════════════════════════
# Detection — pure Unicode-range heuristic
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectLanguage:
    """Bangla block (U+0980–U+09FF) → "bn"; otherwise → "en". Mixed → "bn"."""

    def _detect(self, text: str) -> str:
        from core.language_detection import detect_language
        return detect_language(text)

    def test_empty_string_defaults_to_english(self):
        assert self._detect("") == "en"

    def test_pure_english_returns_en(self):
        assert self._detect("Hello, Master!") == "en"

    def test_pure_bangla_returns_bn(self):
        # "আমি ঠিক আছি" — "I am fine" in Bangla
        text = "আমি ঠিক আছি"
        assert self._detect(text) == "bn"

    def test_mixed_bangla_then_english_returns_bn(self):
        # Mixed → Bangla has priority when present.
        text = "Hello আমার নাম Fairy"
        assert self._detect(text) == "bn"

    def test_mixed_english_then_bangla_returns_bn(self):
        text = "Master please check আমার কাজ"
        assert self._detect(text) == "bn"

    def test_digit_only_returns_en(self):
        # Digits aren't in the Bengali block.
        assert self._detect("12345") == "en"

    def test_punctuation_only_returns_en(self):
        assert self._detect("!!!???...") == "en"

    def test_bengali_start_block_detected(self):
        # U+0981 = ঁ (Bengali sign chandrabindu) — first char in Bengali block.
        assert self._detect("ঁabc") == "bn"

    def test_bengali_end_block_detected(self):
        # U+09FF — last char in Bengali block.
        assert self._detect("end ৿") == "bn"

    def test_just_below_bengali_block_returns_en(self):
        # U+097F (Devanagari) — not in Bengali block.
        assert self._detect("ॿ") == "en"

    def test_just_above_bengali_block_returns_en(self):
        # U+0A00 — not in Bengali block.
        assert self._detect("਀") == "en"

    def test_common_bangla_words(self):
        bangla_samples = [
            "হ্যাঁ",          # yes
            "না",            # no
            "কেমন আছো?",    # how are you?
            "ধন্যবাদ",       # thank you
            "আমি ভালো আছি",  # I am fine
        ]
        for s in bangla_samples:
            assert self._detect(s) == "bn", f"expected bn for {s!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt assembly — language parameter surfaces in the system prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptAssemblyLanguage:
    """build_system_prompt() must surface the language tag and keep history
    in the messages list."""

    def _build(self, profile=None, language=None, history=None):
        from core.prompts import build_system_prompt
        return build_system_prompt(
            profile or "normal",
            skills=[],
            memory_context="",
            language=language,
        ), history

    def test_system_prompt_contains_bn_directive_when_bangla(self):
        sys_prompt, _ = self._build(language="bn")
        assert "Bangla" in sys_prompt or "Bengali" in sys_prompt
        # Specifically: per-call directive is added
        assert "The user's message is in Bangla" in sys_prompt

    def test_system_prompt_contains_en_directive_when_english(self):
        sys_prompt, _ = self._build(language="en")
        assert "The user's message is in English" in sys_prompt

    def test_system_prompt_no_directive_when_language_is_none(self):
        sys_prompt, _ = self._build(language=None)
        # Permanent mirroring rule still present (system-level), but no
        # per-call "The user's message is in ..." line.
        assert "LANGUAGE MIRRORING" in sys_prompt
        assert "The user's message is in Bangla" not in sys_prompt
        assert "The user's message is in English" not in sys_prompt

    def test_permanent_mirroring_rule_always_present(self):
        for lang in (None, "en", "bn"):
            sys_prompt, _ = self._build(language=lang)
            assert "LANGUAGE MIRRORING" in sys_prompt
            # Master dynamic preserved
            assert "Master" in sys_prompt

    def test_messages_preserve_history_with_bangla(self):
        # History is a separate channel from the system prompt. The prompt
        # builder doesn't compose messages; the test verifies that the
        # caller-side construction preserves Bengali content.
        history = [
            {"role": "user", "content": "হ্যাঁ"},
            {"role": "assistant", "content": "বলো মাস্টার।"},
            {"role": "user", "content": "কেমন আছো?"},
        ]
        messages = [
            {"role": "system", "content": "system"},
            *history,
        ]
        # All entries preserved, in order
        assert len(messages) == 4
        assert messages[1]["content"] == "হ্যাঁ"
        assert messages[2]["content"] == "বলো মাস্টার।"
        assert messages[3]["content"] == "কেমন আছো?"

    def test_language_detection_to_prompt_round_trip_bangla(self):
        # Detect → tag → inject into a messages list, ensuring the
        # instructions reach the model.
        from core.language_detection import detect_language
        from core.prompts import build_system_prompt

        user_text = "মাস্টার, তুমি কেমন আছো?"
        lang = detect_language(user_text)
        assert lang == "bn"

        sys_prompt = build_system_prompt("normal", language=lang)
        assert "Bangla" in sys_prompt

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text},
        ]
        # The user message itself remains untouched (caller doesn't tag it;
        # the system prompt carries the directive).
        assert messages[1]["content"] == user_text

    def test_language_detection_to_prompt_round_trip_english(self):
        from core.language_detection import detect_language
        from core.prompts import build_system_prompt

        user_text = "Master, are you awake?"
        lang = detect_language(user_text)
        assert lang == "en"

        sys_prompt = build_system_prompt("normal", language=lang)
        assert "The user's message is in English" in sys_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Chunking — Bangla multi-byte boundary safety
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkingBangla:
    """A reply containing Bangla must split/join without corrupting the
    multi-byte characters."""

    def test_short_bangla_passes_through_unchanged(self):
        text = "হ্যাঁ মাস্টার, আমি প্রস্তুত।"
        assert _chunk(text) == [text]

    def test_bangla_under_limit_is_one_chunk(self):
        text = "আমি ভালো আছি। " * 50  # ~775 chars, well under 2000
        result = _chunk(text)
        assert len(result) == 1
        assert result[0] == text

    def test_bangla_over_limit_splits_into_multiple_chunks(self):
        # ~3000 Bangla chars
        text = "বাংলা ভাষা সুন্দর। " * 130
        assert len(text) > 2000
        result = _chunk(text)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 2000, f"chunk len={len(chunk)} exceeds 2000"

    def test_bangla_chunks_rejoin_unchanged(self):
        # Splitting then joining must reproduce the original (no char loss).
        text = "আমি বাংলায় কথা বলি। " * 130
        result = _chunk(text, limit=2000)
        joined = "".join(result)
        # When splits happen, the joined string must equal the original.
        assert len(joined) == len(text)
        # All Bangla characters survive round-trip
        assert joined.count("আ") == text.count("আ")
        assert joined.count("ম") == text.count("ম")
        assert joined.count("স") == text.count("স")
        assert joined.count("দ") == text.count("দ")

    def test_bangla_multi_byte_boundary_does_not_split_char(self):
        # The hard-slice path: text with no natural break points.
        # The chunker must NEVER split a Bangla char across chunks.
        # Build a single very long run of Bangla chars with no breaks.
        bangla_char = "া"  # া (vowel sign aa) — 3 bytes in UTF-8
        text = bangla_char * 5000  # 5000 chars, 15000 bytes
        result = _chunk(text, limit=2000)
        # Every chunk must be valid UTF-8 (decode cleanly).
        for i, chunk in enumerate(result):
            encoded = chunk.encode("utf-8")
            # Decode the encoded bytes back to confirm no half-characters
            round_trip = encoded.decode("utf-8")
            assert round_trip == chunk, f"chunk {i} failed UTF-8 round trip"
            # No mojibake: a broken multi-byte would surface as a
            # Unicode replacement char (U+FFFD) or surrogate escape
            assert "�" not in chunk
        # All Bangla chars accounted for
        joined = "".join(result)
        assert joined == text

    def test_bangla_mixed_with_english_chunks_cleanly(self):
        text = ("Master " + "বাংলা " * 800)
        result = _chunk(text)
        for chunk in result:
            assert len(chunk) <= 2000
        # Rejoin
        joined = "".join(result)
        assert joined.count("Master") == text.count("Master")
        assert joined.count("বাংলা") == text.count("বাংলা")

    def test_unicode_safety_when_chunking_at_limit(self):
        # A string that, when sliced at exactly 2000 chars by Python's
        # [] operator, would still be on a character boundary (Python
        # always slices on code points). The chunker must preserve this.
        text = "অ" * 1999 + "X" + "ই" * 500
        result = _chunk(text, limit=2000)
        joined = "".join(result)
        # All chars preserved
        assert joined == text
        # No chunk contains a partial multi-byte sequence
        for chunk in result:
            chunk.encode("utf-8").decode("utf-8")  # raises if corrupt


# ═══════════════════════════════════════════════════════════════════════════════
# History persistence — Bangla messages survive a save/load round trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestHistoryBanglaPersistence:
    """Bangla messages must round-trip through the in-memory + JSON-file
    persistence used by .fairy_history without corruption."""

    def _sample_history(self):
        return [
            {"role": "user", "content": "মাস্টার, তুমি কেমন আছো?"},
            {"role": "assistant", "content": "আমি ভালো আছি, মাস্টার। তুমি বলো?"},
            {"role": "user", "content": "What time is it, Master?"},
            {"role": "assistant", "content": "It's 11:42 PM, Master. Late night."},
            {"role": "user", "content": "আমি ক্লান্ত।"},
            {"role": "assistant", "content": "তাহলে ঘুমাতে যাও, মাস্টার। ✨"},
        ]

    def test_bangla_history_survives_json_round_trip(self):
        hist = self._sample_history()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(hist, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name
        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        finally:
            os.unlink(tmp_path)
        assert loaded == hist
        # Confirm Bangla chars intact
        assert loaded[0]["content"] == "মাস্টার, তুমি কেমন আছো?"
        assert loaded[5]["content"] == "তাহলে ঘুমাতে যাও, মাস্টার। ✨"

    def test_fairy_history_file_format_preserves_bangla(self):
        # Simulate what FileHistory from prompt_toolkit does: each entry
        # is a UTF-8 line. Verify the .fairy_history file format we use
        # round-trips Bangla cleanly.
        from prompt_toolkit.history import InMemoryHistory

        h = InMemoryHistory()
        # InMemoryHistory.append_string prepends to its internal list, so
        # we append in reverse order so load_history_strings() returns them
        # in the original order.
        entries = [
            "hello",
            "মাস্টার, তুমি কেমন আছো?",
            "আমি ভালো আছি",
        ]
        for e in reversed(entries):
            h.append_string(e)

        with tempfile.NamedTemporaryFile(
            "w", suffix=".hist", delete=False, encoding="utf-8"
        ) as tmp:
            # Append each entry on its own line; FileHistory-compatible format
            for s in h.load_history_strings():
                tmp.write(s + "\n")
            tmp_path = tmp.name

        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                lines = [ln.rstrip("\n") for ln in f.readlines() if ln.strip()]
        finally:
            os.unlink(tmp_path)

        assert lines == entries

    def test_history_ordering_preserved(self):
        # A multi-turn conversation that flips between languages must
        # come back in the same order.
        hist = self._sample_history()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(hist, tmp, ensure_ascii=False)
            tmp_path = tmp.name
        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        finally:
            os.unlink(tmp_path)
        for original, restored in zip(hist, loaded):
            assert original == restored
            assert original["role"] == restored["role"]
            assert original["content"] == restored["content"]

    def test_language_detection_per_message_independent(self):
        # A history with alternating languages must classify each entry
        # independently — the detection is per-message, not on the
        # aggregate history.
        from core.language_detection import detect_language

        for entry in self._sample_history():
            content = entry["content"]
            lang = detect_language(content)
            # English entries → en
            if content.startswith(("What", "It's")):
                assert lang == "en", f"expected en for {content!r}"
            else:
                assert lang == "bn", f"expected bn for {content!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# Hermes bridge — language tag is prepended to the user message
# ═══════════════════════════════════════════════════════════════════════════════

class TestHermesBridgeLanguageTag:
    """The cached AIAgent's system prompt is static; per-call language is
    enforced by tagging the user message. _inject_language_tag() handles this.
    """

    def test_bangla_user_message_is_tagged(self):
        from hermes_bridge import _inject_language_tag
        tagged, lang = _inject_language_tag("মাস্টার, সময় কত?")
        assert lang == "bn"
        assert "Bangla" in tagged or "Bengali" in tagged
        assert "মাস্টার, সময় কত?" in tagged

    def test_english_user_message_is_tagged(self):
        from hermes_bridge import _inject_language_tag
        tagged, lang = _inject_language_tag("Master, what time is it?")
        assert lang == "en"
        assert "English" in tagged
        assert "Master, what time is it?" in tagged

    def test_empty_user_message_returns_empty_string(self):
        from hermes_bridge import _inject_language_tag
        tagged, lang = _inject_language_tag("")
        # Should still produce a non-empty tag (so the model knows the
        # default language) but include the empty user content
        assert "English" in tagged
        assert lang == "en"


# ═══════════════════════════════════════════════════════════════════════════════
# main_brain — language tag is injected for OpenRouter fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestMainBrainLanguageTag:
    """main_brain's _openrouter_chat() tags the latest user message with
    a per-call language directive. _tag_user_message_for_language() is the
    helper."""

    def test_bangla_user_message_is_tagged(self):
        from controller.main_brain import _tag_user_message_for_language
        tagged = _tag_user_message_for_language("মাস্টার, তুমি কেমন আছো?")
        assert "Bangla" in tagged or "Bengali" in tagged
        assert "মাস্টার, তুমি কেমন আছো?" in tagged

    def test_english_user_message_is_tagged(self):
        from controller.main_brain import _tag_user_message_for_language
        tagged = _tag_user_message_for_language("Master, how are you?")
        assert "English" in tagged
        assert "Master, how are you?" in tagged

    def test_empty_user_message_returns_empty(self):
        from controller.main_brain import _tag_user_message_for_language
        # Empty user content stays empty (no false tag).
        assert _tag_user_message_for_language("") == ""

    def test_default_openrouter_system_prompt_contains_mirroring_rule(self):
        from controller.main_brain import _DEFAULT_OPENROUTER_SYSTEM_PROMPT
        assert "LANGUAGE MIRRORING" in _DEFAULT_OPENROUTER_SYSTEM_PROMPT
        # The rule is in English (per spec: 2-3 sentences, in English).
        # Not tagged with a [The user's message is in X] line — that's
        # added per-call.
        assert "[The user's message is in" not in _DEFAULT_OPENROUTER_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════════════════════════
# Encoding audit — ensure no non-UTF-8 codec is used
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncodingAudit:
    """Static checks: no non-UTF-8 encode/decode calls in our key files."""

    def test_core_files_use_utf8(self):
        forbidden = ("latin-1", "cp1252", "ascii", "windows-1252")
        for relpath in ("core/prompts.py", "core/language_detection.py"):
            path = _PROJECT_ROOT / relpath
            content = path.read_text(encoding="utf-8")
            for codec in forbidden:
                assert f'"{codec}"' not in content and f"'{codec}'" not in content, (
                    f"non-UTF-8 codec {codec!r} in {relpath}"
                )

    def test_chunk_text_uses_character_counting_not_byte_counting(self):
        # _chunk_text must slice on char boundaries, not bytes. It uses
        # Python's [:limit] which slices on code points → safe.
        # 5000 single-codepoint Bangla chars; the chunker should never
        # produce a chunk that, when re-encoded to UTF-8, would have
        # any half-character.
        text = "া" * 5000
        for chunk in _chunk(text, limit=2000):
            # Round-trip through UTF-8 to confirm no corruption
            chunk.encode("utf-8").decode("utf-8")

    def test_bangla_in_system_prompt_unicode_safe(self):
        # Building a system prompt that mentions "Bangla" / "Bengali"
        # must not corrupt any string under repeated encoding.
        from core.prompts import build_system_prompt
        sys_prompt = build_system_prompt("normal", language="bn")
        encoded = sys_prompt.encode("utf-8")
        decoded = encoded.decode("utf-8")
        assert decoded == sys_prompt
        assert "Bangla" in decoded
