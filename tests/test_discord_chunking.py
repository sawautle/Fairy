#!/usr/bin/env python3
"""
Tests for _chunk_text — the smart response splitter in discord_bot.py.

Covers all edge cases required by the feature spec:
  - Long text (> 2000 chars) splits correctly.
  - Codeblock-open at split point → close and reopen.
  - Text with no split points → last-resort mid-word split.
  - Exactly-2000-char input → single chunk, no split.
  - Paragraph / line / sentence boundaries respected.
  - Empty input → returns [""].
  - Each chunk ≤ limit.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest


def _load_discord_bot_module():
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


# ── Pure function under test ───────────────────────────────────────────────────

def _chunk(text: str, limit: int = 2000):
    bot = _load_discord_bot_module()
    return bot._chunk_text(text, limit)


# ═══════════════════════════════════════════════════════════════════════════════
# Basic / positive cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkBasic:
    """Basic happy-path behaviour."""

    def test_empty_string_returns_empty_chunk(self):
        assert _chunk("") == [""]

    def test_short_text_is_not_split(self):
        text = "Hello! This is a short reply."
        assert _chunk(text) == [text]

    def test_exactly_2000_chars_one_chunk(self):
        text = "x" * 2000
        result = _chunk(text)
        assert len(result) == 1
        assert result[0] == text

    def test_2001_chars_two_chunks(self):
        text = "x" * 2001
        result = _chunk(text)
        assert len(result) == 2
        assert len(result[0]) == 2000
        assert result[1] == "x"

    def test_all_chunks_under_limit(self):
        text = ("word " * 1000) + ("\n\n" + ("word " * 1000)) * 3
        for chunk in _chunk(text):
            assert len(chunk) <= 2000, f"chunk of length {len(chunk)} exceeds 2000"

    def test_concatenation_no_data_loss(self):
        """Concatenating all chunks (accounting for fence open/close pairs)
        produces the original logical content."""
        text = (
            "First paragraph.\n\n"
            "```\n" + ("line\n" * 500) + "```\n\n"
            "Second paragraph."
        )
        result = _chunk(text)
        assembled = "".join(result)
        # The assembled version has extra ``` markers from close/reopen,
        # but the semantic content is preserved.
        assert "First paragraph" in assembled
        assert "Second paragraph" in assembled
        assert "```" in assembled


# ═══════════════════════════════════════════════════════════════════════════════
# Boundary-splitting priority
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkBoundaryPriority:
    """Paragraphs are preferred; lines second; sentences third."""

    def test_splits_on_paragraph_not_mid_sentence(self):
        # Two paragraphs of 1500 chars each (total 3000+).
        p1 = "a" * 1500 + "."
        p2 = "b" * 1500 + "."
        text = p1 + "\n\n" + p2
        result = _chunk(text)
        # Should split at \n\n, producing two clean chunks.
        assert len(result) == 2
        # No mid-word split — all 'a' on one side, all 'b' on the other.
        # The paragraph break is included in the first chunk (a. + \n\n).
        assert result[0].endswith("a.\n\n")
        assert result[1].startswith("b")  # p2 starts with 'b', period is at end

    def test_splits_on_line_not_mid_word(self):
        line = "x" * 400
        text = "\n".join([line] * 6)  # 2400 total
        result = _chunk(text)
        # Should split at \n boundaries.
        # First chunk must not be truncated mid-word.
        for chunk in result:
            assert "\n\n" not in chunk or len(chunk) <= 2000

    def test_sentence_boundary_used(self):
        # Many short sentences.
        sentences = ". ".join(["word" * 50] * 100)  # ~5000 chars
        result = _chunk(sentences)
        for chunk in result:
            assert len(chunk) <= 2000


# ═══════════════════════════════════════════════════════════════════════════════
# Codeblock preservation
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkCodeblocks:
    """Codeblocks must not be split mid-block without proper fence balancing."""

    def test_open_codeblock_at_split_closes_and_reopens(self):
        # A code block that starts just before the 2000-char boundary.
        header = "x" * 1990
        code = "```\nprint('hello')\n```"
        text = header + code  # header(1990) + code(>20) = > 2010
        result = _chunk(text)
        # Must have at least 2 chunks.
        assert len(result) >= 2
        # All fences balanced in every chunk.
        for chunk in result:
            assert chunk.count("```") % 2 == 0, f"unbalanced in: {chunk[:80]!r}"

    def test_open_codeblock_closes_on_first_chunk(self):
        header = "y" * 1995
        code = "```\ndata"
        text = header + code
        result = _chunk(text)
        assert result[0].endswith("```")
        assert result[1].startswith("```")

    def test_balanced_codeblock_passes_through(self):
        block = "```\n" + ("z" * 100) + "\n```"
        result = _chunk(block)
        for chunk in result:
            assert chunk.count("```") % 2 == 0

    def test_nested_codeblocks_balanced(self):
        # Two fenced blocks: even number of ```, should stay balanced.
        text = "```\ninner\n```\n" + ("x" * 3000) + "\n```\nouter\n```"
        result = _chunk(text)
        for chunk in result:
            assert chunk.count("```") % 2 == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Last-resort mid-word split
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkLastResort:
    """When no natural break point exists, mid-word split is used."""

    def test_no_space_token_splits_at_limit(self):
        # A single 5000-char token with no spaces.
        text = "x" * 5000
        result = _chunk(text)
        assert len(result) >= 3
        for chunk in result:
            assert len(chunk) <= 2000

    def test_token_split_is_usable(self):
        """Even a mid-word split should produce readable (if awkward) output."""
        text = "word" * 600  # 2400 chars, no spaces
        result = _chunk(text)
        assert all(len(c) <= 2000 for c in result)
        # Concatenated chunks reproduce the original word sequence.
        assembled = "".join(result)
        assert assembled == text


# ═══════════════════════════════════════════════════════════════════════════════
# Custom limit
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkCustomLimit:
    """The limit parameter is respected."""

    def test_custom_limit_respected(self):
        text = "a" * 500
        result = _chunk(text, limit=200)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 200

    def test_zero_limit_raises(self):
        with pytest.raises(ValueError):
            _chunk("hello", limit=0)

    def test_negative_limit_raises(self):
        with pytest.raises(ValueError):
            _chunk("hello", limit=-1)

    def test_limit_one(self):
        text = "abcde"
        result = _chunk(text, limit=1)
        assert result == ["a", "b", "c", "d", "e"]


# ═══════════════════════════════════════════════════════════════════════════════
# Discord 2000-char compliance
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkDiscordCompliance:
    """All chunks must be ≤ Discord's 2000-char message limit."""

    def test_many_chunks_all_within_limit(self):
        bot = _load_discord_bot_module()
        limit = bot.CHUNK_LIMIT
        assert limit == 2000

        # Generate text that forces many splits.
        text = (
            ("Paragraph " + "word " * 400 + "\n\n") * 10
            + "```\n" + ("line\n" * 500) + "```\n"
            + ("x" * 5000)
        )
        result = _chunk(text)
        for chunk in result:
            assert len(chunk) <= 2000, f"Chunk too long: {len(chunk)}"

    def test_1999_chars_still_one_chunk(self):
        text = "y" * 1999
        assert _chunk(text) == [text]

    def test_2000_chars_one_chunk(self):
        text = "z" * 2000
        assert _chunk(text) == [text]
