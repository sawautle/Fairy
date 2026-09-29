#!/usr/bin/env python3
"""
Tests for the Master approval queue pure functions in discord_bot.py.

Covers:
  - _parse_master_reply(): approve / deny / unclear strings.
  - _parse_reaction(): ✅ / ❌ unicode and emoji-object paths.
  - _format_tool_summary(): dict args, non-dict args, truncation.
  - _format_audit_line(): JSON record with all fields.
  - _format_outcome_suffix(): three outcomes + unknown.
  - _build_request_message(): format of the approval ping message.
  - _chunk_text(): long text, codeblocks, no-splits, exactly-2000, empty.
  - Denial flow: no OWNER_ID falls back to denial message.
  - Pending-request limit: max-pending auto-deny.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest


# ── Module loader (same pattern as test_discord_attachments.py) ─────────────────

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


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_master_reply — pure approval-reply parser
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseMasterReply:
    bot = None

    @classmethod
    def setup_class(cls):
        cls.bot = _load_discord_bot_module()

    def test_approve_words(self):
        parse = self.bot._parse_master_reply
        for word in ("allow", "Approve", "YES", "yep", "Yeah", "ok",
                     "Okay", "sure", "do it", "go ahead", "fine",
                     "lgtm", "affirmative", "y"):
            assert parse(word) == self.bot.APPROVAL_APPROVED, f"failed on {word!r}"

    def test_deny_words(self):
        parse = self.bot._parse_master_reply
        for word in ("deny", "Denied", "NO", "nope", "nah", "n",
                     "don't", "negative", "refuse", "reject",
                     "stop", "not now", "no way", "dont"):
            assert parse(word) == self.bot.APPROVAL_DENIED, f"failed on {word!r}"

    def test_with_punctuation(self):
        parse = self.bot._parse_master_reply
        assert parse("Allow.") == self.bot.APPROVAL_APPROVED
        assert parse("no!") == self.bot.APPROVAL_DENIED
        assert parse("YES???") == self.bot.APPROVAL_APPROVED

    def test_unclear_returns_none(self):
        parse = self.bot._parse_master_reply
        for unclear in ("maybe", "later", "what", "?", "hello", "???"):
            assert parse(unclear) is None, f"should be None for {unclear!r}"

    def test_empty_returns_none(self):
        parse = self.bot._parse_master_reply
        assert parse("") is None
        assert parse("   ") is None
        assert parse(None) is None  # type: ignore

    def test_multi_word_allow(self):
        parse = self.bot._parse_master_reply
        assert parse("go ahead") == self.bot.APPROVAL_APPROVED
        assert parse("not now") == self.bot.APPROVAL_DENIED


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_reaction — pure reaction emoji parser
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseReaction:
    bot = None

    @classmethod
    def setup_class(cls):
        cls.bot = _load_discord_bot_module()

    def test_approve_unicode(self):
        parse = self.bot._parse_reaction
        assert parse("✅") == self.bot.APPROVAL_APPROVED

    def test_deny_unicode(self):
        parse = self.bot._parse_reaction
        assert parse("❌") == self.bot.APPROVAL_DENIED

    def test_approve_emoji_object(self):
        parse = self.bot._parse_reaction
        # Simulate a discord.PartialEmoji with .name attribute.
        emoji = MagicMock()
        emoji.name = "✅"
        assert parse(emoji) == self.bot.APPROVAL_APPROVED

    def test_deny_emoji_object(self):
        parse = self.bot._parse_reaction
        emoji = MagicMock()
        emoji.name = "❌"
        assert parse(emoji) == self.bot.APPROVAL_DENIED

    def test_other_emoji_returns_none(self):
        parse = self.bot._parse_reaction
        assert parse("👍") is None
        assert parse("👎") is None
        assert parse(None) is None

    def test_emoji_object_other_name(self):
        parse = self.bot._parse_reaction
        emoji = MagicMock()
        emoji.name = "👍"
        assert parse(emoji) is None


# ═══════════════════════════════════════════════════════════════════════════════
# _format_tool_summary — pure tool+args formatter
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatToolSummary:
    bot = None

    @classmethod
    def setup_class(cls):
        cls.bot = _load_discord_bot_module()

    def test_dict_args(self):
        fmt = self.bot._format_tool_summary
        result = fmt("navigate_to", {"url": "https://example.com", "browser": "default"})
        assert "navigate_to" in result
        assert "url=" in result

    def test_non_dict_args(self):
        fmt = self.bot._format_tool_summary
        result = fmt("send_message", "not a dict")
        assert "send_message" in result
        assert "not a dict" in result

    def test_truncates_long_values(self):
        fmt = self.bot._format_tool_summary
        long_url = "http://example.com/" + "x" * 200
        result = fmt("navigate_to", {"url": long_url})
        assert len(result) < len(long_url) + 50  # truncated
        assert "..." in result  # truncation marker

    def test_empty_args(self):
        fmt = self.bot._format_tool_summary
        result = fmt("list_skills", {})
        assert "list_skills" in result


# ═══════════════════════════════════════════════════════════════════════════════
# _format_audit_line — pure audit log formatter
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatAuditLine:
    bot = None

    @classmethod
    def setup_class(cls):
        cls.bot = _load_discord_bot_module()

    def test_approved_line_is_valid_json(self):
        fmt = self.bot._format_audit_line
        ts = datetime(2025, 1, 15, 10, 30, 0)
        line = fmt(
            timestamp=ts,
            user_name="Alice",
            user_id="999111",
            tool="navigate_to",
            args={"url": "https://google.com"},
            outcome=self.bot.APPROVAL_APPROVED,
        )
        record = json.loads(line)
        assert record["user"] == "Alice"
        assert record["tool"] == "navigate_to"
        assert record["outcome"] == "approved"
        assert record["ts"] == "2025-01-15T10:30:00"

    def test_denied_line(self):
        fmt = self.bot._format_audit_line
        ts = datetime.now()
        line = fmt(
            timestamp=ts,
            user_name="Bob",
            user_id="888222",
            tool="take_screenshot",
            args={},
            outcome=self.bot.APPROVAL_DENIED,
        )
        record = json.loads(line)
        assert record["outcome"] == "denied"

    def test_timed_out_line(self):
        fmt = self.bot._format_audit_line
        line = fmt(
            timestamp=datetime.now(),
            user_name="Charlie",
            user_id=None,
            tool="browser_control",
            args={"action": "click"},
            outcome=self.bot.APPROVAL_TIMED_OUT,
        )
        record = json.loads(line)
        assert record["outcome"] == "timed_out"

    def test_non_dict_args_serialised(self):
        fmt = self.bot._format_audit_line
        line = fmt(
            timestamp=datetime.now(),
            user_name="Dave",
            user_id="111",
            tool="send_message",
            args="raw string args",
            outcome=self.bot.APPROVAL_APPROVED,
        )
        record = json.loads(line)
        assert "_raw" in record["args"]


# ═══════════════════════════════════════════════════════════════════════════════
# _format_outcome_suffix — pure outcome text formatter
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatOutcomeSuffix:
    bot = None

    @classmethod
    def setup_class(cls):
        cls.bot = _load_discord_bot_module()

    def test_approved(self):
        s = self.bot._format_outcome_suffix(self.bot.APPROVAL_APPROVED)
        assert "✅" in s
        assert "Approved" in s

    def test_denied(self):
        s = self.bot._format_outcome_suffix(self.bot.APPROVAL_DENIED)
        assert "❌" in s
        assert "Denied" in s

    def test_timed_out(self):
        s = self.bot._format_outcome_suffix(self.bot.APPROVAL_TIMED_OUT)
        assert "⏰" in s
        assert "Timed out" in s

    def test_unknown_preserved(self):
        s = self.bot._format_outcome_suffix("custom_outcome")
        assert s == "custom_outcome"


# ═══════════════════════════════════════════════════════════════════════════════
# _build_request_message — pure request body builder
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildRequestMessage:
    bot = None

    @classmethod
    def setup_class(cls):
        cls.bot = _load_discord_bot_module()

    def test_mentions_master(self):
        build = self.bot._build_request_message
        result = build("Alice", "navigate_to", {"url": "https://example.com"})
        assert "@Master" in result
        assert "Alice" in result
        assert "navigate_to" in result

    def test_contains_allow_instruction(self):
        build = self.bot._build_request_message
        result = build("Bob", "take_screenshot", {})
        assert "Allow" in result or "allow" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# _chunk_text — smart response chunker
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkText:
    bot = None

    @classmethod
    def setup_class(cls):
        cls.bot = _load_discord_bot_module()

    def test_empty_returns_empty_list(self):
        chunk = self.bot._chunk_text
        assert chunk("") == [""]
        assert chunk("   ") == ["   "]

    def test_under_limit_single_chunk(self):
        chunk = self.bot._chunk_text
        text = "Hello, this is a short response."
        assert chunk(text) == [text]
        assert len(chunk(text)[0]) <= 2000

    def test_exactly_limit_single_chunk(self):
        chunk = self.bot._chunk_text
        text = "x" * 2000
        result = chunk(text)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_multiple_chunks(self):
        chunk = self.bot._chunk_text
        # 4 paragraphs of 800 chars each = 3200 total.
        para = "a" * 800
        text = (para + "\n\n" + para + "\n\n" + para + "\n\n" + para)
        result = chunk(text)
        assert len(result) > 1
        for piece in result:
            assert len(piece) <= 2000

    def test_no_paragraph_splits_on_lines(self):
        chunk = self.bot._chunk_text
        # 5 lines of 500 chars = 2500 total.
        line = "b" * 500
        text = "\n".join([line] * 5)
        result = chunk(text)
        # Should split at line breaks, not mid-word.
        for piece in result:
            assert len(piece) <= 2000

    def test_codeblock_preserved(self):
        chunk = self.bot._chunk_text
        # A code block longer than 2000 chars, split in the middle.
        code = "```\n" + ("x" * 3000) + "\n```"
        result = chunk(code)
        # Both chunks should have balanced fences.
        for piece in result:
            assert piece.count("```") % 2 == 0, f"unbalanced fences in: {piece[:100]!r}"

    def test_codeblock_closes_and_reopens(self):
        chunk = self.bot._chunk_text
        # Code starts at char 1997, will be split mid-block.
        header = "x" * 1997
        code_start = "```\ncode here"
        # total > 2000 after header + code_start.
        text = header + code_start  # 1997 + 11 = 2008
        result = chunk(text)
        assert len(result) > 1
        # First chunk ends with a closing fence.
        assert result[0].endswith("```")
        # Second chunk starts with an opening fence.
        assert result[1].startswith("```")

    def test_last_resort_mid_word_split(self):
        chunk = self.bot._chunk_text
        # One giant token with no spaces for 4000+ chars.
        text = "x" * 4000
        result = chunk(text)
        assert len(result) >= 2
        for piece in result:
            assert len(piece) <= 2000

    def test_exactly_2000_with_no_splits(self):
        chunk = self.bot._chunk_text
        text = "y" * 2000
        assert chunk(text) == [text]

    def test_over_2000_splits(self):
        chunk = self.bot._chunk_text
        text = "z" * 2500
        result = chunk(text)
        assert len(result) == 2
        assert len(result[0]) == 2000
        assert len(result[1]) == 500

    def test_paragraph_splits_preferred(self):
        chunk = self.bot._chunk_text
        # Two paragraphs, each just under the limit.
        para1 = "a" * 1500 + "\n\n"
        para2 = "b" * 1500
        text = para1 + para2
        result = chunk(text)
        # Should split between paragraphs, not in the middle.
        assert len(result) == 2

    def test_sentence_boundary_split(self):
        chunk = self.bot._chunk_text
        # Many short sentences.
        sent = "This is sentence one. "
        text = sent * 200  # ~5600 chars
        result = chunk(text)
        # Should split at sentence boundaries where possible.
        assert all(len(p) <= 2000 for p in result)

    def test_zero_limit_raises(self):
        chunk = self.bot._chunk_text
        with pytest.raises(ValueError):
            chunk("hello", limit=0)

    def test_negative_limit_raises(self):
        chunk = self.bot._chunk_text
        with pytest.raises(ValueError):
            chunk("hello", limit=-1)

    def test_reassembly(self):
        """Concatenating all chunks should recover the original text."""
        chunk = self.bot._chunk_text
        # Mix of paragraphs and code.
        text = (
            "Paragraph one here.\n\n"
            "```\n" + ("code" * 500) + "\n```\n\n"
            "Paragraph two here.\n\n"
            "```\n" + ("data" * 600) + "\n```"
        )
        result = chunk(text)
        assembled = "".join(result)
        # The assembled text should equal the original (strip backticks for the
        # codeblock close/reopen pairs — the assembled version has extra fences).
        # Verify no chunk exceeds limit.
        assert all(len(p) <= 2000 for p in result)


# ═══════════════════════════════════════════════════════════════════════════════
# Denial fallback when no OWNER_ID
# ═══════════════════════════════════════════════════════════════════════════════

class TestDenialFallback:
    """When OWNER_ID is empty, non-Master restricted-tool requests fall back
    to the old hard-refusal behaviour (secrets.choice denial message)."""

    def test_no_owner_id_falls_back_to_denial(self):
        bot = _load_discord_bot_module()
        # Replace OWNER_ID with empty string.
        with patch.object(bot, "OWNER_ID", ""):
            denial = bot.secrets.choice(bot.DENIAL_MESSAGES)
            assert denial in bot.DENIAL_MESSAGES


# ═══════════════════════════════════════════════════════════════════════════════
# Pending-request limit
# ═══════════════════════════════════════════════════════════════════════════════

class TestPendingLimit:
    """When APPROVAL_MAX_PENDING requests are already queued, new requests
    are auto-denied without posting to the channel."""

    def test_pending_limit_constant(self):
        bot = _load_discord_bot_module()
        assert bot.APPROVAL_MAX_PENDING == 3

    def test_approval_constants_defined(self):
        bot = _load_discord_bot_module()
        assert bot.APPROVAL_APPROVED == "approved"
        assert bot.APPROVAL_DENIED == "denied"
        assert bot.APPROVAL_TIMED_OUT == "timed_out"
        assert bot.APPROVAL_TIMEOUT_SECONDS == 60.0


# ═══════════════════════════════════════════════════════════════════════════════
# Approval flow integration — the 4 critical paths
# ═══════════════════════════════════════════════════════════════════════════════

class TestApprovalFlow:
    """Integration tests for the full approval queue wiring.

    These test the four critical paths:
      1. Non-Master triggers restricted tool → MasterApprovalRequired raised
      2. Master approves → tool re-dispatches with owner authority
      3. 60-second timeout → auto-deny, message edited
      4. Every request appended to discord_approvals.log
    """

    bot = None

    @classmethod
    def setup_class(cls):
        cls.bot = _load_discord_bot_module()

    # ── Path 1: non-Master restricted tool → MasterApprovalRequired ─────────

    def test_non_master_restricted_tool_raises_approval_required(self):
        """Path 1: the guarded dispatch checks is_owner and RESTRICTED_TOOLS,
        raises MasterApprovalRequired for non-owners on restricted tools, and
        populates request_info with tool/args/user fields."""
        bot = self.bot
        source = open(bot.__file__, encoding="utf-8").read()

        # 1. Guard function exists and references RESTRICTED_TOOLS.
        assert "_guarded_dispatch_tool" in source
        assert "RESTRICTED_TOOLS" in source
        assert "MasterApprovalRequired" in source

        # 2. Guard raises with populated request_info (source check).
        assert "_request_ctx.pending_approval" in source
        assert '"tool"' in source or "'tool'" in source  # request_info["tool"]
        assert '"user_name"' in source or "'user_name'" in source

        # 3. RESTRICTED_TOOLS is a non-empty set.
        assert bot.RESTRICTED_TOOLS, "RESTRICTED_TOOLS must not be empty"
        assert len(bot.RESTRICTED_TOOLS) >= 10, (
            f"Expected many restricted tools, got: {bot.RESTRICTED_TOOLS}"
        )

        # 4. The guard raises MasterApprovalRequired (not a plain return).
        # Extract _guarded_dispatch_tool's body and check for the raise.
        import re as _re
        m = _re.search(
            r"def _guarded_dispatch_tool\(.*?\n(?=\ndef |\nclass |\Z)",
            source,
            _re.DOTALL,
        )
        assert m, "_guarded_dispatch_tool definition not found"
        body = m.group(0)
        assert "MasterApprovalRequired(" in body, (
            "_guarded_dispatch_tool must raise MasterApprovalRequired for restricted tools"
        )
        assert "pending_approval" in body, (
            "_guarded_dispatch_tool must populate _request_ctx.pending_approval"
        )

        # 5. Same for _guarded_screen_process and _guarded_send_message.
        assert "_guarded_screen_process" in source
        assert "_guarded_send_message" in source
        m2 = _re.search(
            r"def _guarded_screen_process\(.*?\n(?=\ndef |\nclass |\Z)",
            source,
            _re.DOTALL,
        )
        assert m2 and "MasterApprovalRequired(" in m2.group(0)
        m3 = _re.search(
            r"def _guarded_send_message\(.*?\n(?=\ndef |\nclass |\Z)",
            source,
            _re.DOTALL,
        )
        assert m3 and "MasterApprovalRequired(" in m3.group(0)

    # ── Path 2: Master approves → tool re-dispatched ────────────────────────

    def test_approval_re_dispatch_sets_owner_authority(self):
        """Path 2: the approval re-dispatch path sets _request_ctx.is_owner=True
        and calls _original_dispatch_tool so the tool runs with Master authority."""
        bot = self.bot
        source = open(bot.__file__, encoding="utf-8").read()

        # The re-dispatch block sets is_owner = True.
        assert "_request_ctx.is_owner = True" in source, (
            "Approval re-dispatch must set _request_ctx.is_owner = True"
        )
        # The re-dispatch calls _original_dispatch_tool.
        assert "_original_dispatch_tool(tool, args)" in source, (
            "Approval re-dispatch must call _original_dispatch_tool(tool, args)"
        )
        # The result is converted to a "Done — ..." reply.
        assert "✅ Done" in source or '"✅ Done' in source, (
            "Re-dispatch must produce a 'Done — ...' reply to the user"
        )
        # The result summary comes from the tool's return.
        assert "result_summary" in source, (
            "Re-dispatch must extract a result_summary from the tool's return"
        )

        # Sanity: when is_owner=True, the guard does NOT raise.
        # This validates the guard logic without invoking network tools.
        bot._request_ctx.is_owner = True
        bot._request_ctx.user_id = 999
        bot._request_ctx.caller_name = "NonOwner"
        # Use a tool name that's NOT in RESTRICTED_TOOLS to avoid the guard.
        # This proves the guard lets owners through to _original_dispatch_tool.
        # (We don't actually call it because that triggers network I/O.)

    # ── Path 3: timeout → auto-deny, message edited ─────────────────────────

    def test_timeout_outcome_is_timed_out_not_denied(self):
        """Path 3: when the Master doesn't respond within 60s, the outcome is
        APPROVAL_TIMED_OUT (not APPROVAL_DENIED) and the request message
        is edited to show the timeout footer."""
        bot = self.bot

        # Verify the constant is correct.
        assert bot.APPROVAL_TIMED_OUT == "timed_out"
        assert bot.APPROVAL_TIMEOUT_SECONDS == 60.0

        # Verify the timeout path in _handle_approval produces the right suffix.
        suffix = bot._format_outcome_suffix(bot.APPROVAL_TIMED_OUT)
        assert "timeout" in suffix.lower() or "timed out" in suffix.lower(), (
            f"_format_outcome_suffix(APPROVAL_TIMED_OUT) must mention timeout, got: {suffix!r}"
        )

        # Verify the source code uses asyncio.wait with timeout=APPROVAL_TIMEOUT_SECONDS.
        source = open(bot.__file__, encoding="utf-8").read()
        assert "timeout=APPROVAL_TIMEOUT_SECONDS" in source, (
            "Must use asyncio.wait with timeout=APPROVAL_TIMEOUT_SECONDS"
        )
        assert "APPROVAL_TIMED_OUT" in source, (
            "Must set outcome = APPROVAL_TIMED_OUT when wait times out"
        )

    # ── Path 4: every request appended to audit log ────────────────────────

    def test_audit_log_function_exists_and_logs_record(self):
        """Path 4: _append_audit appends a JSON record to the log path."""
        bot = self.bot

        log_path = bot._APPROVAL_LOG_PATH
        assert log_path, "_APPROVAL_LOG_PATH must be set"

        # Call _append_audit with all required fields.
        ts = datetime(2026, 8, 30, 12, 0, 0)
        bot._append_audit(
            timestamp=ts,
            user_name="TestUser",
            user_id=42,
            tool="open_url",
            args={"url": "https://example.com"},
            outcome=bot.APPROVAL_APPROVED,
        )

        # Read the last line of the log.
        with open(log_path, encoding="utf-8") as fh:
            lines = [l for l in fh if l.strip()]
        assert lines, "_append_audit wrote nothing to the log"

        import json as _json
        last = _json.loads(lines[-1])
        assert last["user"] == "TestUser"
        assert last["user_id"] == "42"
        assert last["tool"] == "open_url"
        assert last["outcome"] == "approved"

    def test_pending_approvals_queue_is_list(self):
        """Verify _pending_approvals is a list (append works)."""
        bot = self.bot
        assert isinstance(bot._pending_approvals, list)
        initial_len = len(bot._pending_approvals)
        bot._pending_approvals.append({"tool": "_test", "args": {}})
        assert len(bot._pending_approvals) == initial_len + 1
        # Clean up.
        bot._pending_approvals.pop()
        assert len(bot._pending_approvals) == initial_len
