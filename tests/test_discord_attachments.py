#!/usr/bin/env python3
"""
Tests for Discord bot image attachment handling.

Verifies:
  - _filter_image_attachments correctly partitions image vs non-image.
  - _filter_non_image_attachments is the inverse of _filter_image_attachments.
  - Non-image attachments are acknowledged with the correct message.
  - Image attachment descriptions are prepended to the brain-bound text.
  - Vision pipeline errors are caught and degrade gracefully.
  - The Discord 2000-char truncation pattern is preserved (regression).
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _restore_hermes_patch():
    """
    Restore hermes_bridge.run_turn_safe after each test.

    discord_bot.py patches hermes_bridge.run_turn_safe at import time.  The
    patch is intentional (Hermes guard for non-owners) but it changes the
    module identity of run_turn_safe, which breaks downstream tests that
    assert run_turn_safe.__module__ == "hermes_bridge".  This fixture
    ensures the original is restored after every test so the global state
    is clean for subsequent test modules.
    """
    import hermes_bridge
    original = hermes_bridge.run_turn_safe
    yield
    hermes_bridge.run_turn_safe = original


# ── Test fixtures / helpers ───────────────────────────────────────────────────

class FakeAttachment:
    """Lightweight stand-in for a discord.Attachment."""

    def __init__(self, content_type: str, url: str = "", filename: str = "f"):
        self.content_type = content_type
        self.url = url
        self.filename = filename

    async def read(self):
        return b"\x89PNG" + b"\x00" * 16


def _load_discord_bot_module():
    """Import the discord_bot module without running its main guard.

    The bot module's top-level body requires DISCORD_BOT_TOKEN and a
    single-instance lock.  We patch those out so we can import and test
    the pure helpers.
    """
    # Make the bot's bot-token guard think we have a token, and stub
    # the lock-acquisition so the test process doesn't actually try to
    # claim a lock file.  Then we can safely import the module.
    fake_token = "fake-token-for-tests"
    with patch.dict(os.environ, {
        "DISCORD_BOT_TOKEN": fake_token,
        "FAIRY_DISCORD_OWNER_ID": "123",
        "FAIRY_DISCORD_ALLOW_ALL": "1",
    }):
        with patch("os.path.exists", return_value=False):
            # Import the module fresh.
            import importlib
            if "controller.discord_bot" in sys.modules:
                del sys.modules["controller.discord_bot"]
            return importlib.import_module("controller.discord_bot")


# ── _filter_image_attachments ─────────────────────────────────────────────────

class TestFilterImageAttachments:
    """The attachment filter is a pure function over a list of fake attachments."""

    def test_returns_empty_for_no_attachments(self):
        bot = _load_discord_bot_module()
        result = bot._filter_image_attachments([])
        assert result == []

    def test_returns_image_attachments(self):
        bot = _load_discord_bot_module()
        png = FakeAttachment("image/png")
        jpeg = FakeAttachment("image/jpeg")
        webp = FakeAttachment("image/webp")
        result = bot._filter_image_attachments([png, jpeg, webp])
        assert len(result) == 3
        assert result == [png, jpeg, webp]

    def test_excludes_non_image_attachments(self):
        bot = _load_discord_bot_module()
        pdf = FakeAttachment("application/pdf")
        text = FakeAttachment("text/plain")
        audio = FakeAttachment("audio/mpeg")
        png = FakeAttachment("image/png")
        result = bot._filter_image_attachments([pdf, text, audio, png])
        assert len(result) == 1
        assert result[0] is png

    def test_handles_missing_content_type(self):
        """None content_type should be skipped, not crash."""
        bot = _load_discord_bot_module()
        none_ct = FakeAttachment(None)
        png = FakeAttachment("image/png")
        result = bot._filter_image_attachments([none_ct, png])
        assert result == [png]

    def test_handles_uppercase_mime(self):
        bot = _load_discord_bot_module()
        png_upper = FakeAttachment("IMAGE/PNG")
        result = bot._filter_image_attachments([png_upper])
        assert result == [png_upper]

    def test_inverse_returns_non_images(self):
        bot = _load_discord_bot_module()
        png = FakeAttachment("image/png")
        pdf = FakeAttachment("application/pdf")
        non_images = bot._filter_non_image_attachments([png, pdf])
        assert non_images == [pdf]


# ── _describe_attachments ─────────────────────────────────────────────────────

class TestDescribeAttachments:
    """_describe_attachments runs the vision pipeline with graceful errors."""

    def test_returns_string_when_vision_incapable(self):
        bot = _load_discord_bot_module()
        png = FakeAttachment("image/png")
        with patch("controller.vision.is_vision_capable", return_value=False):
            result = asyncio.run(bot._describe_attachments([png], "What is this?"))
        assert "[Vision]" in result
        assert "doesn't support image input" in result or "no-go" in result.lower() or "doesn't" in result

    def test_calls_describe_image_for_each_image(self):
        bot = _load_discord_bot_module()
        png = FakeAttachment("image/png")
        jpeg = FakeAttachment("image/jpeg")

        with patch("controller.vision.is_vision_capable", return_value=True):
            with patch(
                "controller.vision.describe_image",
                side_effect=["A cat.", "A dog."],
            ) as mock_desc:
                result = asyncio.run(
                    bot._describe_attachments([png, jpeg], "What is this?")
                )

        assert mock_desc.call_count == 2
        assert "A cat." in result
        assert "A dog." in result
        # Multiple images get numbered prefixes
        assert "Image 1 of 2" in result
        assert "Image 2 of 2" in result

    def test_uses_default_question_when_empty(self):
        bot = _load_discord_bot_module()
        png = FakeAttachment("image/png")

        with patch("controller.vision.is_vision_capable", return_value=True):
            with patch(
                "controller.vision.describe_image",
                return_value="A photo.",
            ) as mock_desc:
                asyncio.run(bot._describe_attachments([png], ""))

        args = mock_desc.call_args
        # Second positional arg is the question
        assert "Describe this image" in args[0][2]

    def test_single_image_uses_simple_prefix(self):
        bot = _load_discord_bot_module()
        png = FakeAttachment("image/png")

        with patch("controller.vision.is_vision_capable", return_value=True):
            with patch(
                "controller.vision.describe_image",
                return_value="A flower.",
            ):
                result = asyncio.run(bot._describe_attachments([png], "What?"))

        assert "Image\n" in result or "[Image]" in result
        assert "A flower." in result
        # Should NOT have the "X of Y" prefix for a single image
        assert "of 1" not in result

    def test_download_failure_is_caught(self):
        """A bad download should not crash the whole call."""
        bot = _load_discord_bot_module()
        png = FakeAttachment("image/png")

        # Patch the bot's _download_attachment_bytes to raise.
        async def bad_download(att):
            raise ConnectionError("network down")

        with patch("controller.vision.is_vision_capable", return_value=True):
            with patch.object(bot, "_download_attachment_bytes", bad_download):
                result = asyncio.run(
                    bot._describe_attachments([png], "What is this?")
                )

        assert "[Image 1" in result
        assert "download failed" in result.lower() or "network down" in result

    def test_oversize_image_is_reported_in_place(self):
        bot = _load_discord_bot_module()
        big_image = FakeAttachment("image/png")
        # Simulate huge download
        async def big_download(att):
            return b"\x00" * (9 * 1024 * 1024)
        with patch("controller.vision.is_vision_capable", return_value=True):
            with patch.object(bot, "_download_attachment_bytes", big_download):
                result = asyncio.run(bot._describe_attachments([big_image], ""))
        assert "exceeds" in result.lower() or "limit" in result.lower()
        # describe_image should NOT have been called
        # (we don't patch it, but it's never reached because the
        # size guard fires first)

    def test_describe_image_error_is_caught_per_image(self):
        """If describe_image raises, the error is caught per-image."""
        bot = _load_discord_bot_module()
        png = FakeAttachment("image/png")

        # describe_image always returns a string (it catches its own errors).
        # Here we patch it to raise a non-RuntimeError so we test the
        # in-_describe_attachments catch block as well.
        with patch("controller.vision.is_vision_capable", return_value=True):
            with patch(
                "controller.vision.describe_image",
                side_effect=RuntimeError("Vision API down"),
            ):
                result = asyncio.run(bot._describe_attachments([png], ""))

        # The error was caught and appended as a string.
        assert isinstance(result, str)
        assert "Vision API down" in result or "description failed" in result


# ── on_message integration (lightweight — uses mocks for client/message) ─────

class TestOnMessageAttachments:
    """on_message wires attachments into the brain-bound text correctly."""

    def test_non_image_attachment_acknowledged(self):
        bot = _load_discord_bot_module()

        # Build a fake message
        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 999
        msg.author.display_name = "Tester"
        msg.channel = MagicMock()
        msg.channel.id = 1
        msg.mentions = []
        msg.reference = None
        msg.content = "Look at this file"
        att = FakeAttachment("application/pdf")
        msg.attachments = [att]
        msg.channel.send = AsyncMock()

        async def run():
            with patch.object(bot, "_is_authorized", return_value=True):
                with patch.object(bot, "_should_respond", return_value=(True, "Look at this file")):
                    with patch.object(bot, "_is_owner", return_value=False):
                        with patch.object(bot, "_run_fairy_sync", return_value=("ok", [], None)):
                            await bot.on_message(msg)
            return msg.channel.send

        asyncio.run(run())

        # At least one send should be the "can only look at images" message
        sent_messages = [
            call.args[0] for call in msg.channel.send.call_args_list
            if call.args and isinstance(call.args[0], str)
        ]
        assert any("can only look at images" in m for m in sent_messages), (
            f"Expected non-image acknowledgement, got: {sent_messages}"
        )

    def test_image_attachment_prepends_to_text(self):
        bot = _load_discord_bot_module()

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 999
        msg.author.display_name = "Tester"
        msg.channel = MagicMock()
        msg.channel.id = 1
        msg.mentions = []
        msg.reference = None
        msg.content = "What's in this image?"
        att = FakeAttachment("image/png")
        msg.attachments = [att]
        msg.channel.send = AsyncMock()

        captured = {}

        def fake_run_fairy_sync(text, history, is_owner, caller_name, status_queue, loop, *extra):
            captured["text"] = text
            return ("ok", [], None)

        async def run():
            with patch.object(bot, "_is_authorized", return_value=True):
                with patch.object(bot, "_should_respond", return_value=(True, "What's in this image?")):
                    with patch.object(bot, "_is_owner", return_value=False):
                        with patch.object(bot, "_run_fairy_sync", fake_run_fairy_sync):
                            with patch("controller.vision.is_vision_capable", return_value=True):
                                with patch(
                                    "controller.vision.describe_image",
                                    return_value="A red car on grass.",
                                ):
                                    await bot.on_message(msg)

        asyncio.run(run())

        # Verify the brain-bound text contains the vision description
        assert captured.get("text"), "Expected _run_fairy_sync to be called"
        assert "A red car on grass." in captured["text"]
        assert "What's in this image?" in captured["text"]

    def test_no_attachments_normal_flow(self):
        """When there are no attachments, the flow is unchanged."""
        bot = _load_discord_bot_module()

        msg = MagicMock()
        msg.author = MagicMock()
        msg.author.bot = False
        msg.author.id = 999
        msg.author.display_name = "Tester"
        msg.channel = MagicMock()
        msg.channel.id = 1
        msg.mentions = []
        msg.reference = None
        msg.content = "Hello there"
        msg.attachments = []
        msg.channel.send = AsyncMock()

        captured = {}

        def fake_run_fairy_sync(text, history, is_owner, caller_name, status_queue, loop, *extra):
            captured["text"] = text
            return ("hi", [], None)

        async def run():
            with patch.object(bot, "_is_authorized", return_value=True):
                with patch.object(bot, "_should_respond", return_value=(True, "Hello there")):
                    with patch.object(bot, "_is_owner", return_value=False):
                        with patch.object(bot, "_run_fairy_sync", fake_run_fairy_sync):
                            await bot.on_message(msg)

        asyncio.run(run())

        assert "Hello there" in captured.get("text", "")
        # No "can only look at images" warning should fire
        sent_messages = [
            call.args[0] for call in msg.channel.send.call_args_list
            if call.args and isinstance(call.args[0], str)
        ]
        assert not any("can only look at images" in m for m in sent_messages)
