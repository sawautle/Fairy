#!/usr/bin/env python3
"""
Tests for controller/vision.py — shared image intake pipeline.

Covers:
  - describe_image: success path (injected transport), error paths, size guard.
  - is_vision_capable: cached check, text-only model denial, vision model allow.
  - validate_image_size: pure-function boundary testing.
  - graceful degradation: never raises into callers, always returns a string.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_vision_cache():
    """Reset the vision-capable cache before each test."""
    import controller.vision as vision
    with vision._vision_cache_lock:
        vision._vision_capable_cache["ok"] = None
        vision._vision_capable_cache["ts"] = 0.0
    yield
    with vision._vision_cache_lock:
        vision._vision_capable_cache["ok"] = None
        vision._vision_capable_cache["ts"] = 0.0


# ── validate_image_size (pure function) ────────────────────────────────────────

class TestValidateImageSize:
    """validate_image_size is a pure function — no I/O, no network."""

    def test_ok_under_limit(self):
        from controller.vision import validate_image_size
        # Exactly 8 MB
        data = b"\x89PNG" + b"\x00" * (8 * 1024 * 1024 - 4)
        ok, reason = validate_image_size(data)
        assert ok is True
        assert reason == ""

    def test_ok_well_under_limit(self):
        from controller.vision import validate_image_size
        data = b"\x89PNG" + b"\x00" * 1024
        ok, reason = validate_image_size(data)
        assert ok is True
        assert reason == ""

    def test_over_limit_rejected(self):
        from controller.vision import validate_image_size
        # 8 MB + 1 byte
        data = b"\x89PNG" + b"\x00" * (8 * 1024 * 1024 + 1)
        ok, reason = validate_image_size(data)
        assert ok is False
        assert "exceeds" in reason.lower()
        assert "limit" in reason.lower()

    def test_custom_max_bytes_rejected(self):
        """Custom max_bytes limit is enforced."""
        from controller.vision import validate_image_size
        # 201 bytes — limit is 100
        data = b"\x00" * 201
        ok, reason = validate_image_size(data, max_bytes=100)
        assert ok is False
        assert "100" in reason

    def test_custom_max_bytes_accepted(self):
        """Custom max_bytes within limit is accepted."""
        from controller.vision import validate_image_size
        data = b"\x00" * 50
        ok, reason = validate_image_size(data, max_bytes=100)
        assert ok is True
        assert reason == ""

    def test_empty_bytes_accepted(self):
        from controller.vision import validate_image_size
        ok, reason = validate_image_size(b"")
        assert ok is True
        assert reason == ""


# ── is_vision_capable ─────────────────────────────────────────────────────────

class TestIsVisionCapable:
    """
    is_vision_capable() checks the configured primary model against
    known text-only and vision-capable lists.  Tests control the
    configured model by patching _get_primary_model().
    """

    def test_false_when_not_configured(self):
        """No API key → vision is not available."""
        from controller.vision import is_vision_capable, _OPENROUTER_CONFIGURED
        if _OPENROUTER_CONFIGURED:
            pytest.skip("OpenRouter is configured — skip config-dependent test")

        result = is_vision_capable()
        assert result is False

    def test_false_for_gemma4_text_model(self, reset_vision_cache):
        """gemma4 is on the text-only list."""
        from controller.vision import is_vision_capable

        with patch("controller.vision._get_primary_model", return_value="gemma4"):
            result = is_vision_capable()

        assert result is False

    def test_false_for_qwen_coder_model(self, reset_vision_cache):
        """qwen2.5-coder:7b is on the text-only list."""
        from controller.vision import is_vision_capable

        with patch("controller.vision._get_primary_model", return_value="qwen2.5-coder:7b"):
            result = is_vision_capable()

        assert result is False

    def test_false_for_llama_model(self, reset_vision_cache):
        """llama4 is on the text-only list."""
        from controller.vision import is_vision_capable

        with patch("controller.vision._get_primary_model", return_value="llama4"):
            result = is_vision_capable()

        assert result is False

    def test_true_for_claude_sonnet(self, reset_vision_cache):
        """claude-3.5-sonnet is on the vision-capable list."""
        from controller.vision import is_vision_capable

        with patch("controller.vision._get_primary_model",
                   return_value="anthropic/claude-3.5-sonnet"):
            result = is_vision_capable()

        assert result is True

    def test_true_for_gemini_flash(self, reset_vision_cache):
        """gemini-2.0-flash is on the vision-capable list."""
        from controller.vision import is_vision_capable

        with patch("controller.vision._get_primary_model",
                   return_value="google/gemini-2.0-flash"):
            result = is_vision_capable()

        assert result is True

    def test_true_for_chatgpt_4o(self, reset_vision_cache):
        """chatgpt-4o-latest is on the vision-capable list."""
        from controller.vision import is_vision_capable

        with patch("controller.vision._get_primary_model",
                   return_value="openai/chatgpt-4o-latest"):
            result = is_vision_capable()

        assert result is True

    def test_true_for_gemini_3_6_flash(self, reset_vision_cache):
        """gemini-3.6-flash is on the vision-capable list."""
        from controller.vision import is_vision_capable

        with patch("controller.vision._get_primary_model",
                   return_value="google/gemini-3.6-flash"):
            result = is_vision_capable()

        assert result is True

    def test_caches_result_on_second_call(self, reset_vision_cache):
        """Second call within TTL should NOT re-run the check."""
        from controller.vision import is_vision_capable

        call_count = 0

        def counting_check():
            nonlocal call_count
            call_count += 1
            return True

        with patch("controller.vision._is_vision_capable_sync", counting_check):
            is_vision_capable()
            is_vision_capable()
            is_vision_capable()

        assert call_count == 1, "Result should be cached after first call"

    def test_cache_ttl_expires_triggers_recheck(self, reset_vision_cache):
        """After TTL expires, a fresh check is made."""
        from controller.vision import is_vision_capable, _VISION_CACHE_TTL

        call_count = 0

        def counting_check():
            nonlocal call_count
            call_count += 1
            return True

        with patch("controller.vision._is_vision_capable_sync", counting_check):
            is_vision_capable()
            # Expire the cache
            import controller.vision as vision
            with vision._vision_cache_lock:
                vision._vision_capable_cache["ts"] = 0.0
            is_vision_capable()

        assert call_count == 2

    def test_network_failure_returns_false_conservatively(self, reset_vision_cache):
        """Network error during /models lookup is caught; returns False."""
        from controller.vision import is_vision_capable

        def raising_check():
            raise OSError("DNS failure")

        with patch("controller.vision._is_vision_capable_sync", raising_check):
            result = is_vision_capable()

        assert result is False


# ── describe_image ─────────────────────────────────────────────────────────────

class TestDescribeImage:
    """describe_image should never raise — always return a string."""

    def test_size_guard_rejects_large_images(self):
        from controller.vision import describe_image

        large = b"\x89PNG" + b"\x00" * (9 * 1024 * 1024)
        result = describe_image(large, "image/png", "What is this?")
        assert "[Vision]" in result
        assert "exceeds" in result.lower()

    def test_uses_injected_transport(self):
        """Transport is called with the correct vision message shape."""
        from controller.vision import describe_image

        fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        captured_messages = []
        captured_models = []

        def fake_transport(messages, model="openrouter/free", timeout=60.0):
            captured_messages.append(messages)
            captured_models.append(model)
            return "It is a red circle on a white background."

        with patch("controller.vision._openrouter_vision", fake_transport):
            with patch("controller.vision._get_primary_model",
                       return_value="anthropic/claude-3.5-sonnet"):
                result = describe_image(
                    fake_image,
                    "image/png",
                    "Describe this image.",
                )

        assert result == "It is a red circle on a white background."
        assert len(captured_messages) == 1

        msg = captured_messages[0][0]
        assert msg["role"] == "user"
        content = msg["content"]
        assert isinstance(content, list)

        text_part = next(c for c in content if c["type"] == "text")
        assert "Describe this image." in text_part["text"]

        image_part = next(c for c in content if c["type"] == "image_url")
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

        # Verify base64 decodes to the original image bytes
        b64_data = image_part["image_url"]["url"].split(",", 1)[1]
        decoded = base64.b64decode(b64_data)
        assert decoded == fake_image

    def test_default_question_when_empty(self):
        """Empty question gets a sensible default."""
        from controller.vision import describe_image

        fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        captured_messages = []

        def fake_transport(messages, model="openrouter/free", timeout=60.0):
            captured_messages.append(messages)
            return "Description."

        with patch("controller.vision._openrouter_vision", fake_transport):
            with patch("controller.vision._get_primary_model",
                       return_value="google/gemini-3.6-flash"):
                describe_image(fake_image, "image/png", "")

        msg = captured_messages[0][0]
        text_part = next(c for c in msg["content"] if c["type"] == "text")
        assert "Describe this image" in text_part["text"]
        assert "answer any question" in text_part["text"]

    def test_transport_error_returns_error_string(self):
        """Transport errors surface as user-facing strings, not exceptions."""
        from controller.vision import describe_image

        fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        def fake_transport(messages, model="openrouter/free", timeout=60.0):
            raise RuntimeError("Connection refused")

        with patch("controller.vision._openrouter_vision", fake_transport):
            with patch("controller.vision._get_primary_model",
                       return_value="anthropic/claude-3.5-sonnet"):
                result = describe_image(fake_image, "image/png", "What is this?")

        assert "[Vision]" in result
        assert "Connection refused" in result

    def test_empty_response_returns_placeholder(self):
        """Empty model response gets a graceful placeholder."""
        from controller.vision import describe_image

        fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        def fake_transport(messages, model="openrouter/free", timeout=60.0):
            return ""

        with patch("controller.vision._openrouter_vision", fake_transport):
            with patch("controller.vision._get_primary_model",
                       return_value="openai/chatgpt-4o"):
                result = describe_image(fake_image, "image/png", "What is this?")

        assert "empty response" in result.lower()

    def test_unexpected_exception_returns_error_string(self):
        """Non-RuntimeError exceptions are caught and returned as strings."""
        from controller.vision import describe_image

        fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        def bad_transport(messages, model="openrouter/free", timeout=60.0):
            raise ValueError("Unexpected failure")

        with patch("controller.vision._openrouter_vision", bad_transport):
            with patch("controller.vision._get_primary_model",
                       return_value="anthropic/claude-3.5-sonnet"):
                result = describe_image(fake_image, "image/png", "What is this?")

        assert "[Vision]" in result
        assert "Unexpected failure" in result


# ── _openrouter_vision transport ───────────────────────────────────────────────

class TestOpenRouterVisionTransport:
    """_openrouter_vision sends correct payload and returns text content."""

    def test_sends_vision_message_payload(self):
        from controller.vision import _openrouter_vision

        captured = {}

        class FakeORResponse:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "A blue sky.",
                        },
                        "finish_reason": "stop",
                    }],
                    "created": 0,
                }).encode()

        def capturing_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.headers)
            if req.data:
                captured["payload"] = json.loads(req.data.decode("utf-8"))
            return FakeORResponse()

        with patch.object(urllib.request, "urlopen", side_effect=capturing_urlopen):
            result = _openrouter_vision(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,abc123"
                            },
                        },
                    ],
                }],
                model="anthropic/claude-3.5-sonnet",
            )

        assert result == "A blue sky."
        assert "openrouter.ai" in captured["url"]
        assert captured["payload"]["model"] == "anthropic/claude-3.5-sonnet"

        # Message structure
        msgs = captured["payload"]["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "What is this?"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_http_error_raises_runtime_error(self):
        from controller.vision import _openrouter_vision

        # Use a plain RuntimeError (raised directly) — it simulates any HTTP error
        with patch.object(
            urllib.request, "urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://openrouter.ai/api/v1/chat/completions",
                code=502,
                msg="Bad Gateway",
                hdrs={},
                fp=None,
            ),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _openrouter_vision([{"role": "user", "content": []}])
            assert "502" in str(exc_info.value)

    def test_network_error_raises_runtime_error(self):
        from controller.vision import _openrouter_vision

        with patch.object(
            urllib.request, "urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _openrouter_vision([{"role": "user", "content": []}])
            assert "Connection refused" in str(exc_info.value)

    def test_json_error_in_body_raises_runtime_error(self):
        from controller.vision import _openrouter_vision

        class FakeMalformed:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b"not json at all"

        # Use return_value so the context manager is returned as-is
        with patch.object(
            urllib.request, "urlopen",
            return_value=FakeMalformed(),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _openrouter_vision([{"role": "user", "content": []}])
            assert "malformed" in str(exc_info.value).lower()

    def test_openrouter_json_error_raises_runtime_error(self):
        from controller.vision import _openrouter_vision

        class FakeORApiError:
            code = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return json.dumps({
                    "error": {"code": "model_not_found", "message": "Model unknown"}
                }).encode()

        # Use return_value for the context manager
        with patch.object(
            urllib.request, "urlopen",
            return_value=FakeORApiError(),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                _openrouter_vision([{"role": "user", "content": []}])
            assert "model_not_found" in str(exc_info.value)
