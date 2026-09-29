#!/usr/bin/env python3
"""
Tests for controller/main_brain.py — Ollama-first, OpenRouter `openrouter/free` fallback.

These tests verify:
  1. Ollama is tried first when available.
  2. OpenRouter `openrouter/free` is the fallback when Ollama is unavailable.
  3. RuntimeError is raised when both providers fail.
  4. Response shape is Ollama-compatible regardless of provider.
  5. Health cache avoids hammering Ollama on repeated checks.
  6. invalidate_ollama_health() forces a fresh health check.
  7. Tool calls are correctly converted from OpenAI → Ollama format.
  8. Message sanitization handles None content and tool-role messages.
  9. agent_controller integration with the main_brain module.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from unittest.mock import patch

import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_health_cache():
    """Clear the health cache before each test so isolation is guaranteed."""
    from controller import main_brain
    with main_brain._health_cache["lock"]:
        main_brain._health_cache["available"] = False
        main_brain._health_cache["ts"] = 0.0
    # Also clear the resilience layer's health cache
    try:
        from controller import main_brain_resilience
        with main_brain_resilience._health_lock:
            main_brain_resilience._openrouter_health_cache["available"] = False
            main_brain_resilience._openrouter_health_cache["ts"] = 0.0
    except Exception:
        pass
    # Install a no-op sleep into the resilience context so retries don't wait
    try:
        from controller import main_brain
        _saved_ctx = main_brain._resilience_ctx.copy()
        main_brain.configure_resilience(sleep_fn=lambda x: None)
    except Exception:
        _saved_ctx = None

    yield

    # Restore cache state
    with main_brain._health_cache["lock"]:
        main_brain._health_cache["available"] = False
        main_brain._health_cache["ts"] = 0.0
    try:
        from controller import main_brain_resilience
        with main_brain_resilience._health_lock:
            main_brain_resilience._openrouter_health_cache["available"] = False
            main_brain_resilience._openrouter_health_cache["ts"] = 0.0
    except Exception:
        pass
    # Restore resilience context
    if _saved_ctx is not None:
        main_brain._resilience_ctx = _saved_ctx


# ── is_ollama_available ─────────────────────────────────────────────────────────

class TestOllamaHealthCheck:
    """Tests for is_ollama_available() and the health cache."""

    def test_returns_true_when_ollama_is_up(self, reset_health_cache):
        fake_body = json.dumps({"models": []}).encode("utf-8")

        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return fake_body

        with patch.object(urllib.request, "urlopen", return_value=FakeResponse()):
            from controller.main_brain import is_ollama_available
            result = is_ollama_available()

        assert result is True

    def test_returns_false_when_ollama_is_down_connection_refused(self, reset_health_cache):
        with patch.object(
            urllib.request, "urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            from controller.main_brain import is_ollama_available
            result = is_ollama_available()

        assert result is False

    def test_returns_false_when_ollama_is_down_timeout(self, reset_health_cache):
        with patch.object(
            urllib.request, "urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            from controller.main_brain import is_ollama_available
            result = is_ollama_available()

        assert result is False

    def test_caches_result_for_health_ttl(self, reset_health_cache):
        """Second call within TTL returns cached result without a new HTTP request."""
        call_count = 0
        fake_body = json.dumps({"models": []}).encode("utf-8")

        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return fake_body

        def counting_urlopen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeResponse()

        with patch.object(urllib.request, "urlopen", counting_urlopen):
            from controller.main_brain import is_ollama_available
            is_ollama_available()
            is_ollama_available()
            is_ollama_available()

        assert call_count == 1, "Health check should be cached within TTL"

    def test_invalidate_health_cache_forces_recheck(self, reset_health_cache):
        """After invalidate_ollama_health(), the next check hits the network."""
        call_count = 0
        fake_body = json.dumps({"models": []}).encode("utf-8")

        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def read(self):
                return fake_body

        def counting_urlopen(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return FakeResponse()

        with patch.object(urllib.request, "urlopen", counting_urlopen):
            from controller.main_brain import is_ollama_available, invalidate_ollama_health

            is_ollama_available()
            is_ollama_available()  # cached
            assert call_count == 1

            invalidate_ollama_health()
            is_ollama_available()  # fresh check
            assert call_count == 2


# ── chat() main logic ───────────────────────────────────────────────────────────

class TestBrainChat:
    """Tests for the main chat() entry point."""

    def test_prefers_ollama_when_available(self, reset_health_cache):
        """When Ollama is available, it is used and OpenRouter is not called."""
        fake_ollama_resp = {
            "model": "gemma4",
            "done": True,
            "message": {"role": "assistant", "content": "hello from ollama"},
        }

        with patch("controller.main_brain.is_ollama_available", return_value=True):
            with patch("controller.main_brain._ollama_chat", return_value=(fake_ollama_resp, "ollama")) as mock_ollama:
                from controller.main_brain import chat
                resp, provider = chat(
                    model="gemma4",
                    messages=[{"role": "user", "content": "hi"}],
                )

        assert provider == "ollama"
        assert resp["message"]["content"] == "hello from ollama"
        assert mock_ollama.call_count == 1

    def test_falls_back_to_openrouter_free_when_ollama_is_down(self, reset_health_cache):
        """When Ollama is unavailable, OpenRouter `openrouter/free` is used."""
        fake_or_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "hello from openrouter free",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }],
            "created": 1234567890,
        }

        class FakeDown:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                raise urllib.error.URLError("Connection refused")

        class FakeOR:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps(fake_or_resp).encode()

        with patch.object(urllib.request, "urlopen") as mock_urlopen:
            mock_urlopen.side_effect = [
                FakeDown(),   # Ollama health check → down
                FakeOR(),     # OpenRouter → success
            ]
            from controller.main_brain import chat
            resp, provider = chat(
                model="gemma4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert provider == "openrouter"
        assert resp["message"]["content"] == "hello from openrouter free"

    def test_openrouter_fallback_uses_openrouter_free_model(self, reset_health_cache):
        """OpenRouter fallback always sends `openrouter/free` as the model."""
        captured_payloads = []

        class FakeDown:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                raise urllib.error.URLError("Connection refused")

        class FakeOR:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({
                    "choices": [{
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }],
                    "created": 0,
                }).encode()

        def capturing_urlopen(req, timeout=None):
            # Capture the OpenRouter request body so we can verify the model field
            if "openrouter" in req.full_url and hasattr(req, "data") and req.data:
                captured_payloads.append(json.loads(req.data.decode("utf-8")))
                return FakeOR()
            # Ollama health check → fails
            raise urllib.error.URLError("Connection refused")

        with patch.object(urllib.request, "urlopen", side_effect=capturing_urlopen):
            from controller.main_brain import chat
            resp, provider = chat(
                model="gemma4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert provider == "openrouter"
        assert len(captured_payloads) == 1, f"Expected 1 OpenRouter request, got {len(captured_payloads)}"
        assert captured_payloads[0]["model"] == "openrouter/free", (
            f"Expected model='openrouter/free', got {captured_payloads[0].get('model')!r}"
        )

    def test_raises_when_both_providers_fail(self, reset_health_cache):
        """When both Ollama is down and OpenRouter fails, RuntimeError is raised."""
        # Use callable side_effect so ALL retries (up to 11) get the same error.
        # The no-op sleep in reset_health_cache prevents real delays.
        with patch.object(urllib.request, "urlopen") as mock_urlopen:
            mock_urlopen.side_effect = lambda *a, **k: (_ for _ in ()).throw(
                urllib.error.URLError("Connection refused"))
            from controller.main_brain import chat
            with pytest.raises(RuntimeError) as exc_info:
                chat(model="gemma4", messages=[{"role": "user", "content": "hi"}])

            err = str(exc_info.value)
            assert "ollama" in err.lower() or "openrouter" in err.lower()

    def test_raises_when_openrouter_returns_error(self, reset_health_cache):
        """OpenRouter HTTP error is surfaced as RuntimeError."""
        class FakeDown:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                raise urllib.error.URLError("Connection refused")

        class FakeORError:
            code = 502
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return b"Bad Gateway"

        with patch.object(urllib.request, "urlopen") as mock_urlopen:
            # First call = Ollama health check; all subsequent calls = OpenRouter errors
            mock_urlopen.side_effect = [
                FakeDown(),      # Ollama health → down
                FakeORError(),   # OpenRouter → HTTP 502 (repeats for all retries)
            ]
            from controller.main_brain import chat
            with pytest.raises(RuntimeError) as exc_info:
                chat(model="gemma4", messages=[{"role": "user", "content": "hi"}])

            assert "openrouter" in str(exc_info.value).lower()

    def test_response_shape_is_ollama_compatible_from_openrouter(self, reset_health_cache):
        """OpenRouter response is converted to Ollama shape (model, done, message)."""
        fake_or_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "synthesized response",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }],
            "created": 1700000000,
        }

        class FakeDown:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                raise urllib.error.URLError("Connection refused")

        class FakeOR:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps(fake_or_resp).encode()

        with patch.object(urllib.request, "urlopen") as mock_urlopen:
            mock_urlopen.side_effect = [FakeDown(), FakeOR()]
            from controller.main_brain import chat
            resp, _provider = chat(
                model="gemma4",
                messages=[{"role": "user", "content": "hello"}],
            )

        # Ollama shape assertions
        assert "model" in resp
        assert "done" in resp
        assert "message" in resp
        assert resp["message"]["role"] == "assistant"
        assert resp["message"]["content"] == "synthesized response"
        assert resp.get("done_reason") == "stop"


# ── OpenRouter tool_calls conversion ───────────────────────────────────────────

class TestOpenRouterToolCalls:
    """Tests that tool_calls are correctly converted from OpenAI → Ollama format."""

    def test_tool_calls_converted_from_openai_to_ollama_shape(self):
        """OpenRouter tool_calls (OpenAI format) are converted to Ollama format."""
        fake_or_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "navigate_to",
                                "arguments": '{"url": "https://example.com"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "created": 1700000000,
        }

        class FakeDown:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                raise urllib.error.URLError("Connection refused")

        class FakeOR:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps(fake_or_resp).encode()

        with patch.object(urllib.request, "urlopen") as mock_urlopen:
            mock_urlopen.side_effect = [FakeDown(), FakeOR()]
            from controller.main_brain import chat
            resp, _ = chat(
                model="gemma4",
                messages=[{"role": "user", "content": "open example.com"}],
                tools=[{"type": "function", "function": {"name": "navigate_to", "parameters": {}}}],
            )

        tool_calls = resp["message"].get("tool_calls")
        assert tool_calls is not None, "tool_calls should be present"
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc["id"] == "call_abc123"
        assert tc["function"]["name"] == "navigate_to"
        # Arguments should be a dict, not a string
        assert isinstance(tc["function"]["arguments"], dict)
        assert tc["function"]["arguments"]["url"] == "https://example.com"


# ── _sanitize_messages_for_openrouter ─────────────────────────────────────────

class TestMessageSanitization:
    """Tests for _sanitize_messages_for_openrouter."""

    def test_strips_none_content(self):
        from controller.main_brain import _sanitize_messages_for_openrouter
        cleaned = _sanitize_messages_for_openrouter([
            {"role": "user", "content": None},
            {"role": "tool", "content": 123},  # numeric content
            {"role": "user", "content": "hello"},
        ])
        assert cleaned[0]["content"] == ""
        assert cleaned[1]["content"] == "123"
        assert cleaned[2]["content"] == "hello"

    def test_passes_through_tool_name(self):
        from controller.main_brain import _sanitize_messages_for_openrouter
        cleaned = _sanitize_messages_for_openrouter([
            {"role": "tool", "content": "result", "name": "navigate_to"},
        ])
        assert cleaned[0]["name"] == "navigate_to"

    def test_passes_through_tool_result_role(self):
        from controller.main_brain import _sanitize_messages_for_openrouter
        cleaned = _sanitize_messages_for_openrouter([
            {"role": "tool", "content": "done", "tool_call_id": "call_xyz"},
        ])
        assert cleaned[0]["role"] == "tool"
        assert cleaned[0]["content"] == "done"


# ── Agent controller integration ────────────────────────────────────────────────

class TestAgentControllerIntegration:
    """Verify agent_controller.py uses _brain for MODEL_BRAIN calls."""

    def test_agent_controller_imports_main_brain(self):
        """agent_controller imports _brain_chat from main_brain at module level."""
        from controller import agent_controller as ac
        assert hasattr(ac, "_BRAIN_PROVIDER_AVAILABLE")

    def test_agent_controller_has_brain_wrapper(self):
        """agent_controller defines a _brain() wrapper function."""
        from controller import agent_controller as ac
        assert callable(ac._brain)

    def test_brain_wrapper_signature(self):
        """_brain() accepts the expected parameters and returns a dict."""
        from controller import agent_controller as ac
        fake_resp = {
            "model": "gemma4",
            "done": True,
            "message": {"role": "assistant", "content": "mocked"},
        }
        with patch.object(ac, "_brain_chat", return_value=(fake_resp, "ollama")):
            result = ac._brain("gemma4", [{"role": "user", "content": "hi"}])
        assert isinstance(result, dict)
        assert result["message"]["content"] == "mocked"

    def test_brain_wrapper_provides_fallback_when_module_missing(self):
        """If main_brain can't be imported, _brain falls back to raw ollama_client."""
        from controller import agent_controller as ac
        orig_available = ac._BRAIN_PROVIDER_AVAILABLE
        orig_brain_chat = getattr(ac, "_brain_chat", None)

        ac._BRAIN_PROVIDER_AVAILABLE = False
        try:
            fake_resp = {
                "model": "gemma4",
                "done": True,
                "message": {"role": "assistant", "content": "via fallback"},
            }
            with patch.object(ac.ollama_client, "chat", return_value=fake_resp):
                result = ac._brain("gemma4", [{"role": "user", "content": "hi"}])
            assert result["message"]["content"] == "via fallback"
        finally:
            ac._BRAIN_PROVIDER_AVAILABLE = orig_available
            if orig_brain_chat is not None:
                ac._brain_chat = orig_brain_chat
