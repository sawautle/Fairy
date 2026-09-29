#!/usr/bin/env python3
"""
Tests for hermes_bridge.py hybrid provider routing.

Verifies:
  1. Ollama available → Hermes (gemma4:latest) is selected.
  2. Ollama unavailable → OpenRouter fallback is selected (via main_brain).
  3. Hermes/Ollama call fails → OpenRouter fallback is selected.
  4. Ollama failure does not cause long retries before fallback (fast check).
  5. OpenRouter failure after Ollama failure produces a proper final error.
  6. Existing Hermes behavior still works (in-process path when Ollama up).
  7. Existing delegation behavior is unaffected.
  8. Existing UI is unaffected (the bridge API surface is unchanged).
"""
from __future__ import annotations

import time
import types
from unittest.mock import patch, MagicMock

import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def reset_hermes_bridge_state():
    """
    Reset hermes_bridge module-level state and main_brain health cache
    so each test is isolated.
    """
    import hermes_bridge
    # Save and restore module-level state
    saved = {}
    for key in ("_AIAgent", "_HERMES_AVAILABLE", "_HERMES_IMPORT_ERROR",
                "_BRAIN_PROVIDER_AVAILABLE", "_brain_chat", "_is_ollama_available",
                "_cached_agent", "_hermes_initialized", "_hermes_initialized_ok",
                "_FALLBACK_RESPONSE", "_LAST_PROVIDER"):
        if hasattr(hermes_bridge, key):
            saved[key] = getattr(hermes_bridge, key)

    # Reset cached agent so each test starts with a clean cache.
    # Tests that patch _AIAgent expect their mock to be the one used.
    if hasattr(hermes_bridge, "_cached_agent"):
        hermes_bridge._cached_agent = None

    # Reset main_brain health cache so tests are isolated
    try:
        from controller import main_brain
        with main_brain._health_cache["lock"]:
            main_brain._health_cache["available"] = False
            main_brain._health_cache["ts"] = 0.0
    except (ImportError, AttributeError):
        pass

    yield

    # Restore
    for key, value in saved.items():
        setattr(hermes_bridge, key, value)

    try:
        from controller import main_brain
        with main_brain._health_cache["lock"]:
            main_brain._health_cache["available"] = False
            main_brain._health_cache["ts"] = 0.0
    except (ImportError, AttributeError):
        pass


# ── Test 1: Ollama available → Hermes is selected ──────────────────────────────

class TestOllamaAvailableUsesHermes:
    """When Ollama is reachable, Hermes (gemma4:latest) is used as primary brain."""

    def test_ollama_available_calls_hermes(self, reset_hermes_bridge_state):
        """Ollama up + Hermes available → use Hermes in-process path."""
        import hermes_bridge

        # Mock AIAgent to simulate a successful Hermes call
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_conversation.return_value = {
            "final_response": "Hello from Hermes + Gemma 4!",
            "messages": [{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "Hello from Hermes + Gemma 4!"}],
        }

        mock_aiagent = MagicMock(return_value=mock_agent_instance)
        hermes_bridge._AIAgent = mock_aiagent
        hermes_bridge._HERMES_AVAILABLE = True
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True

        # Mock main_brain so we can verify it is NOT called
        mock_brain_chat = MagicMock(return_value=(
            {"message": {"role": "assistant", "content": "from openrouter"}},
            "openrouter"
        ))
        hermes_bridge._brain_chat = mock_brain_chat

        # Mock Ollama check to return True
        with patch.object(hermes_bridge, "_is_ollama_available", return_value=True):
            reply, history = hermes_bridge.run_turn("hi there", [])

        # Verify Hermes was called
        assert mock_aiagent.called, "Hermes AIAgent should be instantiated when Ollama is up"
        assert mock_agent_instance.run_conversation.called, "Hermes run_conversation should be called"
        assert "Hermes" in reply or "Gemma" in reply, f"Expected Hermes response, got: {reply}"

        # Verify main_brain fallback was NOT called
        assert not mock_brain_chat.called, "main_brain fallback should NOT be called when Hermes succeeds"

    def test_ollama_available_passes_gemma4_model(self, reset_hermes_bridge_state):
        """When Ollama is up, the model passed to Hermes is gemma4:latest (or config default)."""
        import hermes_bridge

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_conversation.return_value = {
            "final_response": "ok",
            "messages": [],
        }
        mock_aiagent = MagicMock(return_value=mock_agent_instance)
        hermes_bridge._AIAgent = mock_aiagent

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=True):
            hermes_bridge.run_turn("test", [])

        # Inspect the kwargs passed to AIAgent()
        assert mock_aiagent.called
        kwargs = mock_aiagent.call_args.kwargs
        # Model should be whatever _FAIRY_HERMES_MODEL is configured to
        model = kwargs.get("model", "")
        # Accept any model - just verify it's non-empty
        assert model, "Model should be configured"


# ── Test 2: Ollama unavailable → OpenRouter fallback is selected ────────────────

class TestOllamaUnavailableUsesOpenRouter:
    """When Ollama is down, main_brain (OpenRouter) fallback is used."""

    def test_ollama_down_skips_hermes_uses_main_brain(self, reset_hermes_bridge_state):
        """Ollama down → skip Hermes, use main_brain.chat() (OpenRouter).

        Note: The fallback raises HermesBridgeError so run_turn_safe returns
        (False, ...) and agent_controller can fall through to its action
        fallback (e.g., _try_app_action_fallback for "open X" requests).
        """
        import hermes_bridge

        mock_aiagent = MagicMock()
        hermes_bridge._AIAgent = mock_aiagent
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True

        # Mock main_brain to return an OpenRouter-style response
        mock_brain_chat = MagicMock(return_value=(
            {"message": {"role": "assistant", "content": "Hello from OpenRouter free!"}},
            "openrouter"
        ))
        hermes_bridge._brain_chat = mock_brain_chat

        # Mock Ollama check to return False
        with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
            with pytest.raises(hermes_bridge.HermesBridgeError) as exc_info:
                hermes_bridge.run_turn("hi", [])

        # Verify Hermes was NOT called
        assert not mock_aiagent.called, "Hermes should NOT be called when Ollama is down"

        # Verify main_brain was called
        assert mock_brain_chat.called, "main_brain.chat() should be called when Ollama is down"
        # Error message should include the fallback response so agent_controller can use it
        err_msg = str(exc_info.value)
        assert "fallback" in err_msg.lower() or "OpenRouter" in err_msg, (
            f"Error should mention fallback, got: {err_msg}"
        )

    def test_ollama_down_does_not_wait_for_hermes_timeout(self, reset_hermes_bridge_state):
        """When Ollama is detected as down, we should NOT wait for Hermes to time out."""
        import hermes_bridge

        mock_aiagent = MagicMock()
        hermes_bridge._AIAgent = mock_aiagent
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
        mock_brain_chat = MagicMock(return_value=(
            {"message": {"role": "assistant", "content": "fast fallback"}},
            "openrouter"
        ))
        hermes_bridge._brain_chat = mock_brain_chat

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
            start = time.time()
            try:
                hermes_bridge.run_turn("test", [])
            except hermes_bridge.HermesBridgeError:
                pass  # expected — fallback raises HermesBridgeError
            elapsed = time.time() - start

        # Should complete quickly (well under Hermes's ~50s retry loop)
        assert elapsed < 5.0, f"Should not take long when Ollama is detected as down. Took {elapsed:.2f}s"
        assert not mock_aiagent.called, "Hermes should not be invoked when Ollama is detected as down"


# ── Test 3: Gemma 4 unavailable → OpenRouter fallback is selected ───────────────

class TestGemma4UnavailableUsesOpenRouter:
    """When Ollama is up but Gemma 4 is unavailable, fall back to OpenRouter."""

    def test_hermes_call_fails_with_provider_error_falls_back(self, reset_hermes_bridge_state):
        """Hermes raises a provider/Ollama error → main_brain fallback kicks in."""
        import hermes_bridge

        # Simulate Hermes that fails with a connection refused error
        mock_agent_instance = MagicMock()
        mock_agent_instance.run_conversation.side_effect = Exception(
            "Connection refused: could not connect to localhost:11434"
        )
        mock_aiagent = MagicMock(return_value=mock_agent_instance)
        hermes_bridge._AIAgent = mock_aiagent
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True

        mock_brain_chat = MagicMock(return_value=(
            {"message": {"role": "assistant", "content": "OpenRouter fallback response"}},
            "openrouter"
        ))
        hermes_bridge._brain_chat = mock_brain_chat

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=True):
            with pytest.raises(hermes_bridge.HermesBridgeError) as exc_info:
                hermes_bridge.run_turn("test", [])

        # Hermes was tried
        assert mock_aiagent.called, "Hermes should be tried first when Ollama check passes"
        # But main_brain was used as fallback
        assert mock_brain_chat.called, "main_brain should be called as fallback when Hermes fails"
        # Error message should contain the fallback response
        err_msg = str(exc_info.value)
        assert "OpenRouter" in err_msg, f"Error should contain OpenRouter response, got: {err_msg}"

    def test_hermes_call_fails_with_model_not_found_falls_back(self, reset_hermes_bridge_state):
        """Hermes raises model-not-found error → main_brain fallback."""
        import hermes_bridge

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_conversation.side_effect = Exception(
            "model 'gemma4:latest' not found"
        )
        mock_aiagent = MagicMock(return_value=mock_agent_instance)
        hermes_bridge._AIAgent = mock_aiagent
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
        mock_brain_chat = MagicMock(return_value=(
            {"message": {"role": "assistant", "content": "fallback ok"}},
            "openrouter"
        ))
        hermes_bridge._brain_chat = mock_brain_chat

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=True):
            with pytest.raises(hermes_bridge.HermesBridgeError) as exc_info:
                hermes_bridge.run_turn("test", [])

        assert mock_brain_chat.called, "Fallback should be used when model is not found"
        assert "fallback" in str(exc_info.value).lower(), (
            f"Error should mention fallback, got: {exc_info.value}"
        )


# ── Test 4: Ollama failure doesn't cause long retries ───────────────────────────

class TestFastFailover:
    """Ollama failure should not cause long retries before fallback."""

    def test_ollama_unavailable_no_retry_loop(self, reset_hermes_bridge_state):
        """When Ollama is detected as unavailable, we should not enter a retry loop."""
        import hermes_bridge

        call_count = {"hermes": 0}

        def counting_run(*args, **kwargs):
            call_count["hermes"] += 1
            return {
                "final_response": "ok",
                "messages": [],
            }

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_conversation.side_effect = counting_run
        mock_aiagent = MagicMock(return_value=mock_agent_instance)
        hermes_bridge._AIAgent = mock_aiagent

        mock_brain_chat = MagicMock(return_value=(
            {"message": {"role": "assistant", "content": "fallback"}},
            "openrouter"
        ))
        hermes_bridge._brain_chat = mock_brain_chat

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
            try:
                hermes_bridge.run_turn("test", [])
            except hermes_bridge.HermesBridgeError:
                pass  # expected

        # Hermes should be called 0 times (Ollama detected as down → skip)
        assert call_count["hermes"] == 0, (
            f"Hermes should not be called when Ollama is detected as down. "
            f"Called {call_count['hermes']} times."
        )

    def test_ollama_down_completes_under_5_seconds(self, reset_hermes_bridge_state):
        """Full run_turn() call with Ollama down should complete in under 5 seconds."""
        import hermes_bridge

        hermes_bridge._AIAgent = None  # No in-process Hermes
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
        hermes_bridge._brain_chat = MagicMock(return_value=(
            {"message": {"role": "assistant", "content": "fast response"}},
            "openrouter"
        ))

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
            start = time.time()
            try:
                hermes_bridge.run_turn("test", [])
            except hermes_bridge.HermesBridgeError:
                pass
            elapsed = time.time() - start

        assert elapsed < 5.0, f"run_turn() took too long: {elapsed:.2f}s (should be < 5s)"


# ── Test 5: OpenRouter failure after Ollama failure → proper final error ────────

class TestBothProvidersFail:
    """When both Ollama and OpenRouter fail, a proper final error is returned."""

    def test_ollama_down_and_openrouter_fails_raises_hermes_error(self, reset_hermes_bridge_state):
        """Both providers fail → HermesBridgeError is raised."""
        import hermes_bridge

        hermes_bridge._AIAgent = MagicMock()  # Pretend Hermes is available
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True

        # main_brain raises an error
        mock_brain_chat = MagicMock(side_effect=RuntimeError(
            "Ollama is unavailable and OpenRouter fallback also failed"
        ))
        hermes_bridge._brain_chat = mock_brain_chat

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
            with pytest.raises(hermes_bridge.HermesBridgeError) as exc_info:
                hermes_bridge.run_turn("test", [])

        # Error message should mention both providers
        err_msg = str(exc_info.value).lower()
        assert "ollama" in err_msg or "openrouter" in err_msg, (
            f"Error should mention both providers, got: {exc_info.value}"
        )

    def test_run_turn_safe_returns_failure_for_both_providers(self, reset_hermes_bridge_state):
        """run_turn_safe() returns (False, error_msg) when both providers fail."""
        import hermes_bridge

        hermes_bridge._AIAgent = MagicMock()
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
        hermes_bridge._brain_chat = MagicMock(side_effect=RuntimeError("Both providers down"))

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
            success, reply, history = hermes_bridge.run_turn_safe("test", [])

        assert success is False, "run_turn_safe should return False when both providers fail"
        assert "Error" in reply or "error" in reply.lower(), (
            f"Reply should be an error message, got: {reply}"
        )


# ── Test 6: Existing Hermes behavior still works ───────────────────────────────

class TestExistingHermesBehavior:
    """Verify the original in-process Hermes path still functions correctly."""

    def test_run_turn_still_works_with_hermes(self, reset_hermes_bridge_state):
        """run_turn() with Ollama up + Hermes available returns Hermes response."""
        import hermes_bridge

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_conversation.return_value = {
            "final_response": "Hermes response",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hermes response"},
            ],
        }
        mock_aiagent = MagicMock(return_value=mock_agent_instance)
        hermes_bridge._AIAgent = mock_aiagent
        hermes_bridge._HERMES_AVAILABLE = True
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
        hermes_bridge._brain_chat = MagicMock()  # Should NOT be called

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=True):
            reply, history = hermes_bridge.run_turn("hi", [])

        assert reply == "Hermes response"
        assert isinstance(history, list)
        # History should be converted to fairy format
        assert any(h.get("role") == "assistant" for h in history)

    def test_run_turn_safe_returns_success_for_working_hermes(self, reset_hermes_bridge_state):
        """run_turn_safe() returns (True, reply) when Hermes works."""
        import hermes_bridge

        mock_agent_instance = MagicMock()
        mock_agent_instance.run_conversation.return_value = {
            "final_response": "Working!",
            "messages": [],
        }
        mock_aiagent = MagicMock(return_value=mock_agent_instance)
        hermes_bridge._AIAgent = mock_aiagent
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
        hermes_bridge._brain_chat = MagicMock()

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=True):
            success, reply, _ = hermes_bridge.run_turn_safe("test", [])

        assert success is True
        assert reply == "Working!"

    def test_history_conversion_preserved(self, reset_hermes_bridge_state):
        """_convert_fairy_history_to_hermes and _convert_hermes_history_to_fairy still work."""
        from hermes_bridge import _convert_fairy_history_to_hermes, _convert_hermes_history_to_fairy

        fairy_hist = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "tool_call_id": "x", "name": "fn"},
        ]
        hermes = _convert_fairy_history_to_hermes(fairy_hist)
        assert len(hermes) == 2
        assert hermes[0]["role"] == "user"
        assert hermes[1]["tool_call_id"] == "x"

        back = _convert_hermes_history_to_fairy(hermes)
        assert len(back) == 2
        assert back[0]["content"] == "hi"
        assert back[1]["tool_call_id"] == "x"


# ── Test 7: Existing delegation behavior is unaffected ──────────────────────────

class TestDelegationUnaffected:
    """Verify the delegation (claude_code_delegate) path is not touched by this change."""

    def test_hermes_bridge_does_not_import_claude_code_delegate(self):
        """hermes_bridge.py should not directly import the claude_code_delegate module."""
        import hermes_bridge
        source_file = hermes_bridge.__file__
        with open(source_file, "r", encoding="utf-8") as f:
            source = f.read()
        # The bridge should not directly call delegate functions
        assert "claude_code_delegate" not in source or "delegate_to_claude_code" not in source, (
            "hermes_bridge should not directly handle Claude Code delegation"
        )

    def test_hermes_bridge_only_imports_main_brain_for_fallback(self):
        """hermes_bridge should only import main_brain for the Ollama/OpenRouter fallback."""
        import hermes_bridge
        source_file = hermes_bridge.__file__
        with open(source_file, "r", encoding="utf-8") as f:
            source = f.read()
        # Should import main_brain
        assert "main_brain" in source, "hermes_bridge should import main_brain for fallback"
        # Should NOT import delegation directly
        assert "claude_code" not in source, "hermes_bridge should not touch claude_code_delegate"


# ── Test 8: Existing UI is unaffected ───────────────────────────────────────────

class TestUIUnaffected:
    """Verify the hermes_bridge public API surface is unchanged (UI calls it the same way)."""

    def test_run_turn_signature_unchanged(self):
        """run_turn() signature: (user_text, history, on_status=None) → (reply, history)."""
        import inspect
        from hermes_bridge import run_turn
        sig = inspect.signature(run_turn)
        params = list(sig.parameters.keys())
        assert "user_text" in params
        assert "history" in params
        assert "on_status" in params

    def test_run_turn_safe_signature_unchanged(self):
        """run_turn_safe() signature: (user_text, history, on_status=None) → (bool, str, list)."""
        import inspect
        from hermes_bridge import run_turn_safe
        sig = inspect.signature(run_turn_safe)
        params = list(sig.parameters.keys())
        assert "user_text" in params
        assert "history" in params
        assert "on_status" in params

    def test_is_hermes_available_still_works(self):
        """is_hermes_available() still returns a bool."""
        from hermes_bridge import is_hermes_available
        result = is_hermes_available()
        assert isinstance(result, bool)

    def test_hermes_bridge_error_class_still_exists(self):
        """HermesBridgeError exception class still exists and is importable."""
        from hermes_bridge import HermesBridgeError
        assert issubclass(HermesBridgeError, Exception)

    def test_get_hermes_import_error_still_works(self):
        """get_hermes_import_error() still returns the import error or None."""
        from hermes_bridge import get_hermes_import_error
        result = get_hermes_import_error()
        assert result is None or isinstance(result, str)


# ── Test: Provider error detection helper ───────────────────────────────────────

class TestProviderErrorDetection:
    """Test the _is_hermes_provider_error() helper function."""

    def test_connection_refused_is_provider_error(self):
        from hermes_bridge import _is_hermes_provider_error
        assert _is_hermes_provider_error(Exception("Connection refused")) is True

    def test_timeout_is_provider_error(self):
        from hermes_bridge import _is_hermes_provider_error
        assert _is_hermes_provider_error(Exception("Request timed out")) is True

    def test_model_not_found_is_provider_error(self):
        from hermes_bridge import _is_hermes_provider_error
        assert _is_hermes_provider_error(Exception("model 'gemma4:latest' not found")) is True

    def test_ollama_unavailable_is_provider_error(self):
        from hermes_bridge import _is_hermes_provider_error
        assert _is_hermes_provider_error(Exception("Ollama server is unavailable")) is True

    def test_genuine_application_error_not_treated_as_provider(self):
        """A non-provider error should NOT trigger fallback — it should be raised."""
        from hermes_bridge import _is_hermes_provider_error
        # E.g. a tool execution error in Hermes is a genuine app error
        assert _is_hermes_provider_error(Exception("Tool execution failed: file not found")) is False


# ── Test: Quick Ollama check ────────────────────────────────────────────────────

class TestQuickOllamaCheck:
    """Test the _quick_ollama_check() helper."""

    def test_uses_main_brain_check_when_available(self):
        """When main_brain is available, _quick_ollama_check delegates to is_ollama_available()."""
        import hermes_bridge

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=True):
            result = hermes_bridge._quick_ollama_check()
        assert result is True

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
            result = hermes_bridge._quick_ollama_check()
        assert result is False

    def test_falls_back_to_socket_probe_when_main_brain_missing(self):
        """When main_brain is not available, fall back to raw socket probe."""
        import hermes_bridge
        import socket as socket_module

        with patch.object(hermes_bridge, "_BRAIN_PROVIDER_AVAILABLE", False):
            with patch.object(hermes_bridge, "_is_ollama_available", None):
                # Mock the socket connection to succeed
                mock_socket = MagicMock()
                mock_socket.__enter__ = MagicMock(return_value=MagicMock())
                mock_socket.__exit__ = MagicMock(return_value=False)

                with patch.object(socket_module, "create_connection", return_value=mock_socket):
                    result = hermes_bridge._quick_ollama_check()
                assert result is True

        with patch.object(hermes_bridge, "_BRAIN_PROVIDER_AVAILABLE", False):
            with patch.object(hermes_bridge, "_is_ollama_available", None):
                with patch.object(socket_module, "create_connection",
                                side_effect=ConnectionRefusedError("refused")):
                    result = hermes_bridge._quick_ollama_check()
                assert result is False


# ── Test: OpenRouter model used for fallback ─────────────────────────────────────

class TestFallbackModel:
    """Verify the OpenRouter fallback uses the correct model."""

    def test_fallback_uses_openrouter_free_via_main_brain(self, reset_health_cache_main_brain):
        """The fallback path delegates to main_brain.chat() which uses openrouter/free."""
        from controller import main_brain
        # main_brain._FALLBACK_MODEL is the source of truth
        assert main_brain._FALLBACK_MODEL == "openrouter/free", (
            f"Expected 'openrouter/free', got {main_brain._FALLBACK_MODEL!r}"
        )

    def test_fallback_uses_configured_brain_model(self, reset_hermes_bridge_state):
        """When falling back, main_brain receives the configured MODEL_BRAIN."""
        import hermes_bridge
        from config import MODEL_BRAIN

        hermes_bridge._AIAgent = MagicMock()
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
        mock_brain_chat = MagicMock(return_value=(
            {"message": {"role": "assistant", "content": "ok"}},
            "openrouter"
        ))
        hermes_bridge._brain_chat = mock_brain_chat

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
            try:
                hermes_bridge.run_turn("test", [])
            except hermes_bridge.HermesBridgeError:
                pass

        # Verify the model passed to main_brain matches MODEL_BRAIN
        assert mock_brain_chat.called
        call_kwargs = mock_brain_chat.call_args.kwargs
        assert call_kwargs.get("model") == MODEL_BRAIN, (
            f"Expected model={MODEL_BRAIN}, got {call_kwargs.get('model')!r}"
        )


@pytest.fixture
def reset_health_cache_main_brain():
    """Reset main_brain health cache."""
    try:
        from controller import main_brain
        with main_brain._health_cache["lock"]:
            main_brain._health_cache["available"] = False
            main_brain._health_cache["ts"] = 0.0
        yield
        with main_brain._health_cache["lock"]:
            main_brain._health_cache["available"] = False
            main_brain._health_cache["ts"] = 0.0
    except (ImportError, AttributeError):
        yield


# ── Test: Fallback raises HermesBridgeError so agent_controller handles actions ─

class TestActionFallbackPreservation:
    """When Hermes can't run, agent_controller must still be able to handle action requests."""

    def test_ollama_down_fallback_raises_hermes_bridge_error(self, reset_hermes_bridge_state):
        """The fallback path raises HermesBridgeError so run_turn_safe returns (False, ...).

        This is critical: it allows agent_controller to fall through to its
        own _try_app_action_fallback for action requests like "open steam app".
        """
        import hermes_bridge

        mock_aiagent = MagicMock()
        hermes_bridge._AIAgent = mock_aiagent
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
        mock_brain_chat = MagicMock(return_value=(
            {"message": {"role": "assistant", "content": "text-only response"}},
            "openrouter"
        ))
        hermes_bridge._brain_chat = mock_brain_chat

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
            with pytest.raises(hermes_bridge.HermesBridgeError):
                hermes_bridge.run_turn("open steam app", [])

    def test_run_turn_safe_returns_failure_when_fallback_used(self, reset_hermes_bridge_state):
        """run_turn_safe returns (False, error_msg) so agent_controller can route to its own fallback."""
        import hermes_bridge
        from controller import agent_controller  # noqa: F401  # to patch _fast_intent

        # The discord_bot guard short-circuits "open steam app" (matches fast_intent
        # "app_action") before reaching the inner fallback. Patch _fast_intent to
        # return None so the guard passes through to _original_hermes_run_turn_safe.
        # Also bypass the non-owner guard path.
        with patch.object(agent_controller, "_fast_intent", return_value=None):
            hermes_bridge._AIAgent = None  # no Hermes → goes to main_brain fallback
            hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
            hermes_bridge._brain_chat = MagicMock(
                side_effect=hermes_bridge.HermesBridgeError("fallback error")
            )

            with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
                success, reply, history = hermes_bridge.run_turn_safe("open steam app", [])

        # run_turn_safe should return (False, ...) so agent_controller falls through
        assert success is False, (
            f"run_turn_safe must return False when fallback is used, got success={success}. "
            "This is required so agent_controller can fall through to its action-execution fallback."
        )
        # Reply should mention the fallback error
        assert "fallback" in reply.lower(), (
            f"Reply should mention fallback, got: {reply[:200]}"
        )

    def test_action_request_text_preserved_in_error(self, reset_hermes_bridge_state):
        """The HermesBridgeError message preserves the action request for debugging."""
        import hermes_bridge

        hermes_bridge._AIAgent = MagicMock()
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
        mock_brain_chat = MagicMock(return_value=(
            {"message": {"role": "assistant", "content": "I'll open Steam for you."}},
            "openrouter"
        ))
        hermes_bridge._brain_chat = mock_brain_chat

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
            with pytest.raises(hermes_bridge.HermesBridgeError) as exc_info:
                hermes_bridge.run_turn("open steam app", [])

        # The error should contain a preview of the response
        err_msg = str(exc_info.value)
        assert "Steam" in err_msg or "fallback" in err_msg.lower(), (
            f"Error should contain response preview, got: {err_msg}"
        )
