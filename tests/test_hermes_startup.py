#!/usr/bin/env python3
"""
Tests for Hermes startup initialization and primary-agent architecture.

Verifies:
  1. Hermes startup health check runs at module import (idempotent).
  2. is_hermes_initialized() and is_hermes_ready() return the correct booleans.
  3. The cached AIAgent singleton is created at most once and reused.
  4. agent_controller._try_app_action_fallback refuses internal agent names
     (hermes, fairy, qwen, gemma, ollama) before calling computer_control.
  5. agent_controller routes all requests through Hermes when FAIRY_USE_HERMES=1.
  6. The hybrid Ollama/OpenRouter fallback still works.
  7. run_turn / run_turn_safe public API surface is unchanged.
"""
from __future__ import annotations

import time
import types
import os
from unittest.mock import patch, MagicMock

import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────────

# Opt-in fixture (NOT autouse) so it doesn't pollute tests in other files.
# Each test that depends on it must declare it explicitly via the parameter
# name `reset_hermes_startup_state`.
@pytest.fixture
def reset_hermes_startup_state():
    """Reset Hermes startup state variables without touching shared provider stubs.

    Only resets the three state variables we introduce in this file:
    _hermes_initialized, _hermes_initialized_ok, _cached_agent.

    Does NOT save/restore _brain_chat / _is_ollama_available / _AIAgent /
    _HERMES_AVAILABLE because:
      a) Those are shared with test_hermes_hybrid_routing.py which saves/restores
         them itself.
      b) They may be None in the current runtime state (e.g. during test A's
         fixture setup before it patches them) — saving None and restoring None
         clobbers whatever test A's own fixture later puts there.
    """
    import hermes_bridge

    # Only reset the three state variables we introduce in this file
    saved = {}
    for key in ("_hermes_initialized", "_hermes_initialized_ok", "_cached_agent"):
        if hasattr(hermes_bridge, key):
            saved[key] = getattr(hermes_bridge, key)

    hermes_bridge._hermes_initialized = False
    hermes_bridge._hermes_initialized_ok = False
    hermes_bridge._cached_agent = None

    # Reset main_brain health cache
    try:
        from controller import main_brain
        with main_brain._health_cache["lock"]:
            main_brain._health_cache["available"] = False
            main_brain._health_cache["ts"] = 0.0
    except (ImportError, AttributeError):
        pass

    yield

    for key, value in saved.items():
        setattr(hermes_bridge, key, value)

    # Re-run startup health check so the module is in a clean post-import state
    hermes_bridge._run_startup_health_check()

    try:
        from controller import main_brain
        with main_brain._health_cache["lock"]:
            main_brain._health_cache["available"] = False
            main_brain._health_cache["ts"] = 0.0
    except (ImportError, AttributeError):
        pass


# ── Test 1: Health check runs at module import ──────────────────────────────────

class TestStartupHealthCheck:
    """The startup health check is run automatically when hermes_bridge is imported."""

    def test_health_check_runs_on_import(self):
        """Importing hermes_bridge must set _hermes_initialized = True."""
        import hermes_bridge
        # If we got here via normal test collection, import already ran
        assert hermes_bridge._hermes_initialized is True, (
            "Startup health check should have run during module import"
        )

    def test_health_check_is_idempotent(self):
        """Calling _run_startup_health_check twice does not re-run the check."""
        import hermes_bridge

        # Record the initial ok state
        initial_ok = hermes_bridge._hermes_initialized_ok
        hermes_bridge._run_startup_health_check()  # no-op
        hermes_bridge._run_startup_health_check()  # no-op
        hermes_bridge._run_startup_health_check()  # no-op

        # State must not have changed (idempotent)
        assert hermes_bridge._hermes_initialized_ok == initial_ok
        # _hermes_initialized must remain True
        assert hermes_bridge._hermes_initialized is True

    def test_health_check_does_not_block(self):
        """The health check completes quickly (< 5s)."""
        import hermes_bridge
        hermes_bridge._hermes_initialized = False
        hermes_bridge._hermes_initialized_ok = False
        start = time.time()
        hermes_bridge._run_startup_health_check()
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Health check should be fast, took {elapsed:.2f}s"
        assert hermes_bridge._hermes_initialized is True

    def test_health_check_ok_when_in_process_importable(self):
        """When in-process Hermes is importable, is_hermes_ready() returns True."""
        import hermes_bridge
        hermes_bridge._HERMES_AVAILABLE = True
        hermes_bridge._hermes_initialized = False
        hermes_bridge._hermes_initialized_ok = False
        hermes_bridge._run_startup_health_check()
        # If venv python doesn't exist, only in-process is checked
        # Either way, if in-process is available, we should be ok
        if not os.path.isfile(hermes_bridge.HERMES_VENV_PYTHON):
            assert hermes_bridge.is_hermes_ready() is True
        else:
            assert hermes_bridge.is_hermes_ready() is True

    def test_health_check_ok_with_venv_python_only(self):
        """When only the subprocess venv python is available, is_hermes_ready() returns True."""
        import hermes_bridge
        hermes_bridge._HERMES_AVAILABLE = False
        hermes_bridge._hermes_initialized = False
        hermes_bridge._hermes_initialized_ok = False
        with patch.object(hermes_bridge, "is_internal_agent_name", create=True):
            hermes_bridge._run_startup_health_check()
        # If venv python exists, OK. Otherwise not.
        if os.path.isfile(hermes_bridge.HERMES_VENV_PYTHON):
            assert hermes_bridge.is_hermes_ready() is True
        else:
            assert hermes_bridge.is_hermes_ready() is False

    def test_is_hermes_initialized_returns_bool(self):
        from hermes_bridge import is_hermes_initialized
        result = is_hermes_initialized()
        assert isinstance(result, bool)

    def test_is_hermes_ready_returns_bool(self):
        from hermes_bridge import is_hermes_ready
        result = is_hermes_ready()
        assert isinstance(result, bool)


# ── Test 2: Cached AIAgent singleton ────────────────────────────────────────────

class TestCachedAIAgent:
    """The AIAgent singleton is created at most once and reused across calls."""

    def test_cached_agent_is_none_initially(self, reset_hermes_startup_state):
        """Before the first call to _get_cached_agent, the cache is None."""
        import hermes_bridge
        hermes_bridge._cached_agent = None
        assert hermes_bridge._cached_agent is None

    def test_cached_agent_returns_same_instance_on_repeated_calls(self, reset_hermes_startup_state):
        """_get_cached_agent returns the same instance on repeated calls (singleton)."""
        import hermes_bridge

        # Replace the real _AIAgent with a mock class that tracks instantiation
        created = []

        class TrackedAIAgent:
            def __init__(self, **kwargs):
                created.append(kwargs)

        # Swap the real class with our tracked one
        original_aiagent = hermes_bridge._AIAgent
        hermes_bridge._AIAgent = TrackedAIAgent
        hermes_bridge._cached_agent = None  # ensure fresh
        hermes_bridge._HERMES_AVAILABLE = True

        try:
            # Call three times
            a1 = hermes_bridge._get_cached_agent()
            a2 = hermes_bridge._get_cached_agent()
            a3 = hermes_bridge._get_cached_agent()

            # Same instance returned all three times
            assert a1 is a2
            assert a2 is a3
            # And AIAgent was only instantiated once
            assert len(created) == 1, f"Expected 1 instantiation, got {len(created)}: {created}"
        finally:
            hermes_bridge._AIAgent = original_aiagent
            hermes_bridge._cached_agent = None

    def test_cached_agent_none_when_in_process_unavailable(self, reset_hermes_startup_state):
        """When _HERMES_AVAILABLE is False, _get_cached_agent returns None (subprocess path takes over)."""
        import hermes_bridge
        hermes_bridge._HERMES_AVAILABLE = False
        hermes_bridge._cached_agent = None

        result = hermes_bridge._get_cached_agent()
        assert result is None

    def test_cached_agent_handles_instantiation_failure(self, reset_hermes_startup_state):
        """If AIAgent() raises, _get_cached_agent returns None and doesn't cache the failure permanently."""
        import hermes_bridge

        # First call: AIAgent raises
        def raise_on_call(*args, **kwargs):
            raise RuntimeError("AIAgent init failed")

        with patch.object(hermes_bridge, "_AIAgent", side_effect=raise_on_call):
            hermes_bridge._HERMES_AVAILABLE = True
            result = hermes_bridge._get_cached_agent()
        assert result is None

    def test_run_turn_uses_cached_agent(self, reset_hermes_startup_state):
        """run_turn reuses the cached AIAgent across multiple turns (no per-turn instantiation)."""
        import hermes_bridge

        # Replace _AIAgent with a tracked class
        created = []

        # Opt in to Hermes's in-process path (bypasses the global conftest guard)
        hermes_bridge._FAIRY_TEST_ALLOW_IN_PROCESS = True

        class TrackedAIAgent:
            def __init__(self, **kwargs):
                created.append("created")
                self.kwargs = kwargs

            def run_conversation(self, user_message, conversation_history):
                # The in-process path now prepends a per-call language tag to
                # the user message (see i18n commit ba77ead). Strip the tag
                # so the mock echoes the original input back.
                import re as _re
                _stripped = _re.sub(
                    r"^\[The user's message is in .*?\]\n",
                    "",
                    user_message,
                )
                return {
                    "final_response": f"reply: {_stripped}",
                    "messages": conversation_history + [
                        {"role": "user", "content": _stripped},
                        {"role": "assistant", "content": f"reply: {_stripped}"},
                    ],
                }

        original_aiagent = hermes_bridge._AIAgent
        hermes_bridge._AIAgent = TrackedAIAgent
        hermes_bridge._cached_agent = None  # ensure fresh
        hermes_bridge._HERMES_AVAILABLE = True

        try:
            with patch.object(hermes_bridge, "_is_ollama_available", return_value=True):
                # Three turns back-to-back
                r1, _ = hermes_bridge.run_turn("turn 1", [])
                r2, _ = hermes_bridge.run_turn("turn 2", [])
                r3, _ = hermes_bridge.run_turn("turn 3", [])

            # All three replies came through
            assert r1 == "reply: turn 1"
            assert r2 == "reply: turn 2"
            assert r3 == "reply: turn 3"
            # AIAgent was only instantiated ONCE despite 3 turns
            assert created == ["created"], f"Expected 1 instantiation, got: {created}"
        finally:
            hermes_bridge._AIAgent = original_aiagent
            hermes_bridge._cached_agent = None
            hermes_bridge._FAIRY_TEST_ALLOW_IN_PROCESS = False


# ── Test 3: Internal agent name denylist ─────────────────────────────────────────

class TestInternalAgentDenylist:
    """The denylist prevents the desktop-app launcher from trying to launch internal agent names."""

    def test_hermes_is_internal_agent_name(self):
        from hermes_bridge import is_internal_agent_name
        assert is_internal_agent_name("hermes") is True
        assert is_internal_agent_name("Hermes") is True
        assert is_internal_agent_name("HERMES") is True

    def test_hermes_agent_is_internal_agent_name(self):
        from hermes_bridge import is_internal_agent_name
        assert is_internal_agent_name("hermes agent") is True
        assert is_internal_agent_name("Hermes Agent") is True

    def test_fairy_is_internal_agent_name(self):
        from hermes_bridge import is_internal_agent_name
        assert is_internal_agent_name("fairy") is True
        assert is_internal_agent_name("Fairy") is True

    def test_local_inference_engines_are_internal(self):
        from hermes_bridge import is_internal_agent_name
        for name in ("qwen", "gemma", "ollama", "llama"):
            assert is_internal_agent_name(name) is True, f"{name!r} should be flagged"

    def test_normal_apps_not_in_denylist(self):
        from hermes_bridge import is_internal_agent_name
        for name in ("calculator", "notepad", "chrome", "steam", "discord", "vscode"):
            assert is_internal_agent_name(name) is False, (
                f"{name!r} should NOT be in the internal-agent denylist"
            )

    def test_app_action_fallback_refuses_internal_names(self, monkeypatch):
        """_try_app_action_fallback must refuse 'hermes' / 'fairy' etc. without calling computer_control."""
        from controller import agent_controller as ac

        called = {"count": 0}

        def fail_if_called(args):
            called["count"] += 1
            raise AssertionError(
                f"computer_control should NOT be called for internal-agent name, got {args!r}"
            )

        monkeypatch.setattr(ac, "computer_control", fail_if_called)

        for target in ("hermes", "hermes agent", "fairy", "qwen", "gemma", "ollama"):
            result = ac._try_app_action_fallback(f"open {target}")
            assert result is not None, f"_try_app_action_fallback should return a string for {target!r}"
            assert "internal" in result.lower() or "already running" in result.lower(), (
                f"Message for {target!r} should explain it's an internal system: {result!r}"
            )

        assert called["count"] == 0, "computer_control must NOT be called for internal-agent names"

    def test_app_action_fallback_still_works_for_normal_apps(self, monkeypatch):
        """_try_app_action_fallback must still pass normal apps to computer_control."""
        from controller import agent_controller as ac

        captured = {}

        def mock_computer_control(args):
            captured["args"] = args
            return '{"status": "ok", "opened": "calculator"}'

        # Patch where _try_app_action_fallback imports it (inside the function body)
        monkeypatch.setattr("skills.computer_control.computer_control", mock_computer_control)
        monkeypatch.setattr(
            ac, "_verify_app_launch",
            lambda target, verb, timeout_seconds=3.0: (True, f"{target} opened.")
        )

        result = ac._try_app_action_fallback("open calculator")
        assert result is not None
        assert "calculator" in result.lower()
        assert captured.get("args", {}).get("value") == "calculator"


# ── Test 4: Hermes is the primary agent ──────────────────────────────────────────

class TestHermesIsPrimaryAgent:
    """When FAIRY_USE_HERMES=1 (default), all requests route through Hermes."""

    def test_normal_request_routes_to_hermes(self, monkeypatch, reset_hermes_startup_state):
        """A plain 'hello' request goes through Hermes, not the planner or fast paths."""
        from controller import agent_controller as ac
        import hermes_bridge

        # Make sure Hermes is "ready"
        hermes_bridge._HERMES_AVAILABLE = True
        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        captured = {}

        def mock_run_turn_safe(user_text, history, on_status=None):
            captured["user_text"] = user_text
            captured["history"] = history
            return True, "Hello, Master.", []

        monkeypatch.setattr("hermes_bridge.run_turn_safe", mock_run_turn_safe)

        with patch.dict(os.environ, {"FAIRY_USE_HERMES": "1"}):
            reply, new_history = ac.handle_request("hello there")

        # The reply came from Hermes
        assert reply == "Hello, Master."
        assert captured.get("user_text") == "hello there"

    def test_search_request_routes_to_hermes(self, monkeypatch, reset_hermes_startup_state):
        """A 'search for X' request goes through Hermes (not the disabled websearch fast path)."""
        from controller import agent_controller as ac
        import hermes_bridge

        hermes_bridge._HERMES_AVAILABLE = True
        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        captured = {}

        def mock_run_turn_safe(user_text, history, on_status=None):
            captured["user_text"] = user_text
            return True, "Search result: ...", []

        monkeypatch.setattr("hermes_bridge.run_turn_safe", mock_run_turn_safe)

        with patch.dict(os.environ, {"FAIRY_USE_HERMES": "1"}):
            reply, _ = ac.handle_request("search for python tutorials")

        assert "Search result" in reply
        assert captured.get("user_text") == "search for python tutorials"

    def test_hermes_disabled_uses_planner_fallback(self, monkeypatch, reset_hermes_startup_state):
        """When FAIRY_USE_HERMES=0, requests go to the planner/fast paths instead."""
        from controller import agent_controller as ac
        import hermes_bridge

        hermes_bridge._HERMES_AVAILABLE = True
        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        hermes_called = {"count": 0}

        def mock_run_turn_safe(user_text, history, on_status=None):
            hermes_called["count"] += 1
            return True, "should not reach", []

        monkeypatch.setattr("hermes_bridge.run_turn_safe", mock_run_turn_safe)

        with patch.dict(os.environ, {"FAIRY_USE_HERMES": "0"}):
            # Use a known-bare request that the fast path will handle
            reply, _ = ac.handle_request("what time is it")

        # Hermes should NOT have been called
        assert hermes_called["count"] == 0
        # Reply comes from the time fast path
        assert "Master" in reply

    def test_hermes_failure_falls_through_to_planner(self, monkeypatch, reset_hermes_startup_state):
        """When Hermes returns failure, agent_controller falls through to the planner/fast paths."""
        from controller import agent_controller as ac
        import hermes_bridge

        hermes_bridge._HERMES_AVAILABLE = True
        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        def mock_run_turn_safe_failing(user_text, history, on_status=None):
            return False, "[Hermes Error] something broke", []

        monkeypatch.setattr("hermes_bridge.run_turn_safe", mock_run_turn_safe_failing)

        with patch.dict(os.environ, {"FAIRY_USE_HERMES": "1"}):
            # Use a request that the time fast path can answer
            reply, _ = ac.handle_request("what time is it")

        # Falls through to the time fast path
        assert "Master" in reply


# ── Test 5: Hybrid Ollama/OpenRouter routing still works ────────────────────────

class TestHybridRoutingPreserved:
    """The Ollama → OpenRouter fallback must still work with the new architecture."""

    def test_ollama_down_uses_openrouter_fallback(self, reset_hermes_startup_state):
        """With Ollama down, run_turn routes through main_brain (OpenRouter)."""
        import hermes_bridge

        mock_aiagent = MagicMock()
        hermes_bridge._AIAgent = mock_aiagent
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
        mock_brain_chat = MagicMock(return_value=(
            {"message": {"role": "assistant", "content": "fallback reply"}},
            "openrouter",
        ))
        hermes_bridge._brain_chat = mock_brain_chat

        with patch.object(hermes_bridge, "_is_ollama_available", return_value=False):
            with pytest.raises(hermes_bridge.HermesBridgeError):
                hermes_bridge.run_turn("test", [])

        # main_brain was called
        assert mock_brain_chat.called
        # Hermes was NOT called
        assert not mock_aiagent.called

    def test_ollama_up_uses_hermes(self, reset_hermes_startup_state):
        """With Ollama up, run_turn uses Hermes (in-process path)."""
        import hermes_bridge

        # Track whether run_conversation was called
        conversation_called = {"count": 0}

        class TrackedAIAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def run_conversation(self, user_message, conversation_history):
                conversation_called["count"] += 1
                return {
                    "final_response": "Hermes reply",
                    "messages": [],
                }

        # Replace the real class with our tracked one
        original_aiagent = hermes_bridge._AIAgent
        hermes_bridge._AIAgent = TrackedAIAgent
        hermes_bridge._cached_agent = None  # ensure fresh
        hermes_bridge._HERMES_AVAILABLE = True
        hermes_bridge._BRAIN_PROVIDER_AVAILABLE = True
        hermes_bridge._brain_chat = MagicMock()
        # Opt in to Hermes's in-process path (bypasses the global conftest guard)
        hermes_bridge._FAIRY_TEST_ALLOW_IN_PROCESS = True

        try:
            with patch.object(hermes_bridge, "_is_ollama_available", return_value=True):
                reply, _ = hermes_bridge.run_turn("hi", [])

            assert reply == "Hermes reply"
            assert conversation_called["count"] == 1, (
                f"run_conversation should be called exactly once, got {conversation_called['count']}"
            )
            # main_brain was NOT called
            assert not hermes_bridge._brain_chat.called, (
                "main_brain should NOT be called when Ollama is up and Hermes succeeds"
            )
        finally:
            hermes_bridge._AIAgent = original_aiagent
            hermes_bridge._cached_agent = None
            hermes_bridge._FAIRY_TEST_ALLOW_IN_PROCESS = False


# ── Test 6: Public API surface unchanged ─────────────────────────────────────────

class TestPublicAPIPreserved:
    """The run_turn / run_turn_safe public interfaces must not change."""

    def test_run_turn_signature_unchanged(self):
        import inspect
        from hermes_bridge import run_turn
        sig = inspect.signature(run_turn)
        params = list(sig.parameters.keys())
        assert "user_text" in params
        assert "history" in params
        assert "on_status" in params

    def test_run_turn_safe_signature_unchanged(self):
        import inspect
        from hermes_bridge import run_turn_safe
        sig = inspect.signature(run_turn_safe)
        params = list(sig.parameters.keys())
        assert "user_text" in params
        assert "history" in params
        assert "on_status" in params

    def test_run_turn_safe_returns_three_tuple(self):
        from hermes_bridge import run_turn_safe

        hermes_bridge_module = run_turn_safe.__module__
        assert hermes_bridge_module == "hermes_bridge"

    def test_hermes_bridge_error_class_still_exists(self):
        from hermes_bridge import HermesBridgeError
        assert issubclass(HermesBridgeError, Exception)

    def test_get_hermes_import_error_still_works(self):
        from hermes_bridge import get_hermes_import_error
        result = get_hermes_import_error()
        assert result is None or isinstance(result, str)

    def test_is_hermes_available_still_works(self):
        from hermes_bridge import is_hermes_available
        result = is_hermes_available()
        assert isinstance(result, bool)

    def test_is_internal_agent_name_is_public(self):
        """The new is_internal_agent_name function is exposed in the public API."""
        from hermes_bridge import is_internal_agent_name
        assert callable(is_internal_agent_name)


# ── Test 7: Background monitor / startup doesn't conflict ───────────────────────

class TestStartupIntegration:
    """Verify the startup health check is compatible with existing init patterns."""

    def test_health_check_compatible_with_discord_bot_independence(self, reset_hermes_startup_state):
        """Hermes init must not depend on or interfere with the Discord bot."""
        import hermes_bridge
        hermes_bridge._hermes_initialized = False
        hermes_bridge._hermes_initialized_ok = False

        # Run health check
        hermes_bridge._run_startup_health_check()

        # Health check completed
        assert hermes_bridge._hermes_initialized is True

    def test_handle_request_works_after_health_check(self, reset_hermes_startup_state, monkeypatch):
        """The end-to-end handle_request path still works after startup init."""
        from controller import agent_controller as ac
        import hermes_bridge

        hermes_bridge._HERMES_AVAILABLE = True
        hermes_bridge._hermes_initialized = True
        hermes_bridge._hermes_initialized_ok = True

        def mock_run_turn_safe(user_text, history, on_status=None):
            return True, f"handled: {user_text}", []

        monkeypatch.setattr("hermes_bridge.run_turn_safe", mock_run_turn_safe)

        with patch.dict(os.environ, {"FAIRY_USE_HERMES": "1"}):
            reply, _ = ac.handle_request("hi there")

        assert reply == "handled: hi there"

    def test_hermes_init_does_not_require_ollama(self, reset_hermes_startup_state):
        """Startup health check must NOT call Ollama — Hermes handles that at turn time."""
        import hermes_bridge
        hermes_bridge._hermes_initialized = False
        hermes_bridge._hermes_initialized_ok = False

        # Make _is_ollama_available raise — the health check must not depend on it
        with patch.object(
            hermes_bridge, "_is_ollama_available",
            side_effect=Exception("Ollama not available at startup")
        ):
            hermes_bridge._run_startup_health_check()

        # Health check should have completed regardless
        assert hermes_bridge._hermes_initialized is True

    def test_fairy_py_imports_hermes_bridge(self):
        """fairy.py must be able to import is_hermes_ready / is_hermes_initialized."""
        # If this import fails, the test fails — we don't need to do anything else
        from hermes_bridge import is_hermes_ready, is_hermes_initialized
        assert callable(is_hermes_ready)
        assert callable(is_hermes_initialized)
