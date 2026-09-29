#!/usr/bin/env python3
"""
Global test fixtures for the fairy test suite.

The most important fixture: _block_all_network (autouse=True) patches
urllib.request.urlopen with a fake response so no test can accidentally
hit a real HTTP endpoint. Tests that need specific HTTP behaviour do their
own patch.object() context manager, which overrides this fixture's patch.

Design contract:
  1. Every test is isolated from the real network by default.
  2. A test opts into real networking via @pytest.mark.requires_network.
  3. A test opts into custom HTTP behaviour via its own
     `with patch.object(urllib.request, "urlopen", ...)` block, which
     overrides the autouse fixture.
"""
from __future__ import annotations

import json
import sys
import urllib.request

import pytest

# A minimal valid OpenRouter /chat/completions success response.
_FAKE_OR_BODY = json.dumps({
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [],
        },
        "finish_reason": "stop",
    }],
    "created": 0,
}).encode()


class _FakeResponse:
    """Minimal context-manager fake matching urllib's response interface."""

    def __init__(self, body: bytes = b"{}"):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


def _default_fake_urlopen(req, timeout=None):
    """Default fake: always returns a valid OR success response."""
    return _FakeResponse(body=_FAKE_OR_BODY)


def pytest_configure(config):
    """Register custom marks so pytest doesn't warn about unknown markers.

    The autouse `_block_all_network` fixture reads this marker to let a test
    opt out of the global urlopen patch.
    """
    config.addinivalue_line("markers", "requires_network: test performs real network I/O")


@pytest.fixture(autouse=True)
def _block_all_network(request, monkeypatch):
    """Patch urllib.request.urlopen globally for every test.

    This fixture installs a no-op fake urlopen before the test runs so that
    any code that makes an HTTP call (including the new resilience layer)
    gets a deterministic fake response instead of hitting a real server.

    Test-specific overrides:
      - Tests that use `with patch.object(urllib.request, "urlopen", ...)` are
        NOT affected — the context manager patches the attribute *after* this
        fixture sets it up, so the test's mock wins.
      - Tests that need real networking must be marked
        `@pytest.mark.requires_network` and will skip the patch.

    Strategy:
      - urlopen is patched to a fake that returns a valid OR success response.
      - _run_turn_in_process is replaced so Hermes's in-process agent can never
        make its own (non-urllib) Ollama calls and hang the test. Tests that
        route through hermes_bridge get a HermesBridgeError → main_brain fallback.
      - is_ollama_available is NOT patched globally (its own tests verify caching).
    """
    # Allow opt-out via @pytest.mark.requires_network
    if "requires_network" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    # Always patch urlopen — tests doing their own patch.object override us
    _orig_urlopen = urllib.request.urlopen
    monkeypatch.setattr(urllib.request, "urlopen", _default_fake_urlopen)

    # Also reset both health caches so availability checks always start fresh.
    try:
        from controller import main_brain as _mb
        with _mb._health_cache["lock"]:
            _mb._health_cache["available"] = False
            _mb._health_cache["ts"] = 0.0
    except Exception:
        pass
    try:
        from controller import main_brain_resilience as _res
        with _res._health_lock:
            _res._openrouter_health_cache["available"] = False
            _res._openrouter_health_cache["ts"] = 0.0
    except Exception:
        pass

    # Block Hermes's in-process agent and the subprocess retry from running.
    #
    # Strategy for _run_turn_in_process:
    #   - If _AIAgent is a MagicMock (test has mocked Hermes), call the real
    #     function so the mock is exercised.
    #   - If a test has opted in via hermes_bridge._FAIRY_TEST_ALLOW_IN_PROCESS
    #     = True (e.g. to install a custom test class), call the real function.
    #   - Otherwise, raise HermesBridgeError("unavailable") so run_turn()
    #     catches it and falls through to main_brain.chat().
    #     _is_hermes_provider_error() returns True for "unavailable".
    #
    # Strategy for _run_turn_subprocess:
    #   - Always raise HermesBridgeError("unavailable"). We never want to
    #     spawn a real Python subprocess in unit tests.
    #
    # We do NOT patch _quick_ollama_check globally — its own tests need
    # the real function to verify caching/calling behavior.
    try:
        import hermes_bridge as _hb  # E:\fairy\hermes_bridge.py on sys.path
        from unittest.mock import MagicMock

        # Save the real functions so we can delegate to them when needed.
        _real_run_in_process = _hb._run_turn_in_process
        _real_run_subprocess = _hb._run_turn_subprocess

        def _safe_run_in_process(user_text, hermes_history):
            """Smart stub: defer to the real path if the test opted in."""
            if isinstance(getattr(_hb, "_AIAgent", None), MagicMock):
                return _real_run_in_process(user_text, hermes_history)
            if getattr(_hb, "_FAIRY_TEST_ALLOW_IN_PROCESS", False):
                return _real_run_in_process(user_text, hermes_history)
            raise _hb.HermesBridgeError(
                "Hermes in-process agent unavailable in unit test"
            )

        def _safe_run_subprocess(*args, **kwargs):
            raise _hb.HermesBridgeError(
                "Hermes subprocess unavailable in unit test"
            )

        monkeypatch.setattr(_hb, "_run_turn_in_process", _safe_run_in_process)
        monkeypatch.setattr(_hb, "_run_turn_subprocess", _safe_run_subprocess)
    except Exception:
        pass

    yield

    urllib.request.urlopen = _orig_urlopen
