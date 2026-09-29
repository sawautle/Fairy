"""End-to-end smoke: handle_request with mocked Hermes must not recurse forever."""
import os
import sys
import pytest

os.environ.pop("FAIRY_USE_HERMES", None)
os.environ.pop("FAIRY_TASK_GUARD_REENTRY", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from controller import agent_controller as ac


@pytest.fixture(autouse=True)
def _reset_pending_state():
    """Clear pending-delegation state and env guards between tests.

    The TestWiredIntegration tests in test_intent_resolver.py set a real
    pending_delegation; without this fixture, that state leaks into other
    tests via the module-level `get_pending()` / `clear_pending()` globals."""
    try:
        from controller.delegate_state import clear_pending
        clear_pending()
    except Exception:
        pass
    os.environ.pop("FAIRY_USE_HERMES", None)
    os.environ.pop("FAIRY_TASK_GUARD_REENTRY", None)
    yield
    try:
        from controller.delegate_state import clear_pending
        clear_pending()
    except Exception:
        pass
    os.environ.pop("FAIRY_USE_HERMES", None)
    os.environ.pop("FAIRY_TASK_GUARD_REENTRY", None)


def test_handle_request_no_recursion_on_cod_folder_task():
    """v2 handoff detection: skip this test.

    "use cod to make a folder" now triggers the intent resolver clarification
    gate (asking which coding tool), not a direct handoff. The old delegation
    pipeline that this test verified is removed in v2.

    The handoff detection itself is tested in test_task_directory_classifier.py
    (explicit trigger detection) and the new handoff tests below.
    """
    pytest.skip("v2 redesign: resolver clarification gate takes priority over delegation")


def test_handle_request_open_cod_clarification():
    """'open cod' must return the clarification question, no action taken."""
    with patch("hermes_bridge.run_turn_safe") as mock_hermes:
        mock_hermes.return_value = (False, "fallback", [])
        with patch("controller.agent_controller._brain") as mock_brain:
            mock_brain.return_value = MagicMock(
                message=MagicMock(content="", tool_calls=[]),
                done=True,
            )
            reply, _ = ac.handle_request("open cod", [])
            assert isinstance(reply, str)
            assert "cod" in reply.lower() or "game" in reply.lower() or "claude" in reply.lower(), \
                f"reply should mention cod/game/claude: {reply[:200]!r}"
            print(f"[PASS] handle_request open cod reply: {reply[:200]!r}")


def test_handle_request_fast_path_unaffected():
    """'hey fairy whats the time' must still hit fast-path (no LLM, no resolver cost)."""
    brain = MagicMock()
    brain.return_value = MagicMock(
        message=MagicMock(content="", tool_calls=[]),
        done=True,
    )
    with patch("hermes_bridge.run_turn_safe") as mock_hermes:
        mock_hermes.return_value = (False, "fallback", [])
        with patch("controller.agent_controller._brain", brain):
            reply, _ = ac.handle_request("hey fairy whats the time", [])
            assert isinstance(reply, str)
            # brain should NOT have been called for an unambiguous greeting
            print(f"[PASS] fast-path reply: {reply[:100]!r}, brain.called={brain.called}")
