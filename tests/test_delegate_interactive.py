#!/usr/bin/env python3
"""
Tests for delegate_to_claude_code_interactive (controller/claude_code_delegate.py).

RETIRED in v2 handoff redesign. The interactive delegation pipeline
(delegate_to_claude_code_interactive, BackgroundTaskRegistry + live panel)
has been replaced by the direct handoff (_suspend_and_handoff).
The tests that remain are marked as skipped at module level.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

# Make controller/ importable
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

pytestmark = pytest.mark.skip(reason="removed in v2 handoff redesign — interactive delegation pipeline gone")


class TestInteractiveDelegateCancellation:
    """When the architect panel is cancelled, no spawn occurs."""

    def test_cancelled_returns_cancelled_status(self, monkeypatch):
        """If architect returns confirmed=False, status is cancelled."""
        from controller import claude_code_delegate as delegate

        # Mock the architect to return N (cancel)
        monkeypatch.setattr(
            "controller.prompt_architect.run_prompt_architect",
            lambda *args, **kwargs: type(
                "R", (), {"confirmed": False, "composed_prompt": "test prompt"}
            )(),
        )
        result = delegate.delegate_to_claude_code_interactive(
            project_root=os.getcwd(),
            user_intent="fix the bug",
            history=[],
        )
        assert result["status"] == "cancelled"
        assert result["executed"] is False
        assert "no spawn" in result["message"]

    def test_cancelled_does_not_spawn(self, monkeypatch):
        """Cancellation does not call spawn_background_process.

        We verify this indirectly: the result has status=cancelled
        and no task_id (no spawn occurred).
        """
        from controller import claude_code_delegate as delegate

        monkeypatch.setattr(
            "controller.prompt_architect.run_prompt_architect",
            lambda *args, **kwargs: type(
                "R", (), {"confirmed": False, "composed_prompt": "test"}
            )(),
        )
        result = delegate.delegate_to_claude_code_interactive(
            project_root=os.getcwd(),
            user_intent="x",
            history=[],
        )
        # No spawn occurred — no task_id in result
        assert "task_id" not in result
        assert result["status"] == "cancelled"


class TestInteractiveDelegateNoClaude:
    """When claude binary is absent."""

    def test_missing_binary_returns_claude_not_installed(self, monkeypatch):
        """If claude binary is missing, status is claude_not_installed."""
        from controller import claude_code_delegate as delegate

        monkeypatch.setattr(
            "controller.prompt_architect.run_prompt_architect",
            lambda *args, **kwargs: type("R", (), {
                "confirmed": True,
                "composed_prompt": "fix the thing",
            })(),
        )
        # Force _find_claude_binary to return None
        monkeypatch.setattr(delegate, "_find_claude_binary", lambda: None)

        result = delegate.delegate_to_claude_code_interactive(
            project_root=os.getcwd(),
            user_intent="fix the thing",
            history=[],
        )
        assert result["status"] == "claude_not_installed"
        assert result["executed"] is False
        assert "not found" in result["message"].lower()



class TestInteractiveDelegateArchitectError:
    """When the architect itself raises an exception."""

    def test_architect_raises_returns_architect_error(self, monkeypatch):
        """If run_prompt_architect raises, status is architect_error."""
        from controller import claude_code_delegate as delegate

        monkeypatch.setattr(
            "controller.prompt_architect.run_prompt_architect",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced")),
        )
        result = delegate.delegate_to_claude_code_interactive(
            project_root=os.getcwd(),
            user_intent="x",
            history=[],
        )
        assert result["status"] == "architect_error"
        assert result["executed"] is False


class TestInteractiveDelegateSpawnError:
    """When the spawn itself fails after architect confirmed."""

    def test_spawn_error_returns_spawn_error_status(self, monkeypatch):
        """If spawn_background_process raises, we get spawn_error."""
        from controller import claude_code_delegate as delegate

        monkeypatch.setattr(
            "controller.prompt_architect.run_prompt_architect",
            lambda *args, **kwargs: type("R", (), {
                "confirmed": True,
                "composed_prompt": "do it",
            })(),
        )
        # Make _find_claude_binary return a fake path
        monkeypatch.setattr(
            delegate, "_find_claude_binary",
            lambda: "E:/fairy/nonexistent-claude-binary",
        )
        result = delegate.delegate_to_claude_code_interactive(
            project_root=os.getcwd(),
            user_intent="x",
            history=[],
        )
        # Should either be claude_not_installed or spawn_error
        assert result["status"] in ("claude_not_installed", "spawn_error")
        assert result["executed"] is False
        assert "composed_prompt" in result


class TestInteractiveDelegateReturnsDict:
    """Result dict has expected keys."""

    def test_cancelled_result_has_required_keys(self, monkeypatch):
        """The cancelled result has all required dict keys."""
        from controller import claude_code_delegate as delegate

        monkeypatch.setattr(
            "controller.prompt_architect.run_prompt_architect",
            lambda *args, **kwargs: type("R", (), {
                "confirmed": False,
                "composed_prompt": "the prompt",
            })(),
        )
        result = delegate.delegate_to_claude_code_interactive(
            project_root=os.getcwd(),
            user_intent="fix it",
            history=[],
        )
        required_keys = {"executed", "status", "message", "exit_code", "duration_seconds"}
        assert required_keys.issubset(result.keys())

    def test_confirmed_result_has_composed_prompt(self, monkeypatch):
        """A confirmed result includes the composed_prompt key."""
        from controller import claude_code_delegate as delegate

        monkeypatch.setattr(
            "controller.prompt_architect.run_prompt_architect",
            lambda *args, **kwargs: type("R", (), {
                "confirmed": True,
                "composed_prompt": "the composed prompt",
            })(),
        )
        monkeypatch.setattr(
            delegate, "_find_claude_binary",
            lambda: "nonexistent",
        )
        result = delegate.delegate_to_claude_code_interactive(
            project_root=os.getcwd(),
            user_intent="x",
            history=[],
        )
        assert "composed_prompt" in result
        assert result["composed_prompt"] == "the composed prompt"


class TestExistingDelegateFunctions:
    """Existing delegate functions still work after additions."""

    def test_detect_repository_task_still_works(self):
        from controller import claude_code_delegate as delegate
        assert delegate.detect_repository_task("fix this bug in fairy") is True
        assert delegate.detect_repository_task("hello, how are you?") is False
        assert delegate.detect_repository_task("") is False

    def test_classify_delegate_result_still_works(self):
        from controller import claude_code_delegate as delegate
        result = {
            "executed": True,
            "status": "ok",
            "exit_code": 0,
            "duration_seconds": 5.0,
            "stdout": "All good",
            "stderr": "",
        }
        classified = delegate.classify_delegate_result(result, os.getcwd(), "fix bug")
        assert "outcome" in classified
        assert "change_info" in classified

    def test_format_report_still_works(self):
        from controller import claude_code_delegate as delegate
        result = {
            "executed": True,
            "status": "ok",
            "exit_code": 0,
            "duration_seconds": 3.0,
            "stdout": "Done",
        }
        msg = delegate.format_report(result)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_request_permission_still_works(self):
        from controller import claude_code_delegate as delegate
        msg = delegate.request_permission("/tmp", "fix bug")
        assert "repository access" in msg
        assert "fix bug" in msg


class TestModuleLoadsCleanly:
    """The module loads without errors."""

    def test_module_imports_cleanly(self):
        from controller import claude_code_delegate as delegate
        # All existing symbols should be present
        assert hasattr(delegate, "detect_repository_task")
        assert hasattr(delegate, "request_permission")
        assert hasattr(delegate, "delegate_to_claude_code")
        assert hasattr(delegate, "delegate_to_claude_code_interactive")
        assert hasattr(delegate, "classify_delegate_result")
        assert hasattr(delegate, "format_report")
        assert hasattr(delegate, "_find_claude_binary")
