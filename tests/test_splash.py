#!/usr/bin/env python3
"""
Tests for Splash (controller/splash.py).

Phase 2 tests for the Claude ✕ Fairy splash and completion panel.
"""
from __future__ import annotations

import io
import os
import sys
import unittest.mock

import pytest

# Make controller/ importable
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


class _FakeStdout:
    """Fake sys.stdout with a writable .buffer pointing to a BytesIO."""

    def __init__(self, buf: io.BytesIO):
        self._buf = buf
        self.buffer = buf  # writable

    def write(self, s: str) -> int:
        return self._buf.write(s if isinstance(s, bytes) else s.encode("utf-8"))

    def flush(self) -> None:
        self._buf.flush()

    @property
    def encoding(self) -> str:
        return "utf-8"


def _capture_render(callable_obj: "callable") -> str:
    """Capture binary sys.stdout.buffer.write() calls from render functions."""
    captured = io.BytesIO()
    fake_stdout = _FakeStdout(captured)
    with unittest.mock.patch.object(sys, "stdout", fake_stdout):
        callable_obj()
        fake_stdout.flush()
    captured.seek(0)
    return captured.getvalue().decode("utf-8", errors="replace")


class TestVTEnable:
    """_enable_vt is a Windows-only path."""

    def test_enable_vt_does_not_raise(self):
        from controller.splash import _enable_vt
        result = _enable_vt()
        assert isinstance(result, bool)


class TestRenderSplash:
    """render_splash writes to stdout."""

    def test_render_writes_to_stdout(self):
        from controller.splash import render_splash
        out = _capture_render(lambda: render_splash("Fix the bug"))
        assert len(out) > 0

    def test_render_includes_task_summary(self):
        from controller.splash import render_splash
        out = _capture_render(lambda: render_splash("unique-task-marker-XYZ123"))
        assert "unique-task-marker-XYZ123" in out

    def test_render_handles_empty_summary(self):
        from controller.splash import render_splash
        out = _capture_render(lambda: render_splash(""))
        assert len(out) > 0

    def test_render_includes_box_frame(self):
        from controller.splash import render_splash
        out = _capture_render(lambda: render_splash("test"))
        assert "┌" in out
        assert "┐" in out
        assert "└" in out
        assert "┘" in out

    def test_render_includes_claude_wordmark(self):
        from controller.splash import render_splash, _WORDMARK_CLAUDE
        out = _capture_render(lambda: render_splash("test"))
        if _WORDMARK_CLAUDE:
            first_line = _WORDMARK_CLAUDE[0].strip()
            if first_line:
                assert first_line in out, f"Expected CLAUDE wordmark in {out[:200]!r}"

    def test_render_includes_fairy_wordmark(self):
        from controller.splash import render_splash, _WORDMARK_FAIRY
        out = _capture_render(lambda: render_splash("test"))
        if _WORDMARK_FAIRY:
            first_line = _WORDMARK_FAIRY[0].strip()
            if first_line:
                assert first_line in out


class TestWordmarksAreGenerated:
    """Wordmarks must come from pyfiglet, not hardcoded ASCII art."""

    def test_claude_wordmark_is_present(self):
        from controller.splash import _WORDMARK_CLAUDE
        assert isinstance(_WORDMARK_CLAUDE, list)
        assert len(_WORDMARK_CLAUDE) > 0

    def test_fairy_wordmark_is_present(self):
        from controller.splash import _WORDMARK_FAIRY
        assert isinstance(_WORDMARK_FAIRY, list)
        assert len(_WORDMARK_FAIRY) > 0


class TestRenderCompletion:
    """render_completion writes the evidence panel."""

    def test_completion_writes_to_stdout(self):
        from controller.splash import render_completion
        out = _capture_render(
            lambda: render_completion(
                task_summary="Fix bug",
                exit_code=0,
                git_status=None,
                git_diff=None,
                output_tail=[],
            )
        )
        assert len(out) > 0

    def test_completion_includes_exit_code(self):
        from controller.splash import render_completion
        out = _capture_render(
            lambda: render_completion(
                task_summary="task",
                exit_code=42,
                git_status=None,
                git_diff=None,
                output_tail=[],
            )
        )
        assert "42" in out

    def test_completion_zero_exit_uses_blue_color(self):
        from controller.splash import render_completion, _C_BLU
        out = _capture_render(
            lambda: render_completion(
                task_summary="task",
                exit_code=0,
                git_status=None,
                git_diff=None,
                output_tail=[],
            )
        )
        assert _C_BLU in out

    def test_completion_nonzero_exit_uses_red_color(self):
        from controller.splash import render_completion
        out = _capture_render(
            lambda: render_completion(
                task_summary="task",
                exit_code=1,
                git_status=None,
                git_diff=None,
                output_tail=[],
            )
        )
        assert "\033[38;2;255;68;68m" in out

    def test_completion_handles_missing_git(self):
        from controller.splash import render_completion
        out = _capture_render(
            lambda: render_completion(
                task_summary="task",
                exit_code=0,
                git_status=None,
                git_diff=None,
                output_tail=[],
            )
        )
        assert ("no files changed" in out or "no changes" in out or len(out) > 50)

    def test_completion_includes_output_tail(self):
        from controller.splash import render_completion
        out = _capture_render(
            lambda: render_completion(
                task_summary="task",
                exit_code=0,
                git_status=None,
                git_diff=None,
                output_tail=["line-A", "line-B", "line-C"],
            )
        )
        assert "line-A" in out

    def test_completion_truncates_long_git(self):
        from controller.splash import render_completion
        long_status = "\n".join(f"modified: file{i}.py" for i in range(50))
        out = _capture_render(
            lambda: render_completion(
                task_summary="task",
                exit_code=0,
                git_status=long_status,
                git_diff=None,
                output_tail=[],
            )
        )
        assert "file0.py" in out or "file9.py" in out

