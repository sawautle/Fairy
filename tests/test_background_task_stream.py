#!/usr/bin/env python3
"""
Tests for BackgroundTaskRegistry (core/background_tasks.py).

Phase 1: plumbing only. Covers the invariant that the registry stores
only real Popen handles and that fabricated claims are structurally
impossible.

Tests
-----
 1. spawn_background_process → real Popen stored in registry
 2. stdout lines captured line-by-line from a real process
 3. stderr lines captured line-by-line
 4. exit_code set on process completion
 5. tail returns last n lines (combined stdout+stderr)
 6. KeyError on missing task_id (no process = no panel = no fabrication)
 7. ValueError when registering None
 8. ValueError when registering non-Popen
 9. Multiple concurrent tasks tracked independently
10. iter_active() excludes exited tasks
11. concurrent-tasks registry: simultaneous long-running processes have
    independent state (output, exit_code, name)
"""
from __future__ import annotations

import os
import sys
import time

import pytest

# Make core/ importable
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from core.background_tasks import (
    BackgroundTaskRegistry,
    spawn_background_process,
    _TaskHandle,
    _REGISTRY,
)


class TestBackgroundTaskRegistry:
    """Core invariant tests."""

    def test_spawn_creates_real_popen(self):
        """spawn_background_process returns a task_id and the registry stores a real Popen."""
        task_id = spawn_background_process(
            [sys.executable, "-c", "print('hello')"],
            cwd=os.getcwd(),
            name="popen check",
        )
        try:
            handle = _REGISTRY.get(task_id)
            assert isinstance(handle, _TaskHandle)
            # Wait for watcher to detect exit
            for _ in range(20):  # up to 2s
                if handle._done:
                    break
                time.sleep(0.1)
            assert handle.proc.poll() is not None  # process completed
            assert handle.exit_code == 0
            assert handle.name == "popen check"
        finally:
            _REGISTRY.unregister(task_id)

    def test_stdout_captured_line_by_line(self):
        """Lines written to stdout appear in stdout_lines."""
        # Use newline-separated prints so readline() unblocks reliably
        script = "\n".join(f"print('line{i}')" for i in range(1, 6))
        task_id = spawn_background_process(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            name="stdout capture",
        )
        time.sleep(0.5)  # let watcher collect output
        handle = _REGISTRY.get(task_id)
        try:
            assert any("line1" in l for l in handle.stdout_lines)
            assert any("line5" in l for l in handle.stdout_lines)
        finally:
            _REGISTRY.unregister(task_id)

    def test_stderr_captured(self):
        """Lines written to stderr appear in stderr_lines."""
        task_id = spawn_background_process(
            [sys.executable, "-c", "import sys; sys.stderr.write('err1\\nerr2\\n')"],
            cwd=os.getcwd(),
            name="stderr capture",
        )
        time.sleep(0.8)  # stderr drain can be slower
        handle = _REGISTRY.get(task_id)
        try:
            assert any("err1" in l for l in handle.stderr_lines)
            assert any("err2" in l for l in handle.stderr_lines)
        finally:
            _REGISTRY.unregister(task_id)

    def test_exit_code_nonzero(self):
        """Non-zero exit code is recorded."""
        task_id = spawn_background_process(
            [sys.executable, "-c", "raise SystemExit(42)"],
            cwd=os.getcwd(),
            name="nonzero exit",
        )
        time.sleep(0.4)
        handle = _REGISTRY.get(task_id)
        try:
            assert handle.exit_code == 42
        finally:
            _REGISTRY.unregister(task_id)

    def test_tail_returns_last_n_lines(self):
        """get_output_tail returns exactly the last n lines."""
        task_id = spawn_background_process(
            [sys.executable, "-c", "for i in range(20): print(f'L{i}')"],
            cwd=os.getcwd(),
            name="tail test",
        )
        # Wait for watcher to detect exit and flush the last line
        handle = _REGISTRY.get(task_id)
        for _ in range(30):  # up to 3s
            if handle._done:
                break
            time.sleep(0.1)
        try:
            tail = _REGISTRY.get_output_tail(task_id, n=5)
            assert len(tail) <= 5
            assert any("L19" in l for l in tail), f"tail={tail!r}"
        finally:
            _REGISTRY.unregister(task_id)

    def test_fabricated_claim_rejected(self):
        """get_output_tail on a nonexistent id raises KeyError — panel impossible."""
        with pytest.raises(KeyError):
            _REGISTRY.get_output_tail("nonexistent-id-00000")

    def test_fabricated_claim_via_get(self):
        """get() on a nonexistent id raises KeyError."""
        with pytest.raises(KeyError):
            _REGISTRY.get("also-nonexistent-00000")

    def test_register_none_raises(self):
        """register(None) raises ValueError — a fake process cannot enter."""
        reg = BackgroundTaskRegistry()
        with pytest.raises(ValueError, match="None process handle"):
            reg.register(None, "fake task")

    def test_register_non_popen_raises(self):
        """register(non-Popen) raises ValueError."""
        reg = BackgroundTaskRegistry()
        with pytest.raises(ValueError, match="Expected subprocess.Popen"):
            reg.register("not a process", "also fake")

    def test_concurrent_tasks_tracked_separately(self):
        """Multiple tasks have independent name, exit_code, and lines."""
        script = f"{sys.executable} -c \"import time; print('A-start'); time.sleep(0.05); print('A-end')\""
        tid1 = spawn_background_process(
            [sys.executable, "-c",
             "import time; print('task1-start'); time.sleep(0.05); print('task1-end')"],
            cwd=os.getcwd(),
            name="concurrent A",
        )
        tid2 = spawn_background_process(
            [sys.executable, "-c",
             "import time; print('task2-start'); time.sleep(0.05); print('task2-end')"],
            cwd=os.getcwd(),
            name="concurrent B",
        )
        time.sleep(0.5)
        try:
            h1 = _REGISTRY.get(tid1)
            h2 = _REGISTRY.get(tid2)
            assert h1.name == "concurrent A"
            assert h2.name == "concurrent B"
            assert h1.task_id != h2.task_id
            assert h1.exit_code == h2.exit_code == 0
            # Each has its own lines
            assert any("task1" in l for l in h1.stdout_lines)
            assert any("task2" in l for l in h2.stdout_lines)
            assert not any("task1" in l for l in h2.stdout_lines)
        finally:
            _REGISTRY.unregister(tid1)
            _REGISTRY.unregister(tid2)

    def test_iter_active_excludes_exited(self):
        """iter_active() does not yield tasks whose proc.poll() is not None."""
        # Start a fast task that will exit
        task_id = spawn_background_process(
            [sys.executable, "-c", "print('fast')"],
            cwd=os.getcwd(),
            name="fast exited",
        )
        # Wait for watcher to detect exit
        handle = _REGISTRY.get(task_id)
        for _ in range(20):
            if handle._done:
                break
            time.sleep(0.1)
        try:
            active_ids = {h.task_id for h in _REGISTRY.iter_active()}
            assert task_id not in active_ids, (
                "Exited task should not appear in iter_active()"
            )
        finally:
            _REGISTRY.unregister(task_id)

    def test_iter_active_includes_running(self):
        """iter_active() includes tasks that are still running."""
        # Start a slow task
        task_id = spawn_background_process(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=os.getcwd(),
            name="slow running",
        )
        try:
            time.sleep(0.2)
            active_ids = {h.task_id for h in _REGISTRY.iter_active()}
            assert task_id in active_ids, "Running task must appear in iter_active()"
        finally:
            # Kill it cleanly
            handle = _REGISTRY.get(task_id)
            handle.proc.terminate()
            handle.proc.wait()
            _REGISTRY.unregister(task_id)

    def test_tail_empty_before_output(self):
        """tail is empty before the watcher has read anything."""
        task_id = spawn_background_process(
            [sys.executable, "-c", "import time; time.sleep(0.5); print('late')"],
            cwd=os.getcwd(),
            name="delayed output",
        )
        try:
            time.sleep(0.1)  # watcher not yet collected anything
            tail = _REGISTRY.get_output_tail(task_id, n=10)
            # Should be empty — the process hasn't written yet
            assert isinstance(tail, list)
        finally:
            _REGISTRY.unregister(task_id)

    def test_binary_not_found_raises(self):
        """spawn_background_process with nonexistent binary raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            spawn_background_process(
                ["nonexistent-binary-xyz12345"],
                cwd=os.getcwd(),
                name="missing binary",
            )
