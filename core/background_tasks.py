"""
Background task registry — spawns Popen processes and tracks their
stdout/stderr line-by-line for live rendering in Fairy's TUI.

Key invariant
-------------
The registry stores ONLY REAL process handles. A fabricated "running"
claim with no Popen raises KeyError from get_task_output_tail(). The
TUI code wraps this; no panel is possible without an actual subprocess.

Phase 1 plumbing. Phase 2 (Claude Code splash + prompt architect) builds
on this — it does not replace it.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class _TaskHandle:
    """Internal record of a live (or recently completed) background task."""
    task_id: str
    name: str
    proc: subprocess.Popen
    stdout_lines: list[str] = field(default_factory=list)
    stderr_lines: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    exit_code: Optional[int] = None
    _done: bool = False
    # Lock protecting exit_code — needed because the watcher thread and the
    # timeout watchdog thread both write to it concurrently. Writers must
    # acquire the lock; readers can read directly (they see either the
    # old or new value, both valid).
    _lock: threading.Lock = field(default_factory=threading.Lock)


class BackgroundTaskRegistry:
    """Thread-safe process registry. Singleton (one instance per process)."""

    def __init__(self) -> None:
        self._tasks: dict[str, _TaskHandle] = {}
        self._lock = threading.Lock()

    def register(self, proc: subprocess.Popen, name: str) -> str:
        """Register a real Popen process. Raises ValueError if proc is None.

        This is the ONLY way to get a task_id. There is no `register_fake()`
        or text-only claim. If you have no Popen, you have no panel.
        """
        if proc is None:
            raise ValueError("Cannot register a None process handle")
        if not isinstance(proc, subprocess.Popen):
            raise ValueError(
                f"Expected subprocess.Popen, got {type(proc).__name__}"
            )
        task_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._tasks[task_id] = _TaskHandle(
                task_id=task_id, name=name, proc=proc
            )
        self._start_watcher(task_id)
        return task_id

    def _start_watcher(self, task_id: str) -> None:
        """Spawn a daemon thread that reads stdout and records exit code."""
        def watcher() -> None:
            with self._lock:
                task = self._tasks.get(task_id)
            if task is None:
                return
            proc = task.proc

            def _drain_stream(stream, target_attr: str) -> None:
                """Read all currently-available lines from `stream`."""
                if stream is None:
                    return
                while True:
                    try:
                        raw = stream.readline()
                    except Exception:
                        return
                    if not raw:
                        return
                    try:
                        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    except Exception:
                        line = repr(raw)
                    with self._lock:
                        if task_id in self._tasks:
                            getattr(self._tasks[task_id], target_attr).append(line)

            while True:
                # Drain stdout until the pipe would block
                _drain_stream(proc.stdout, "stdout_lines")
                # Drain stderr
                _drain_stream(proc.stderr, "stderr_lines")
                # Check exit
                if proc.poll() is not None:
                    # Process exited. Drain any final lines that may have
                    # arrived between the previous readline and the close.
                    _drain_stream(proc.stdout, "stdout_lines")
                    _drain_stream(proc.stderr, "stderr_lines")
                    with self._lock:
                        if task_id in self._tasks:
                            handle = self._tasks[task_id]
                            # Don't overwrite a sentinel exit code set by
                            # the timeout watchdog (e.g. 124 = killed for timeout).
                            with handle._lock:
                                if not handle._done:
                                    handle.exit_code = proc.returncode
                                    handle._done = True
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
                    try:
                        proc.stderr.close()
                    except Exception:
                        pass
                    break
                time.sleep(0.05)

        threading.Thread(target=watcher, daemon=True).start()

    def get(self, task_id: str) -> _TaskHandle:
        """Return the handle for task_id. Raises KeyError if not registered."""
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"No background task with id: {task_id!r}")
            return self._tasks[task_id]

    def iter_active(self) -> Iterator[_TaskHandle]:
        """Yield handles whose proc.poll() is None (still running)."""
        with self._lock:
            snapshot = list(self._tasks.values())
        return iter([h for h in snapshot if h.proc.poll() is None])

    def unregister(self, task_id: str) -> None:
        """Remove a task from the registry. No-op if not present."""
        with self._lock:
            self._tasks.pop(task_id, None)

    def get_output_tail(self, task_id: str, n: int = 10) -> list[str]:
        """Return the last n lines of combined stdout+stderr.

        Raises KeyError if task_id is not registered. This is intentional:
        a nonexistent task means no real process exists, and no panel can
        be rendered. Callers MUST handle the KeyError.
        """
        handle = self.get(task_id)
        with self._lock:
            combined = list(handle.stdout_lines) + list(handle.stderr_lines)
        return combined[-n:]


# Module-level singleton — one registry per Python process.
_REGISTRY = BackgroundTaskRegistry()


def spawn_background_process(
    cmd: list[str],
    cwd: str,
    name: str,
    env: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> str:
    """Spawn a subprocess with piped stdout/stderr and start its watcher.

    Returns a task_id (8-char UUID prefix). Raises FileNotFoundError if the
    binary is missing, OSError on other spawn failures, ValueError on
    internal misuse. The returned task_id is the only handle the caller
    needs to retrieve output later.

    If `timeout` is provided (seconds), a watchdog thread will kill the
    process if it runs longer than that. The exit code will be set to a
    sentinel (-1) and the process will be reaped. This prevents the
    "infinite spinner" failure mode when a delegated process hangs.
    """
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        startupinfo=startupinfo,
        creationflags=(
            subprocess.CREATE_NO_WINDOW
            | subprocess.CREATE_NEW_PROCESS_GROUP
        ),
    )
    task_id = _REGISTRY.register(proc, name)

    # Optional watchdog: kill the process if it exceeds `timeout`.
    if timeout is not None and timeout > 0:
        _start_timeout_watchdog(task_id, proc, timeout)

    return task_id


def _start_timeout_watchdog(task_id: str, proc: subprocess.Popen, timeout: float) -> None:
    """Start a daemon thread that kills the process if it exceeds timeout.

    When the timeout fires the process is killed with SIGKILL (or TerminateProcess
    on Windows). Exit code 124 is set as the sentinel value — this is the
    standard exit code used by the Unix `timeout` command for killed processes,
    making it recognizable in any tooling that reads exit codes.
    """
    TIMEOUT_EXIT_CODE = 124

    def _watchdog():
        # Wait for up to `timeout` seconds for the process to exit naturally.
        # subprocess.run() already has its own timeout handling; this watchdog
        # is a safety net for spawn_background_process() callers that don't use
        # subprocess.run().
        deadline = time.time() + timeout
        remaining = timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                # Process exited naturally before our deadline — nothing to do.
                return
            time.sleep(min(0.25, remaining))
            remaining = deadline - time.time()

        # Timeout expired and process is still running — kill it.
        try:
            proc.kill()
        except Exception:
            pass

        # Wait up to 5s for the kill to take effect, then record the sentinel.
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Process still didn't exit — force terminate (Windows may need this).
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except Exception:
                pass

        # Mark the handle as timed-out. Use the handle's own lock so the
        # watcher can't race us and overwrite with the kill's exit code (1 on
        # Windows). The watcher's "if not handle._done" guard ensures our
        # sentinel value (124) wins.
        try:
            handle = _REGISTRY.get(task_id)
            with handle._lock:
                handle.exit_code = TIMEOUT_EXIT_CODE
                handle._done = True
        except KeyError:
            pass

    threading.Thread(target=_watchdog, daemon=True).start()
