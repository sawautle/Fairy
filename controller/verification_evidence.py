"""
Verification evidence recording — captures what was checked before Fairy
replies so failures are surfaced honestly instead of being silently swallowed.

Every tool execution, capability check, and artifact verification is recorded
here.  ``handle_request`` consults the evidence log on the reply path to
ensure the final answer reflects what actually happened on disk.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_BASE = Path(__file__).resolve().parent.parent
_EVIDENCE_FILE = _BASE / "fairy_verification_evidence.jsonl"

_lock = threading.Lock()

# In-memory buffer for the current turn (cleared at the start of each turn).
_current_evidence: list[dict[str, Any]] = []
_current_turn_id: str | None = None


def start_turn(user_text: str) -> str:
    """Reset the per-turn buffer and return a new turn-id."""
    global _current_evidence, _current_turn_id
    with _lock:
        _current_turn_id = f"{int(time.time())}-{os.getpid()}"
        _current_evidence = []
    record("turn_start", {"user_text_preview": (user_text or "")[:200]})
    return _current_turn_id


def record(event: str, payload: dict[str, Any] | None = None) -> None:
    """Append a verification record to the in-memory buffer and the JSONL file."""
    entry = {
        "turn_id": _current_turn_id,
        "event": event,
        "timestamp": time.time(),
        "payload": payload or {},
    }
    with _lock:
        _current_evidence.append(entry)
    try:
        with open(_EVIDENCE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass


def record_tool_result(fn: str, args: dict, result: Any, verified: bool | None = None) -> None:
    """Record the outcome of a single tool call for accountability."""
    record("tool_result", {
        "tool": fn,
        "args": {k: (str(v)[:200] if not isinstance(v, (int, float, bool, type(None))) else v)
                 for k, v in (args or {}).items()},
        "ok": _extract_ok(result),
        "verified": verified,
        "error": _extract_error(result),
    })


def record_artifact_check(fn: str, path: str, exists: bool, non_empty: bool) -> None:
    """Record a post-execution artifact-verification check."""
    record("artifact_check", {
        "tool": fn,
        "path": path,
        "exists": exists,
        "non_empty": non_empty,
        "passed": bool(exists and non_empty),
    })


def record_capability_check(cap: str, available: bool, detail: str = "") -> None:
    """Record the result of an upfront capability check."""
    record("capability_check", {"capability": cap, "available": available, "detail": detail})


def record_failure(source: str, error: str, context: dict | None = None) -> None:
    """Record a failure that was surfaced to the user (not silently swallowed)."""
    record("failure_surfaced", {"source": source, "error": error, **(context or {})})


def record_reply(reply: str, evidence_summary: dict | None = None) -> None:
    """Record the final reply that was about to be returned to the user."""
    record("reply", {
        "reply_preview": (reply or "")[:500],
        "evidence_summary": evidence_summary or {},
        "evidence_count": len(_current_evidence),
    })


def get_turn_evidence() -> list[dict[str, Any]]:
    """Return a copy of all evidence recorded this turn."""
    with _lock:
        return list(_current_evidence)


def summarize_turn() -> dict[str, Any]:
    """Produce a summary of verification results for the current turn.

    Used by ``handle_request`` to decide whether to append an honesty notice
    to the reply or to surface failures explicitly.
    """
    with _lock:
        ev = list(_current_evidence)
    tools_ok = sum(1 for e in ev if e["event"] == "tool_result" and e["payload"].get("ok"))
    tools_failed = sum(1 for e in ev if e["event"] == "tool_result" and not e["payload"].get("ok"))
    tools_unverified = sum(
        1 for e in ev
        if e["event"] == "tool_result" and e["payload"].get("ok") and e["payload"].get("verified") is False
    )
    artifacts_fail = sum(
        1 for e in ev
        if e["event"] == "artifact_check" and not e["payload"].get("passed")
    )
    capabilities_missing = [
        e["payload"].get("capability") for e in ev
        if e["event"] == "capability_check" and not e["payload"].get("available")
    ]
    failures_surfaced = sum(1 for e in ev if e["event"] == "failure_surfaced")

    return {
        "tools_total": tools_ok + tools_failed,
        "tools_ok": tools_ok,
        "tools_failed": tools_failed,
        "tools_unverified": tools_unverified,
        "artifacts_failed": artifacts_fail,
        "capabilities_missing": capabilities_missing,
        "failures_surfaced": failures_surfaced,
        "has_failures": (tools_failed > 0 or tools_unverified > 0
                         or artifacts_fail > 0 or capabilities_missing),
    }


# ── internal helpers ───────────────────────────────────────────────────


def _extract_ok(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("ok", result.get("status") == "ok"))
    return False


def _extract_error(result: Any) -> str | None:
    if isinstance(result, dict):
        return result.get("error") or result.get("message") or result.get("error_message")
    return None


def clear() -> None:
    """Clear the in-memory buffer (used in tests)."""
    global _current_evidence, _current_turn_id
    with _lock:
        _current_evidence = []
        _current_turn_id = None
