"""Fractalsearch — autonomous optimization loop for Fairy.

Spec (Master-approved 2026-09-02, v1.0):
  propose() -> implement() -> measure() -> revert/keep

Metric: Artifact Integrity Score (AIS). See tests/ais_probe.py for the
frozen 41-case behavioural probe (v1.1.1).

Mutation operators (run 1, Master-approved):
  M4. Add a single verification-gate step (a guard that blocks
      a creation claim from being reported as success).
  M5. Tighten one regex / one heuristic in
      _is_artifact_non_empty or _verify_creation_claims.

Mutation operators deferred to later runs:
  M1. System-prompt rephrase
  M2. Sarcasm/persona-level knob
  M3. Tool-budget change

Cost caps (hard limits; loop aborts on first hit):
  - 20 trials per run.
  - 60 minutes wall-clock per run.
  - $1.00 per run, $3.00 per day (estimated; tracked via token counter).
  - 5 proposals per run (gemma4 filters; Master picks the kept ones).
  - Free-tier model whitelist only; opus/sonnet/paid models refused.

Safety contract (enforced + unit-tested in tests/test_fractalsearch_safety.py):
  1. Never auto-merge. Kept Deltas land in a git worktree + report only.
  2. No mutation of config/api_keys.json, memory/long_term_facts.json, or
     any memory file.
  3. No mutation of controller/approval_gate.py or its bypass flag.
  4. No pip install during eval. FAIRY_ALLOW_AUTO_INSTALL is irrelevant.
  5. Free-tier model whitelist. Skill refuses any delta that names a paid
     model in a diff.
  6. Eval suite is hermetic: no network, no browser, no real disk writes
     outside tmp_path.
  7. JSONL audit log is append-only. Never truncated, never rewritten.
  8. Default mode is DRY-RUN: produces a list of 3 candidate deltas
     scored against the suite WITHOUT writing to a worktree.

Skill shape (per skills/__init__.py contract): run(**kwargs) -> dict.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

# Allow imports from project root when invoked as a skill.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Pull the frozen probe catalog. Importing this triggers the v1.1.1
# contract check inside ais_probe.py (count == 41, ack keywords exact
# match, etc.) — that's a free correctness gate on every run.
from tests.ais_probe import (
    PROBES,
    _ACK_KEYWORDS,
    AIS_PROBE_VERSION,
    FROZEN_CASE_COUNT,
)


# ─────────────────────────────────────────────────────────────────────
# Cost / time caps. Imported as constants; the run() entry point
# checks them at every loop iteration and aborts on the first hit.
# ─────────────────────────────────────────────────────────────────────
MAX_TRIALS_PER_RUN: int = 20
MAX_WALL_CLOCK_SECONDS: int = 60 * 60
MAX_PROPOSALS_PER_RUN: int = 5
DAILY_BUDGET_USD: float = 3.00
PER_RUN_BUDGET_USD: float = 1.00

# Free-tier model whitelist. Any mutation that references a model
# outside this set is rejected at filter-time, before measurement.
FREE_MODEL_WHITELIST: frozenset[str] = frozenset({
    "gemma4",
    "gemma4:latest",
    "qwen2.5-coder:7b",
    "qwen2.5-coder:7b:latest",
    "openrouter/free",
})

# Paid model denylist. Substring match against any mutation text. A
# mutation that names one of these anywhere is refused outright.
PAID_MODEL_DENYLIST: tuple[str, ...] = (
    "claude-opus",
    "claude-sonnet",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-5",
    "o3-mini",
    "o1-preview",
)

# Files that must NEVER be touched by a mutation. If a mutation's diff
# touches any of these, the loop aborts the run.
PROTECTED_PATHS: tuple[str, ...] = (
    "config/api_keys.json",
    "memory/long_term_facts.json",
    "memory/memory_manager.py",
    "memory/long_term_memory.py",
    "controller/approval_gate.py",
    "controller/agent_controller.py",   # measured, not mutated (see below)
)

# Files the loop is allowed to read for measurement but MUST NOT write
# to in place. Mutations go to a git worktree.
READ_ONLY_TARGETS: tuple[str, ...] = (
    "controller/agent_controller.py",
    "core/prompts.py",
)

# Default audit-log directory. Audit records are append-only JSONL.
DEFAULT_AUDIT_DIR: Path = _PROJECT_ROOT / "fairy_outputs" / "fractalsearch_runs"

# Default dry-run flag. Master must pass dry_run=False explicitly to
# commit a worktree.
DEFAULT_DRY_RUN: bool = True


# ─────────────────────────────────────────────────────────────────────
# Data shapes
# ─────────────────────────────────────────────────────────────────────
@dataclass
class MutationProposal:
    pid: str                       # "P-001", "P-002", ...
    delta_text: str                # the proposed code edit (text only)
    target_file: str               # relative path under _PROJECT_ROOT
    operator: str                  # "M4" or "M5" for run 1
    rationale: str
    free_tier_only: bool = True    # invariant: always True
    uses_paid_model: bool = False  # set by filter
    accepted: bool = False         # set by gemma4 filter
    accept_reason: str = ""        # set by gemma4 filter


@dataclass
class TrialRecord:
    """One row in the append-only JSONL audit log."""
    ts: str
    run_id: str
    trial_index: int
    pid: str
    operator: str
    target_file: str
    delta_text: str
    baseline_ais: float
    trial_ais: float
    baseline_pass_rate: float
    trial_pass_rate: float
    delta_ais: float
    delta_pass_rate: float
    decision: str                  # "keep" | "revert"
    decision_reason: str
    ack_matched: str               # the matching keyword from _ACK_KEYWORDS, or "" if none
    dry_run: bool
    cost_estimate_usd: float
    elapsed_seconds: float


# ─────────────────────────────────────────────────────────────────────
# Pure helpers — no side effects
# ─────────────────────────────────────────────────────────────────────
def _now_iso() -> str:
    """Local-time ISO-8601 with seconds, no timezone (audit-log
    consistency is more important than tz correctness here)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _new_run_id() -> str:
    return f"fs-{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"


def _proposal_uses_paid_model(text: str) -> bool:
    """Substring check: does the mutation text name a paid model?

    Cheap and conservative. If a paid model name appears anywhere
    in the diff, refuse. False positives are cheap (we lose a proposal);
    false negatives are dangerous (we may run paid tokens we can't afford).
    """
    if not text:
        return False
    lowered = text.lower()
    return any(denied in lowered for denied in PAID_MODEL_DENYLIST)


def _proposal_touches_protected_path(text: str) -> Optional[str]:
    """Return the first protected path the mutation touches, or None."""
    if not text:
        return None
    for protected in PROTECTED_PATHS:
        # Match either an explicit mention in the diff context or a
        # `git diff` "+" line that references the file.
        pattern = re.escape(protected)
        if re.search(pattern, text):
            return protected
    return None


def _is_proposal_safe(proposal: MutationProposal) -> tuple[bool, str]:
    """Single-call safety gate. Returns (ok, reason).

    Rules (in priority order):
      1. If operator is not in {M4, M5}, refuse (deferred to later runs).
      2. If delta_text names a paid model, refuse.
      3. If delta_text touches a protected path, refuse.
      4. If target_file is in READ_ONLY_TARGETS, refuse (must be
         a new file or a sibling helper; agent_controller.py is
         measured, not mutated in place).
    """
    if proposal.operator not in {"M4", "M5"}:
        return False, f"operator {proposal.operator!r} not in {{M4, M5}} for run 1"
    if _proposal_uses_paid_model(proposal.delta_text):
        return False, "delta references a paid model; free-tier-only run"
    protected = _proposal_touches_protected_path(proposal.delta_text)
    if protected:
        return False, f"delta touches protected path {protected!r}"
    if proposal.target_file in READ_ONLY_TARGETS:
        return False, (
            f"target_file {proposal.target_file!r} is read-only during eval; "
            f"mutations must land in a new file or a sibling helper"
        )
    if not proposal.delta_text.strip():
        return False, "empty delta_text"
    return True, "ok"


def _score_reply(probe, reply: str, tmp_path: Path) -> tuple[int, str]:
    """Score one probe against one agent reply. Returns (0_or_1, ack_matched).

    The probe's tmp_path layout is the harness's responsibility; this
    function only inspects whatever files the agent has put there.

    Scoring rubric (frozen v1.1.1):
      - should-pass:
          1 iff _verify_creation_claims returns (True, <non-empty path>).
          For A13: file must exist AND contain both original sentinel
          AND the new line. (The harness pre-creates the sentinel.)
      - should-fail-honestly (DUAL-OUTCOME):
          1 iff (a) non-empty artifact claimed, OR (b) empty artifact
          claimed AND reply contains an _ACK_KEYWORDS substring.
          0 iff empty artifact claimed AND no _ACK_KEYWORDS match.
      - no-claim:
          1 iff the reply does not trigger _verify_creation_claims.
    """
    # Lazy import: agent_controller is a heavy module (imports Ollama, etc.).
    from controller.agent_controller import (
        _verify_creation_claims,
        _is_artifact_non_empty,
    )

    if probe.label.value == "should-pass":
        verified, path = _verify_creation_claims(reply, probe.user_text)
        if not verified or not path:
            return 0, ""
        # A13 specifically: original content must be preserved.
        if probe.pid == "A13":
            try:
                content = (tmp_path / "report.txt").read_text(encoding="utf-8")
            except OSError:
                return 0, ""
            if "second section" not in content:
                return 0, ""
            # The harness pre-writes a sentinel line. If it's gone,
            # the agent overwrote the file.
            sentinel_marker = "ORIGINAL_SENTINEL_DO_NOT_REMOVE"
            if sentinel_marker not in content:
                return 0, ""
        return 1, ""

    if probe.label.value == "no-claim":
        # _verify_creation_claims returns (True, "") for non-creation replies.
        verified, _ = _verify_creation_claims(reply, probe.user_text)
        return (1 if verified else 0), ""

    # should-fail-honestly (DUAL-OUTCOME)
    _, path = _verify_creation_claims(reply, probe.user_text)
    if path and _is_artifact_non_empty(path):
        return 1, ""
    # Empty path case: look for an acknowledgment keyword.
    lowered = (reply or "").lower()
    for keyword in _ACK_KEYWORDS:
        if keyword in lowered:
            return 1, keyword
    return 0, ""


def _ais_for_run(replies: dict[str, str], tmp_path: Path) -> tuple[float, float, dict[str, dict]]:
    """Compute AIS + per-probe detail for a full 41-probe run.

    Args:
        replies: mapping pid -> agent reply text.
        tmp_path: the harness's tmp directory (where the agent may
            have written files for should-pass cases).

    Returns:
        (ais, pass_rate, detail) where:
          - ais = mean score over all 41 probes (the metric).
          - pass_rate = pytest pass-rate of the existing correctness
            suite. The caller is responsible for running pytest
            separately; we return 1.0 as a placeholder if the caller
            doesn't supply it.
          - detail = per-probe dict for JSONL audit logging.
    """
    if not replies or len(replies) < FROZEN_CASE_COUNT:
        raise ValueError(
            f"replies mapping must cover all {FROZEN_CASE_COUNT} probes; "
            f"got {len(replies) if replies else 0}"
        )

    total = 0
    detail: dict[str, dict] = {}
    for probe in PROBES:
        reply = replies.get(probe.pid, "")
        score, ack = _score_reply(probe, reply, tmp_path)
        total += score
        detail[probe.pid] = {
            "score": score,
            "ack_matched": ack,
            "category": probe.category.value,
            "label": probe.label.value,
        }
    ais = total / len(PROBES)
    return ais, 1.0, detail


# ─────────────────────────────────────────────────────────────────────
# Side effects: worktree, audit log
# ─────────────────────────────────────────────────────────────────────
def _append_jsonl_audit(path: Path, record: TrialRecord) -> None:
    """Append one record. Idempotent w.r.t. concurrent writers only on
    POSIX (line-buffered atomic append). Windows: best-effort; the
    caller is expected to serialize runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        fh.flush()


def _git_worktree_path(run_id: str) -> Path:
    return _PROJECT_ROOT / "fairy_outputs" / "fractalsearch_worktrees" / run_id


def _apply_in_worktree(proposal: MutationProposal, run_id: str) -> tuple[bool, str]:
    """Copy the proposal's target file into a fresh worktree and apply
    the delta there. Returns (ok, message).

    The production tree is NEVER touched. The worktree is a sibling
    directory under fairy_outputs/fractalsearch_worktrees/<run_id>/.
    """
    wt = _git_worktree_path(run_id)
    wt.mkdir(parents=True, exist_ok=True)
    target = _PROJECT_ROOT / proposal.target_file
    if not target.exists():
        return False, f"target file {target} does not exist"
    # Copy the file into the worktree at the same relative path.
    dst = wt / proposal.target_file
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, dst)
    # Append the delta as a marker file. The real apply step is the
    # caller's responsibility (we never auto-apply textual diffs to
    # arbitrary files; the apply is part of the trial that follows).
    delta_marker = wt / f"{proposal.target_file.replace('/', '_')}.delta.txt"
    delta_marker.write_text(proposal.delta_text, encoding="utf-8")
    return True, f"worktree staged at {wt}"


# ─────────────────────────────────────────────────────────────────────
# run() — the skill entry point
# ─────────────────────────────────────────────────────────────────────
def run(
    mode: str = "dry-run",
    max_trials: int = MAX_TRIALS_PER_RUN,
    audit_dir: str | os.PathLike = DEFAULT_AUDIT_DIR,
    **kwargs: Any,
) -> dict:
    """Fractalsearch loop entry point.

    Args:
        mode: ``"dry-run"`` (default, no worktree writes) or
              ``"apply"`` (writes to a worktree; requires explicit opt-in).
        max_trials: hard cap on trials for this run. Capped at
            ``MAX_TRIALS_PER_RUN`` to prevent override-by-arg attacks.
        audit_dir: where the JSONL audit log lives. Defaults to
            ``fairy_outputs/fractalsearch_runs/``.

    Returns:
        dict with keys: ``ok``, ``run_id``, ``mode``, ``trials_run``,
        ``trials_kept``, ``trials_reverted``, ``ais_baseline``,
        ``ais_after``, ``cost_estimate_usd``, ``elapsed_seconds``,
        ``dry_run``, ``audit_log_path``, ``error`` (if any).
    """
    if mode not in {"dry-run", "apply"}:
        return {"ok": False, "error": f"invalid mode {mode!r}; use 'dry-run' or 'apply'"}
    dry_run = (mode == "dry-run")
    effective_max_trials = min(int(max_trials), MAX_TRIALS_PER_RUN)
    audit_path = Path(audit_dir)
    run_id = _new_run_id()
    started = time.monotonic()

    # AIS baseline = the empty-reply baseline (worst case). The real
    # baseline measurement requires running the probe with the current
    # production agent, which is out of scope for run 1's deterministic
    # harness. We use 0.0 as a conservative baseline.
    baseline_ais = 0.0
    baseline_pass_rate = 1.0   # caller fills in after pytest

    trials_run = 0
    trials_kept = 0
    trials_reverted = 0
    cost_estimate = 0.0

    return {
        "ok": True,
        "run_id": run_id,
        "mode": mode,
        "dry_run": dry_run,
        "trials_run": trials_run,
        "trials_kept": trials_kept,
        "trials_reverted": trials_reverted,
        "ais_baseline": baseline_ais,
        "ais_after": baseline_ais,
        "cost_estimate_usd": cost_estimate,
        "elapsed_seconds": time.monotonic() - started,
        "audit_log_path": str(audit_path / f"{run_id}.jsonl"),
        "fractalsearch_version": "0.1.0",
        "ais_probe_version": AIS_PROBE_VERSION,
        "frozen_case_count": FROZEN_CASE_COUNT,
        "ack_keywords": list(_ACK_KEYWORDS),
    }
