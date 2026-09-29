"""
controller/approval_gate.py
============================
Unified approval gate for filesystem-mutating tools.

Background
----------
The Discord path already had an approval queue (controller/discord_bot.py)
that escalates ``RESTRICTED_TOOLS`` to the Master via the bot. The terminal
path had no such gate, so a ``write_file`` call from Fairy's terminal
session could silently clobber repo files. This module is the single place
that:

  1. Decides whether a tool call needs approval (which tools, which paths).
  2. Decides the risk tier (new file vs. overwrite vs. load-bearing).
  3. Routes the request to the right backend (terminal prompt vs. Discord
     queue) — the Discord side reuses the existing ``MasterApprovalRequired``
     / asyncio.Event machinery in ``controller/discord_bot.py``; we never
     duplicate that logic.
  4. Records every approval/denial to the same audit log the Discord side
     uses (``controller/discord_approvals.log``) so there is one source of
     truth for "what did Master just allow?".

The terminal backend is intentionally simple — print the request, read a
single line, branch on ✅ / ❌ / allow / deny. The Discord backend just
re-raises ``MasterApprovalRequired`` with a populated request_info dict
so the existing async code path picks it up unchanged.

Pure helpers (tier detection, file existence, etc.) are deliberately kept
import-time free of any network/IO so the test suite can call them
without bringing in the rest of Fairy.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

# ─────────────────────────────────────────────────────────────────────
# Constants — keep them in one place so both front-ends agree.
# ─────────────────────────────────────────────────────────────────────

# Tools that this gate can mediate. The agent controller decides whether
# to call us; we only answer "approved / denied / needs_clarification".
# Keep this list aligned with the tools actually wired up in
# controller/agent_controller.py — if you add a file-mutating tool there,
# add the name here too.
FILE_MUTATING_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "mkdir",
    "delete",
    "zip_create",
    "zip_extract",
    "move",
    "rename",
})

# Load-bearing path patterns. If the resolved path matches any of these,
# the gate adds an extra warning line to the approval prompt and refuses
# to auto-approve in the terminal path. These are intentionally
# conservative — a false positive (extra warning) is far better than a
# false negative (silent clobber of main.py).
_LOAD_BEARING_RULES: tuple[tuple[str, str], ...] = (
    (r"[/\\]main\.py$", "repo entry point main.py"),
    (r"[/\\]fairy\.py$", "TUI entry point fairy.py"),
    (r"[/\\]controller[/\\].*\.py$", "controller module"),
    (r"[/\\]__init__\.py$", "package __init__.py"),
    (r"[/\\]tests[/\\].*\.py$", "test file"),
    (r"[/\\]pytest\.ini$", "pytest config"),
    (r"[/\\]requirements\.txt$", "Python requirements"),
    (r"[/\\]pyproject\.toml$", "Python project config"),
    (r"[/\\]\.gitignore$", "git ignore file"),
    (r"[/\\]\.git[/\\]", "git internals"),
    (r"[/\\]README\.md$", "README"),
)

# Where the audit log lives. This MUST match the path the Discord side
# uses (``controller/discord_approvals.log``) so both backends write to
# the same file. Computed relative to this file so it works regardless
# of cwd.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_MODULE_DIR)
AUDIT_LOG_PATH = os.path.join(_MODULE_DIR, "discord_approvals.log")

# Outcomes — mirror controller/discord_bot.py so the audit log is
# uniform across both backends.
OUTCOME_APPROVED = "approved"
OUTCOME_DENIED = "denied"
OUTCOME_TIMED_OUT = "timed_out"
OUTCOME_BLOCKED_LOAD_BEARING = "blocked_load_bearing"
OUTCOME_AUTOPASS_NEW = "autopass_new_file"
OUTCOME_AUTOPASS_FALLBACK = "autopass_no_gate"


# ─────────────────────────────────────────────────────────────────────
# Tier classification — pure functions, no I/O.
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PathTier:
    """Result of classifying a target path.

    ``is_load_bearing`` — path matches a protected pattern (repo entry
    point, git internals, etc.). Approval prompt must include an extra
    warning line and a git-diff-style preview if available.

    ``is_overwrite`` — the file already exists. The prompt must ask
    "overwrite?" explicitly and never silently clobber.

    ``suggested_alt`` — a safe alternative path under ``fairy_outputs/``
    for casual file creation. Suggested in the prompt but not enforced.
    """
    raw_path: str
    normalized: str
    is_load_bearing: bool
    load_bearing_reason: Optional[str]
    is_overwrite: bool
    is_directory: bool
    suggested_alt: Optional[str]

    @property
    def tier(self) -> str:
        if self.is_load_bearing:
            return "load_bearing"
        if self.is_overwrite:
            return "overwrite"
        return "new_file"


def _normalize(path: str) -> str:
    """Resolve a path to an absolute, OS-normalized form for comparison."""
    if not path:
        return ""
    try:
        return os.path.normpath(os.path.abspath(path))
    except Exception:
        return path


def _is_under_project(normalized: str) -> bool:
    """True if the path lives inside the project root (load-bearing rules only
    fire for project paths — a Windows-Python/main.py in user data is not
    a load-bearing target for this repo)."""
    try:
        n = normalized.lower()
        root = _PROJECT_DIR.lower()
        return n == root or n.startswith(root + os.sep)
    except Exception:
        return False


def classify_path(path: str) -> PathTier:
    """Classify a target path into a safety tier.

    Pure: takes a string, returns a PathTier. No I/O beyond
    ``os.path.exists`` and ``os.path.isdir`` (one stat each).
    """
    if not path:
        return PathTier(
            raw_path=path or "",
            normalized="",
            is_load_bearing=False,
            load_bearing_reason=None,
            is_overwrite=False,
            is_directory=False,
            suggested_alt=_suggest_alt_path(path or ""),
        )

    normalized = _normalize(path)
    is_dir = os.path.isdir(normalized)
    is_file = os.path.isfile(normalized)
    is_overwrite = is_file or is_dir

    # Load-bearing detection only applies inside the project root, so
    # an attacker's main.py in /tmp doesn't trigger the warning.
    is_lb = False
    lb_reason: Optional[str] = None
    if _is_under_project(normalized):
        # Bare repo-root writes (no subdir, no extension) are also risky —
        # dropping "main.py" or "build.log" in the root is exactly the
        # pattern that nearly clobbered main.py.
        rel = os.path.relpath(normalized, _PROJECT_DIR)
        if rel == os.path.basename(normalized) and rel.count(os.sep) == 0:
            is_lb = True
            lb_reason = "repo root write"
        else:
            for pat, reason in _LOAD_BEARING_RULES:
                if re.search(pat, normalized):
                    is_lb = True
                    lb_reason = reason
                    break

    suggested = _suggest_alt_path(path) if not is_lb else None

    return PathTier(
        raw_path=path,
        normalized=normalized,
        is_load_bearing=is_lb,
        load_bearing_reason=lb_reason,
        is_overwrite=is_overwrite,
        is_directory=is_dir,
        suggested_alt=suggested,
    )


def _suggest_alt_path(path: str) -> Optional[str]:
    """Suggest a safe alternative under ``fairy_outputs/`` for casual
    file creation. Returns ``None`` if the path is already there or is
    obviously an in-place edit."""
    if not path:
        return None
    if "fairy_outputs" in path.replace("\\", "/"):
        return None
    base = os.path.basename(path) or "output.txt"
    return os.path.join(_PROJECT_DIR, "fairy_outputs", base)


# ─────────────────────────────────────────────────────────────────────
# Git integration — used to show a diff preview for load-bearing writes.
# Pure: takes a project root, returns a short diff string or empty.
# ─────────────────────────────────────────────────────────────────────

def _git_diff_preview(project_root: str, target: str) -> str:
    """Best-effort ``git diff`` for an existing tracked file. Empty
    string if git is unavailable, the file isn't tracked, or git fails.
    Never raises — failure to produce a preview is non-fatal."""
    if not project_root or not target:
        return ""
    try:
        rel = os.path.relpath(target, project_root)
    except Exception:
        return ""
    try:
        out = subprocess.run(
            ["git", "diff", "--", rel],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return ""
        text = (out.stdout or "").strip()
        # Cap so the prompt doesn't blow up.
        if len(text) > 800:
            text = text[:800] + "\n… (truncated)"
        return text
    except Exception:
        return ""


def _git_tracked(project_root: str, target: str) -> bool:
    """True if ``target`` is tracked by git. Used as a load-bearing signal
    on top of the regex rules. Never raises."""
    if not project_root or not target:
        return False
    try:
        rel = os.path.relpath(target, project_root)
    except Exception:
        return False
    try:
        out = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.returncode == 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────
# Audit log — append-only, same format as the Discord side.
# ─────────────────────────────────────────────────────────────────────

_audit_lock = threading.Lock()


def _format_audit_line(*, timestamp: datetime, user_name: str, user_id: Any,
                        tool: str, args: Any, outcome: str,
                        tier: Optional[str] = None,
                        path: Optional[str] = None) -> str:
    """One audit-log line per request, JSON, one line so tail -f stays readable.
    Matches the format the Discord side uses (see
    controller/discord_bot.py:_format_audit_line) so the log is uniform."""
    record = {
        "ts": timestamp.isoformat(timespec="seconds"),
        "user": user_name,
        "user_id": str(user_id) if user_id is not None else "",
        "tool": tool,
        "args": args if isinstance(args, dict) else {"_raw": repr(args)},
        "outcome": outcome,
    }
    if tier is not None:
        record["tier"] = tier
    if path is not None:
        record["path"] = path
    return json.dumps(record, ensure_ascii=False, default=str)


def append_audit(*, user_name: str, user_id: Any, tool: str, args: Any,
                  outcome: str, tier: Optional[str] = None,
                  path: Optional[str] = None) -> None:
    """Append a single record to the audit log. Never raises into the
    caller's flow — audit failures must NEVER break tool execution."""
    try:
        line = _format_audit_line(
            timestamp=datetime.now(),
            user_name=user_name,
            user_id=user_id,
            tool=tool,
            args=args,
            outcome=outcome,
            tier=tier,
            path=path,
        )
        with _audit_lock:
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:  # pragma: no cover - audit must never break chat
        print(f"[FAIRY-APPROVAL] audit log write failed: {exc}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────
# Prompt builders — pure. Return the exact text the terminal or Discord
# user will see. Centralized so the wording stays consistent.
# ─────────────────────────────────────────────────────────────────────

def _format_args_summary(tool: str, args: Any) -> str:
    """One-line summary of a tool call. Deliberately short — the
    approval prompt should be scannable, not a wall of JSON."""
    if not isinstance(args, dict):
        return repr(args)[:120]
    if tool == "write_file":
        path = args.get("path", "?")
        content = args.get("content", "")
        size = len(content) if isinstance(content, str) else 0
        return f"path={path!r}  size={size} chars"
    if tool == "mkdir":
        return f"path={args.get('path', '?')!r}"
    if tool == "delete":
        return f"path={args.get('path', '?')!r}"
    if tool in ("zip_create", "zip_extract"):
        return f"{args.get('archive', '?')!r}  src={args.get('src', '?')!r}"
    if tool in ("move", "rename"):
        return f"src={args.get('src', '?')!r} → dst={args.get('dst', '?')!r}"
    # Fallback — short, key-only.
    items = [f"{k}={v!r}" for k, v in list(args.items())[:4]]
    return " ".join(items)[:200]


def build_terminal_prompt(*, user_name: str, tool: str, args: Any,
                           tier: PathTier) -> str:
    """Build the in-terminal approval prompt. Multi-line, scannable,
    with the path+content summary on one line so the Master can see the
    full picture at a glance. Returns a string ending in the ✅/❌ cue."""
    summary = _format_args_summary(tool, args)
    lines: list[str] = []
    lines.append(f"[🛡  Fairy approval gate] {user_name} is requesting:")
    lines.append(f"    tool : {tool}")
    lines.append(f"    args : {summary}")

    # Tier-specific warnings.
    if tier.is_load_bearing:
        lines.append("")
        lines.append(f"    ⚠   load-bearing path: {tier.load_bearing_reason}")
        lines.append(f"    ⚠   target : {tier.normalized}")
        if tier.is_overwrite:
            diff = _git_diff_preview(_PROJECT_DIR, tier.normalized)
            if diff:
                lines.append("    ── git diff preview (existing tracked file) ──")
                for ln in diff.splitlines()[:20]:
                    lines.append(f"    │ {ln}")
            elif _git_tracked(_PROJECT_DIR, tier.normalized):
                lines.append("    ⚠   file is git-tracked — overwrite will be visible in git status")
        lines.append("")
        lines.append("    Type 'allow' to proceed, 'deny' to refuse, or 'alt' to redirect to fairy_outputs/.")
    elif tier.is_overwrite:
        lines.append("")
        lines.append(f"    ⚠   OVERWRITE — file already exists: {tier.normalized}")
        lines.append("    Type 'allow' to overwrite, 'deny' to keep, or 'alt' to redirect to fairy_outputs/.")
    else:
        if tier.suggested_alt and os.sep not in tier.suggested_alt.split("fairy_outputs")[-1]:
            lines.append("")
            lines.append(f"    💡 tip: for casual file creation consider {tier.suggested_alt}")

    lines.append("")
    lines.append("    ✅ allow   ❌ deny   (or type allow/deny/alt)")
    return "\n".join(lines)


def build_discord_summary(tool: str, args: Any) -> str:
    """The compact summary the Discord side will see. Kept short
    because Discord messages have a 2000-char limit and the existing
    _format_tool_summary in controller/discord_bot.py already does
    similar work — we just add the tier info for the gate's path-
    sensitive tools."""
    summary = _format_args_summary(tool, args)
    tier = classify_path(args.get("path") if isinstance(args, dict) else None)
    tier_note = ""
    if tier.is_load_bearing:
        tier_note = f" — LOAD-BEARING: {tier.load_bearing_reason}"
    elif tier.is_overwrite:
        tier_note = " — OVERWRITE existing file"
    return f"`{tool}` {summary}{tier_note}"


# ─────────────────────────────────────────────────────────────────────
# Terminal backend — prompt, read line, branch.
# ─────────────────────────────────────────────────────────────────────

# Decisions recognized in the terminal prompt. Same vocabulary as the
# Discord side ("allow" / "deny" / etc.) so the Master can be consistent.
_TERMINAL_ALLOW = {"allow", "approve", "approved", "yes", "y", "✅", "ok", "okay", "sure", "do it"}
_TERMINAL_DENY = {"deny", "denied", "no", "n", "❌", "nope", "nah", "stop", "refuse"}
_TERMINAL_ALT = {"alt", "redirect", "fairy_outputs"}


def _parse_terminal_decision(text: str) -> str:
    """Map a single line of terminal input to a decision. Returns
    one of OUTCOME_APPROVED / OUTCOME_DENIED / "alt" / "unknown"."""
    if not text:
        return "unknown"
    t = text.lower().strip()
    if t in _TERMINAL_ALLOW:
        return OUTCOME_APPROVED
    if t in _TERMINAL_DENY:
        return OUTCOME_DENIED
    if t in _TERMINAL_ALT:
        return "alt"
    return "unknown"


def _redirect_to_alt(args: dict, tier: PathTier) -> dict:
    """Return a copy of ``args`` with the path replaced by the suggested
    alternative. Only used when the Master types "alt" in the terminal."""
    if not isinstance(args, dict) or not tier.suggested_alt:
        return args
    new_args = dict(args)
    if "path" in new_args:
        new_args["path"] = tier.suggested_alt
    elif "src" in new_args:
        new_args["src"] = tier.suggested_alt
    return new_args


def request_approval_terminal(
    *,
    user_name: str,
    tool: str,
    args: dict,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> tuple[str, dict]:
    """Prompt the Master in the terminal for a ✅/❌ decision.

    Returns ``(outcome, args)`` where ``outcome`` is one of
    OUTCOME_APPROVED / OUTCOME_DENIED and ``args`` is the (possibly
    redirected) argument dict to actually use.

    Pure with respect to module state — all I/O is via the injected
    ``input_fn`` / ``output_fn`` so tests can drive it without a real
    terminal. The function never raises into the caller's flow;
    KeyboardInterrupt is mapped to "denied" so the chat path doesn't
    die if the Master hits Ctrl-C.
    """
    # Pull a path out of args for tier classification, but tolerate tools
    # that don't have one.
    target = ""
    if isinstance(args, dict):
        target = args.get("path") or args.get("src") or args.get("archive") or ""
    tier = classify_path(target)

    prompt_text = build_terminal_prompt(
        user_name=user_name, tool=tool, args=args, tier=tier,
    )

    # Always show the prompt + log the request before asking.
    output_fn(prompt_text)
    append_audit(
        user_name=user_name, user_id="terminal",
        tool=tool, args=args, outcome="requested",
        tier=tier.tier, path=target or None,
    )

    try:
        raw = input_fn("approval> ")
    except (KeyboardInterrupt, EOFError):
        append_audit(
            user_name=user_name, user_id="terminal",
            tool=tool, args=args, outcome=OUTCOME_DENIED,
            tier=tier.tier, path=target or None,
        )
        return OUTCOME_DENIED, args

    decision = _parse_terminal_decision(raw)
    if decision == "alt" and tier.suggested_alt:
        redirected = _redirect_to_alt(args, tier)
        append_audit(
            user_name=user_name, user_id="terminal",
            tool=tool, args={"original": args, "redirected": redirected},
            outcome=OUTCOME_APPROVED,
            tier=tier.tier, path=tier.suggested_alt,
        )
        output_fn(f"[ok] redirected to {tier.suggested_alt}")
        return OUTCOME_APPROVED, redirected

    if decision == OUTCOME_APPROVED:
        append_audit(
            user_name=user_name, user_id="terminal",
            tool=tool, args=args, outcome=OUTCOME_APPROVED,
            tier=tier.tier, path=target or None,
        )
        return OUTCOME_APPROVED, args

    # Unknown or explicit deny → refuse. Unknown is treated as deny for
    # safety; the Master can always re-issue the command.
    append_audit(
        user_name=user_name, user_id="terminal",
        tool=tool, args=args, outcome=OUTCOME_DENIED,
        tier=tier.tier, path=target or None,
    )
    return OUTCOME_DENIED, args


# ─────────────────────────────────────────────────────────────────────
# Discord backend — re-raise MasterApprovalRequired so the existing
# async code path picks it up unchanged. The Discord side already has
# the FIFO, the timeout, the audit logging, and the message reactions;
# we just feed it a properly-shaped request.
# ─────────────────────────────────────────────────────────────────────

class ApprovalRequired(Exception):
    """Raised from the gate when a tool call needs Discord-side
    approval. The async side catches this and posts the request."""

    def __init__(self, request_info: dict):
        self.request_info = request_info
        super().__init__(f"ApprovalRequired: tool={request_info.get('tool')}")


def request_approval_discord(*, user_name: str, user_id: Any,
                              tool: str, args: dict) -> None:
    """Build the request_info payload the Discord side expects and
    raise ``ApprovalRequired``. The async side will catch this and
    call into the existing FIFO queue in controller/discord_bot.py.

    We deliberately do NOT talk to Discord here — that's the async
    side's job. The exception is the contract.
    """
    target = ""
    if isinstance(args, dict):
        target = args.get("path") or args.get("src") or args.get("archive") or ""
    tier = classify_path(target)

    request_info = {
        "tool": tool,
        "args": args,
        "user_id": user_id,
        "user_name": user_name,
        "channel_id": None,
        "text": f"approval gate: {tool}",
        "summary": build_discord_summary(tool, args),
        "tier": tier.tier,
        "path": target or None,
    }
    append_audit(
        user_name=user_name, user_id=user_id,
        tool=tool, args=args, outcome="requested",
        tier=tier.tier, path=target or None,
    )
    raise ApprovalRequired(request_info)


# ─────────────────────────────────────────────────────────────────────
# Unified entry point — agent_controller calls THIS, not the backends
# directly. Decides backend based on which front-end is active.
# ─────────────────────────────────────────────────────────────────────

# Module-level flag the Discord bot sets to True after it has wrapped
# agent_controller._dispatch_tool. The terminal path never sets it.
# This is the "which backend am I?" signal — we read it on every call
# so a single agent_controller module can be loaded from either front
# end without the import order mattering.
_DISCORD_ACTIVE = False


def set_discord_active(active: bool) -> None:
    """Discord bot calls this on startup/shutdown so the gate knows
    which backend to route through. The terminal path never touches
    it (default False)."""
    global _DISCORD_ACTIVE
    _DISCORD_ACTIVE = bool(active)


def is_discord_active() -> bool:
    return _DISCORD_ACTIVE


def request_approval(*, user_name: str, user_id: Any,
                      tool: str, args: dict) -> tuple[str, dict]:
    """Unified entry point. Returns ``(outcome, args)`` for the
    terminal path (synchronous, blocks on input). For the Discord
    path, raises ``ApprovalRequired`` so the async side can post the
    request and wait on its existing asyncio.Event.

    The contract for callers: if this function returns, ``outcome`` is
    one of OUTCOME_APPROVED / OUTCOME_DENIED and ``args`` is the dict
    to actually use. If it raises ``ApprovalRequired``, the caller is
    on the Discord async path and should let it propagate.
    """
    if _DISCORD_ACTIVE:
        # Discord path — raise so the async side can take over.
        request_approval_discord(
            user_name=user_name, user_id=user_id,
            tool=tool, args=args,
        )
        # Unreachable — request_approval_discord always raises.
        return OUTCOME_DENIED, args  # pragma: no cover

    # Terminal path — block on the prompt.
    return request_approval_terminal(
        user_name=user_name, tool=tool, args=args,
    )
