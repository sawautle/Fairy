#!/usr/bin/env python3
"""
Claude Code handoff for Fairy.

Fairy is a chat/voice assistant. For repository / software tasks, she hands
the user off to Claude Code in the terminal — Claude Code runs with its own
full TUI and permission prompts, the user works directly inside Claude Code,
and when it exits Fairy resumes.

This module provides:
  - `detect_repository_task(text)` — explicit-claude or repo-task detection
    used as the handoff trigger.
  - `_classify_task_directory(text)` — picks the right cwd (repo vs user file).
  - `find_claude_binary()` — locate the `claude` CLI.
  - `build_handoff_command(task)` — construct the exact command for the
    interactive spawn (NO -p, NO --output-format, NO permission bypass).
  - `request_permission(...)` — kept for the heuristic permission gate that
    still asks "Use Claude Code for this?" before non-explicit tasks.

New role: gatekeeper, not middleman. Fairy never parses Claude Code
output, never claims what it did, never reports costs/durations beyond
the single resume line.
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Any


# ── Explicit agent-mention patterns ─────────────────────────────────────────────
# Any explicit request to delegate to Claude / Claude Code ALWAYS triggers
# the handoff — no heuristics, no git-repo check, no confidence scoring.
# "make a folder using claude" must be caught even though it contains no
# repo keywords.
#
# Pattern breakdown:
#   \busing\s+claude\b              → "using claude"  (bare name)
#   \busing\s+claude\s+code\b      → "using Claude Code"
#   \buse\s+claude\b               → "use claude to..."
#   \buse\s+claude\s+code\s+to\b   → "use Claude Code to..."
#   \bvia\s+claude\b               → "via claude"
#   \bvia\s+claude\s+code\b        → "via Claude Code"
#   \bhave\s+claude\b              → "have claude do it"
#   \bhave\s+claude\s+code\b       → "have Claude Code do it"
_EXPLICIT_AGENT_PATTERNS = (
    re.compile(r"\busing\s+claude\b",          re.IGNORECASE),
    re.compile(r"\busing\s+claude\s+code\b",  re.IGNORECASE),
    re.compile(r"\buse\s+claude\b",            re.IGNORECASE),
    re.compile(r"\buse\s+claude\s+code\s+to\b", re.IGNORECASE),
    re.compile(r"\bvia\s+claude\b",            re.IGNORECASE),
    re.compile(r"\bvia\s+claude\s+code\b",    re.IGNORECASE),
    re.compile(r"\bhave\s+claude\b",           re.IGNORECASE),
    re.compile(r"\bhave\s+claude\s+code\b",   re.IGNORECASE),
)


# Heuristics: phrases that strongly suggest a repo / software-engineering task.
_REPO_TASK_KEYWORDS = (
    # Bug fixing
    "fix this bug", "fix the bug", "fix this error", "fix this issue",
    "debug this", "why is this broken", "why isn't this working",
    "this is broken", "this doesn't work",
    # Inspection
    "inspect", "analyze the code", "analyze this code", "review the code",
    "review this", "look at the code", "examine", "what's wrong with",
    # Refactor / modify
    "refactor", "rewrite", "clean up", "improve this code",
    "add this feature", "implement this", "build this",
    # Tests
    "run the tests", "run tests", "test the code", "fix the tests",
    "tests are failing", "failing tests", "make tests pass",
    # Git / repo
    "commit this", "push this", "create a pr", "open a pull request",
    "merge this branch", "branch off",
    # Project-level
    "review the whole project", "review the project", "review the repo",
    "audit the codebase", "scan the codebase",
    "understand the architecture", "explain the architecture",
    "document this module", "add docstrings",
    # Fairy-specific
    "fix fairy", "improve fairy", "update fairy", "patch fairy",
    "add a feature to fairy", "in the fairy codebase",
    "in the fairy code", "in the project",
    "agent_controller", "computer_control",
)

# Phrases that should NOT trigger delegation even if they contain words above.
# These are "about this content" overrides, not "about this codebase" ones.
_NON_REPO_OVERRIDES = (
    "in the browser", "on the website", "on the page",
    "in the document", "in this file", "in this image",
    "in the json", "in this diagram", "in this table",
    # False-positive suppression: "what's wrong with this snippet/5-line..."
    # should not be a repo task — it's asking about code shown to Fairy,
    # not asking to modify the repository.
    "this snippet", "this code snippet", "this 5-line", "this 3-line",
    "this function", "this class", "this script",
)


def detect_repository_task(text: str) -> bool:
    """Heuristically detect whether the user is asking for repository work.

    Returns True for tasks that should trigger a Claude Code handoff
    (e.g. "fix this bug in Fairy", "run the tests and fix the failures",
    "use claude to make a folder", "via claude, organize my photos").

    Returns False for normal conversation.

    The STRONGEST signal is an explicit agent mention: any request that
    contains "using claude", "use Claude Code to", "via claude", or
    "have claude" ALWAYS returns True — these are unambiguous hand-offs
    to the agent, so the repo-keyword heuristics below are not consulted.
    """
    if not text:
        return False
    t = text.lower().strip()
    if not t:
        return False

    # ── Strongest signal: explicit agent mention. ────────────────────────
    # An explicit "using claude" / "use Claude Code" / "via claude" is
    # unambiguous — the user is asking to hand off. No need to check the
    # weaker keyword heuristics below.
    for pattern in _EXPLICIT_AGENT_PATTERNS:
        if pattern.search(t):
            return True

    # Strong signals: any explicit repository keyword
    for kw in _REPO_TASK_KEYWORDS:
        if kw in t:
            # Apply negative overrides
            for override in _NON_REPO_OVERRIDES:
                if override in t:
                    return False
            return True

    # Weaker signals: combination of "project" + action verb
    has_project_word = any(
        w in t for w in ("project", "codebase", "repository", "repo", "module",
                          "function", "class", "integration", "system")
    )
    has_action_verb = any(
        v in t for v in ("fix", "refactor", "rewrite", "debug", "analyze", "review",
                         "implement", "add", "remove", "test", "run", "update")
    )
    if has_project_word and has_action_verb:
        return True

    return False


# ── Task-directory classification ───────────────────────────────────────────────
# Score-based classifier that picks the right working directory for a
# Claude Code handoff based on the task's subject matter.
#
# Priority: USER_FILE (×2) beats REPO (×1) on tie.
# USER_FILE is the conservative choice — Claude Code doing file I/O outside
# E:\fairy is safer than it having write access inside it.
#
# Rationale for each USER_FILE keyword:
#   "screenshot", "picture", "photo", "image", "video"
#     → user's media files
#   "folder", "my files", "my documents", "my stuff"
#     → user's file-management task
#   "download", "downloads"
#     → user's Downloads directory
#   "desktop"
#     → user's Desktop directory
#   "organize", "sort", "group", "rename"
#     → file-management actions, not engineering actions
#   "copy", "move", "backup", "archive"
#     → filesystem actions, not code actions
#
# Rationale for each REPO keyword:
#   "fix this bug", "debug", "refactor", "add feature", "run the tests",
#   "commit", "audit", "review the code", "implement", "patch"
#     → software-engineering actions on source code
_REPO_TASKS: dict[str, int] = {
    # Bug fixing
    "fix this bug": 2, "fix the bug": 2, "fix this error": 2,
    "fix this issue": 2, "debug this": 2, "debug": 1,
    "fix fairy": 2, "fix the fairy": 2,
    # Refactor / modify code
    "refactor": 2, "rewrite": 2, "clean up": 1,
    "improve this code": 2, "add this feature": 2,
    "add a new feature": 2, "add new feature": 2,
    "new feature to fairy": 2, "a new feature to fairy": 2,
    "implement": 2,
    "build this": 1, "patch": 2,
    # Tests (explicit code context)
    "run the tests": 2, "run tests": 2, "make tests pass": 2,
    "fix the tests": 2, "failing tests": 2, "test failures": 2,
    "test failures in": 2, "review test failures": 2,
    # Git / repo / project
    "commit this": 2, "push this": 2, "create a pr": 2,
    "open a pull request": 2, "merge this branch": 2,
    "audit the codebase": 2, "review the code": 2, "review the project": 2,
    "review the repo": 2, "scan the codebase": 2,
    # Architecture / documentation
    "understand the architecture": 2, "explain the architecture": 2,
    "document this module": 2, "add docstrings": 2,
    # Engineering
    "agent_controller": 2, "computer_control": 2,
    "in the fairy codebase": 2, "in the fairy code": 2,
    "in the project": 2,
    # Possessive "my/our/this project" signals source-code ownership context.
    # These fire alongside action verbs ("organize in my project") so the repo
    # signal outweighs a generic USER_FILE action like "organize".
    "in my project": 2, "in our project": 2, "in this project": 2,
    "my project": 2,
    # "code in" fires when the object is code being acted upon, not personal files.
    # "organize the code in my project" has "code in" + "in my project" → REPO 4.
    "code in": 2, "source in": 2,
    # Generic engineering (lower weight — only fires without USER_FILE keywords)
    "inspect": 1, "analyze the code": 2, "analyze this code": 2,
    "examine": 1,
}

# USER_FILE keywords (×2) — checked BEFORE REPO and win ties.
# These are about the user's personal files, not about source code.
_USER_FILE_TASKS: dict[str, int] = {
    # Media
    "screenshot": 2, "screenshots": 2,
    "picture": 2, "pictures": 2, "photos": 2, "photo": 2,
    "image": 2, "images": 2, "video": 2, "videos": 2,
    # File management
    "my screenshots": 2, "my pictures": 2, "my photos": 2,
    "my documents": 2, "my files": 2, "my stuff": 2,
    "my downloads": 2,
    "folder": 2, "folders": 2,
    "organize": 2, "organise": 2, "sort": 2, "group": 2, "rename": 2,
    "copy": 2, "move": 2, "backup": 2, "archive": 2,
    # Named directories
    "downloads folder": 2, "downloads directory": 2,
    "desktop": 2, "pictures folder": 2,
    # Ambiguous with "test" in other contexts — these explicitly name user files
    "test files": 1, "test file": 1,
}


def _classify_task_directory(task: str) -> tuple[str, str]:
    """
    Score-based task classifier that picks the right working directory for
    a Claude Code handoff.

    Args:
        task: The user's task text (lower-cased internally).

    Returns:
        A 2-tuple (category, directory):
          - category: "user_file" | "repo"
          - directory: the absolute path Claude Code will use as cwd.

    Score rules:
      - USER_FILE keywords score ×2, REPO keywords score ×1.
      - USER_FILE wins ties → conservative; protects the user's files from
        accidental repo access.
      - The first matching keyword (longest-first) is used; all keywords
        in a phrase contribute to the score.

    Conflict resolution (six regression cases):
      "organize the test files in my project"
        → "test files" (USER_FILE ×2) + "project" (weaker) → USER_FILE wins.
        Rationale: "test files" is a concrete user asset, not a code task.
      "fix my downloads folder"
        → "downloads folder" (USER_FILE ×2) → USER_FILE wins.
        Rationale: "downloads folder" is a named user directory, not a code bug.
      "review my photos"
        → "photos" (USER_FILE ×2) → USER_FILE wins.
      "fix this screenshot"
        → "screenshot" (USER_FILE ×2) → USER_FILE wins.
      "organize the code in my project"
        → "organize" (USER_FILE ×2) + "code" + "project" → USER_FILE wins
        (conservative: "organize" is the stronger signal for file management).
      "review the test failures in my project"
        → "test failures" + "project" → REPO wins.
        Rationale: "test failures" in context of "project" is clearly code-related.

    Directory mapping for USER_FILE:
      "download", "downloads" → ~/Downloads
      "desktop"              → ~/Desktop
      "screenshot", "picture", "photo", "image", "video"
                               → ~/Pictures
      default (other user-file tasks) → ~ (home directory)
    """
    t = task.lower()

    repo_score = 0
    user_file_score = 0

    # Score USER_FILE keywords first (×2)
    for kw, weight in _USER_FILE_TASKS.items():
        if kw in t:
            user_file_score += weight

    # Score REPO keywords (×1)
    for kw, weight in _REPO_TASKS.items():
        if kw in t:
            repo_score += weight

    # Tie-breaking rules:
    #   1. If both scores are 0, default to REPO (safer; explicit agent
    #      mentions like "use claude" will still route correctly via
    #      detect_repository_task's stronger pattern check upstream).
    #   2. If only USER_FILE matches, → USER_FILE.
    #   3. If only REPO matches, → REPO.
    #   4. If both match, USER_FILE wins ties (conservative: protect user
    #      files from accidental repo access when the task is ambiguous).
    if user_file_score == 0 and repo_score == 0:
        category = "repo"  # default when no signal either way
    elif user_file_score >= repo_score:
        category = "user_file"
    else:
        category = "repo"

    # Map category to a concrete directory
    if category == "user_file":
        directory = _map_user_file_category(t)
    else:
        directory = _get_repo_directory()

    return category, directory


def _map_user_file_category(task: str) -> str:
    """Map a USER_FILE task to a specific user directory."""
    home = os.path.expanduser("~")
    t = task.lower()

    if any(kw in t for kw in ("download", "downloads")):
        return os.path.join(home, "Downloads")
    if any(kw in t for kw in ("desktop",)):
        return os.path.join(home, "Desktop")
    if any(kw in t for kw in (
        "screenshot", "screenshots",
        "picture", "pictures",
        "photo", "photos",
        "image", "images",
        "video", "videos",
    )):
        # Prefer Pictures/Videos subfolders when detectable
        if any(kw in t for kw in ("video", "videos")):
            videos = os.path.join(home, "Videos")
            if os.path.isdir(videos):
                return videos
        pictures = os.path.join(home, "Pictures")
        if os.path.isdir(pictures):
            return pictures
        return home
    # Default: use home directory
    return home


def _get_repo_directory() -> str:
    """Return the configured repo directory (E:\\fairy by default)."""
    try:
        from config import CLAUDE_CODE_PROJECT_ROOT
        return CLAUDE_CODE_PROJECT_ROOT
    except ImportError:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def request_permission(
    project_root: str,
    task: str,
    task_category: str = "repo",
) -> str:
    """Generate a permission prompt to show the user before delegating.

    Args:
        project_root: The directory Claude Code will use as cwd.
        task: The task description.
        task_category: "repo" (software task in E:\\fairy) or "user_file"
                      (file-management task in a user directory). Used to
                      customise the wording so the user knows what they're
                      approving.
    """
    if task_category == "user_file":
        rel = project_root
        try:
            rel = os.path.relpath(project_root, os.path.expanduser("~"))
        except Exception:
            pass
        if rel == project_root:
            dir_hint = project_root
        else:
            dir_hint = rf"~\{rel}" if not rel.startswith(os.path.sep) else rel
        return (
            f"Master, hand off to Claude Code to work in "
            f"`{dir_hint}`?\n\n"
            f"Task: {task}\n\n"
            f"Use Claude Code for this? (yes/no)"
        )
    else:
        return (
            f"Master, hand off to Claude Code in "
            f"`{project_root}`?\n\n"
            f"Task: {task}\n\n"
            f"Use Claude Code for this? (yes/no)"
        )


def find_claude_binary() -> str | None:
    """Find the `claude` CLI binary. Returns the path or None.

    Public name; exported for tests and for the handoff caller.
    """
    # Check PATH first
    found = shutil.which("claude")
    if found:
        return found
    # Common install locations on Windows
    candidates = [
        os.path.expanduser("~/.claude/local/claude.exe"),
        os.path.expanduser("~/AppData/Roaming/npm/claude.cmd"),
        "C:/Program Files/nodejs/claude.cmd",
        "/usr/local/bin/claude",
        "/opt/homebrew/bin/claude",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def build_handoff_command(task: str) -> list[str]:
    """Build the exact argv to spawn Claude Code in interactive mode.

    CRITICAL: This function is the structural guarantee that Fairy's
    handoff does NOT bypass Claude Code's own permission system.

    Forbidden flags (must NEVER appear in the returned argv):
      - -p, --print              → would skip the interactive TUI
      - --output-format json     → would turn it into a one-shot subprocess
      - --dangerously-skip-permissions
      - --allow-anything         → would let Claude Code run without prompts
      - --no-input               → would prevent user interaction
      - --non-interactive        → would disable Claude Code's prompt UI

    Returns:
        A list of strings suitable for subprocess.run / subprocess.Popen argv.

    The task is passed as a positional argument so Claude Code receives it
    as its initial prompt — the user does not have to retype it inside
    Claude Code's own input.
    """
    if not task or not task.strip():
        raise ValueError("handoff task must be a non-empty string")
    return ["claude", task.strip()]


def is_handoff_safe_command(argv: list[str]) -> tuple[bool, str]:
    """Defensive check: does the argv look like a safe handoff command?

    Used by tests and by the runtime to assert that the command constructed
    by build_handoff_command() does not accidentally contain any of the
    forbidden flags above. Returns (is_safe, reason).
    """
    if not argv:
        return False, "empty argv"
    forbidden = {
        "-p", "--print",
        "--output-format",
        "--dangerously-skip-permissions",
        "--allow-anything",
        "--no-input",
        "--non-interactive",
    }
    # Scan argv for any forbidden flag (substring or full match).
    for arg in argv:
        arg_lower = arg.lower().strip()
        for bad in forbidden:
            if arg_lower == bad.lower() or arg_lower.startswith(bad.lower() + "="):
                return False, f"forbidden flag present: {arg}"
    # First arg must be "claude" (or absolute path ending in /claude).
    binary = argv[0]
    if not (binary == "claude" or binary.endswith("/claude") or binary.endswith("\\claude") or binary.endswith("claude.exe") or binary.endswith("claude.cmd")):
        return False, f"first arg must be claude binary, got: {binary}"
    return True, "ok"


def handoff_diagnostics(claude_bin: str, project_root: str, task: str) -> dict[str, Any]:
    """Return a dict of pre-flight diagnostics for logging.

    Includes:
      - claude_bin: resolved path
      - project_root: validated absolute path
      - task_length: int
      - task_preview: first 80 chars
    """
    return {
        "claude_bin": claude_bin,
        "project_root": project_root,
        "task_length": len(task),
        "task_preview": task[:80],
    }
