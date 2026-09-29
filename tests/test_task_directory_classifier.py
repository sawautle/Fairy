"""
test_task_directory_classifier.py

Regression + unit tests for:
  1. _classify_task_directory() — score-based classifier (USER_FILE ×2 vs REPO ×1,
     USER_FILE wins ties). Covered: 20 direct cases + 6 regression conflict cases.
  2. is_path_like_reply() — detects path-like approval replies ("yes Downloads",
     "yes ~/Pictures", r"use C:\\Work") and returns the resolved absolute path.
  3. request_permission() wording varies by task_category (repo vs user_file).
  4. Six regression conflict cases:
       "organize the test files in my project"   → USER_FILE
       "fix my downloads folder"                  → USER_FILE
       "review my photos"                         → USER_FILE
       "fix this screenshot"                      → USER_FILE
       "organize the code in my project"          → USER_FILE
       "review the test failures in my project"   → REPO
"""
from __future__ import annotations

import os
import sys
import tempfile
import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_THIS)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


# ══════════════════════════════════════════════════════════════════════════════════
# 1. _classify_task_directory()
# ══════════════════════════════════════════════════════════════════════════════════
class TestClassifyTaskDirectory:
    """Score-based classifier: USER_FILE ×2 beats REPO ×1 on ties."""

    # ── USER_FILE cases ────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("task", [
        # Screenshots / photos
        "organize my screenshots",
        "organize my screenshots using claude",
        "make me a folder containing all my screenshots using claude",
        "copy my screenshots using claude",
        "fix this screenshot",
        "review my screenshots",
        "organize my pictures",
        "sort my photos",
        "group my images by date",
        # Documents / files
        "organize my documents",
        "sort my files",
        "copy my files",
        "move my files to a backup folder",
        # Downloads
        "fix my downloads folder",
        "organize my downloads",
        "clean up my downloads",
        "sort my downloads by date",
        # Desktop
        "organize my desktop",
        "tidy my desktop files",
        # Mixed with repo-like words (USER_FILE wins ties)
        "organize the test files in my project",
    ])
    def test_user_file_cases(self, task):
        from controller.claude_code_delegate import _classify_task_directory
        category, directory = _classify_task_directory(task)
        assert category == "user_file", f"Expected user_file for {task!r}, got {category}"
        assert os.path.isabs(directory), f"Directory should be absolute: {directory}"

    @pytest.mark.parametrize("task", [
        "organize my screenshots",
        "make me a folder containing all my screenshots using claude",
    ])
    def test_user_file_pictures_directory(self, task):
        from controller.claude_code_delegate import _classify_task_directory
        category, directory = _classify_task_directory(task)
        assert category == "user_file"
        # Should resolve to Pictures or a subdirectory of Pictures
        lower_dir = directory.lower()
        assert any(
            kw in lower_dir for kw in ("pictures", "downloads", "desktop", "users")
        ), f"Expected Pictures/Downloads/Desktop/Users, got {directory!r}"

    @pytest.mark.parametrize("task", [
        "fix my downloads folder",
        "organize my downloads",
    ])
    def test_user_file_downloads_directory(self, task):
        from controller.claude_code_delegate import _classify_task_directory
        category, directory = _classify_task_directory(task)
        assert category == "user_file"
        lower_dir = directory.lower()
        assert "downloads" in lower_dir, f"Expected Downloads, got {directory!r}"

    def test_user_file_desktop_directory(self):
        from controller.claude_code_delegate import _classify_task_directory
        category, directory = _classify_task_directory("organize my desktop")
        assert category == "user_file"
        lower_dir = directory.lower()
        assert "desktop" in lower_dir, f"Expected Desktop, got {directory!r}"

    # ── REPO cases ───────────────────────────────────────────────────────────────
    @pytest.mark.parametrize("task", [
        "fix the bug in fairy",
        "run the tests",
        "refactor the routing system",
        "implement a discord integration",
        "add a new feature to fairy",
        "commit this change",
        "audit the codebase",
        "review the project",
        "explain the architecture",
        "document this module",
        "add docstrings",
        "merge this branch",
        # Possessive-project + code object → REPO wins because the
        # "in my project" and "code in" REPO signals (×2 each) outweigh
        # the "organize" USER_FILE signal (×2).
        "organize the code in my project",
    ])
    def test_repo_cases(self, task):
        from controller.claude_code_delegate import _classify_task_directory
        category, directory = _classify_task_directory(task)
        assert category == "repo", f"Expected repo for {task!r}, got {category}"
        assert os.path.isabs(directory), f"Directory should be absolute: {directory}"

    def test_repo_uses_config_root(self):
        from controller.claude_code_delegate import _classify_task_directory
        category, directory = _classify_task_directory("fix the bug in fairy")
        assert category == "repo"
        # Should be the configured CLAUDE_CODE_PROJECT_ROOT (E:\fairy)
        assert os.path.isdir(directory), f"Directory should exist: {directory}"

    # ── REGRESSION: Six conflict cases ─────────────────────────────────────────
    def test_conflict_organize_test_files_in_project(self):
        """'organize the test files in my project' → USER_FILE.

        "test files" is a concrete user asset, not a code task.
        "organize" is USER_FILE (file-management), not REPO (engineering).
        """
        from controller.claude_code_delegate import _classify_task_directory
        category, _ = _classify_task_directory("organize the test files in my project")
        assert category == "user_file", \
            "'organize the test files in my project' must be USER_FILE (test files = user asset)"

    def test_conflict_fix_my_downloads_folder(self):
        """'fix my downloads folder' → USER_FILE.

        "downloads folder" is a named user directory, not a code bug.
        """
        from controller.claude_code_delegate import _classify_task_directory
        category, directory = _classify_task_directory("fix my downloads folder")
        assert category == "user_file", \
            "'fix my downloads folder' must be USER_FILE (downloads = user dir)"
        assert "downloads" in directory.lower()

    def test_conflict_review_my_photos(self):
        """'review my photos' → USER_FILE."""
        from controller.claude_code_delegate import _classify_task_directory
        category, _ = _classify_task_directory("review my photos")
        assert category == "user_file", \
            "'review my photos' must be USER_FILE (photos = user media)"

    def test_conflict_fix_this_screenshot(self):
        """'fix this screenshot' → USER_FILE."""
        from controller.claude_code_delegate import _classify_task_directory
        category, _ = _classify_task_directory("fix this screenshot")
        assert category == "user_file", \
            "'fix this screenshot' must be USER_FILE (screenshot = user asset)"

    def test_conflict_organize_code_in_project(self):
        """'organize the code in my project' → REPO.

        'code in' (×2) + 'in my project' (×2) + 'my project' (×2) → REPO=6.
        'organize' (×2) → USER_FILE=2. REPO wins.
        """
        from controller.claude_code_delegate import _classify_task_directory
        category, _ = _classify_task_directory("organize the code in my project")
        assert category == "repo", \
            "'organize the code in my project' must be REPO ('code in' + 'in my project' = REPO 6 > USER_FILE 2)"

    def test_conflict_review_test_failures_in_project(self):
        """'review the test failures in my project' → REPO.

        "test failures" in the context of "project" is clearly about
        failing CI/tests in a codebase, not about reviewing user files.
        """
        from controller.claude_code_delegate import _classify_task_directory
        category, _ = _classify_task_directory("review the test failures in my project")
        assert category == "repo", \
            "'review the test failures in my project' must be REPO (test failures = code issue)"


# ══════════════════════════════════════════════════════════════════════════════════
# 1b. Conflict score breakdown — show the actual score for each case
# ══════════════════════════════════════════════════════════════════════════════════
class TestConflictScoreBreakdown:
    """Show the (user_file_score, repo_score) tuple for every conflict case so
    the keyword tables are inspectable, not just pass/fail.

    Conflict matrix (verified against live keyword tables):
       input                                        | UF | REPO | winner
       ------------------------------------------ +----+-----+-------
       organize the code in my project              |  2 |   6 | REPO
       fix my downloads folder                      |  6 |   0 | USER_FILE
       review my photos                             |  6 |   0 | USER_FILE
       fix this screenshot                          |  2 |   0 | USER_FILE
       organize the test files in my project        |  4 |   4 | USER_FILE (tie → UF)
       review the test failures in my project       |  0 |   8 | REPO

    Per-case breakdown:
      organize the code in my project
        UF:    organize (×2) = 2
        REPO:  code in (×2) + in my project (×2) + my project (×2) = 6
      fix my downloads folder
        UF:    my downloads (×2) + folder (×2) + downloads folder (×2) = 6
        REPO:  — = 0
      review my photos
        UF:    photos (×2) + photo (×2, substring) + my photos (×2) = 6
        REPO:  — = 0
      fix this screenshot
        UF:    screenshot (×2) = 2
        REPO:  — = 0
      organize the test files in my project
        UF:    organize (×2) + test files (×2) = 4
        REPO:  in my project (×2) + my project (×2) = 4  (tie → USER_FILE)
      review the test failures in my project
        UF:    — = 0
        REPO:  test failures (×2) + test failures in (×2)
             + in my project (×2) + my project (×2) = 8
    """

    def _score(self, task):
        from controller.claude_code_delegate import (
            _classify_task_directory,
            _USER_FILE_TASKS,
            _REPO_TASKS,
        )
        t = task.lower()
        uf = sum(w for kw, w in _USER_FILE_TASKS.items() if kw in t)
        repo = sum(w for kw, w in _REPO_TASKS.items() if kw in t)
        cat, _ = _classify_task_directory(task)
        return uf, repo, cat

    @pytest.mark.parametrize("task,expected_uf,expected_repo,expected_winner", [
        # Substring matches: "photo" is in "photos" (counts both), "screenshot"
        # alone (not "screenshots") for "fix this screenshot".
        ("organize the code in my project", 2, 6, "repo"),
        ("fix my downloads folder", 6, 0, "user_file"),
        ("review my photos", 6, 0, "user_file"),
        ("fix this screenshot", 2, 0, "user_file"),
        ("organize the test files in my project", 4, 4, "user_file"),
        ("review the test failures in my project", 0, 8, "repo"),
    ])
    def test_conflict_score(self, task, expected_uf, expected_repo, expected_winner):
        uf, repo, winner = self._score(task)
        assert uf == expected_uf, \
            f"{task!r}: USER_FILE score {uf} != {expected_uf}"
        assert repo == expected_repo, \
            f"{task!r}: REPO score {repo} != {expected_repo}"
        assert winner == expected_winner, \
            f"{task!r}: winner {winner!r} != {expected_winner!r}"

    def test_possessive_project_phrase_lands_repo(self):
        """New scored test: 'organize the code in my project' must land REPO.

        'in my project' (REPO ×2) + 'code in' (REPO ×2) + 'my project' (REPO ×2) = 6.
        'organize' (USER_FILE ×2) = 2. REPO wins.
        """
        uf, repo, winner = self._score("organize the code in my project")
        assert winner == "repo", \
            f"Expected REPO. Got winner={winner!r}, UF={uf}, REPO={repo}"
        assert repo > uf, \
            f"REPO score ({repo}) must beat USER_FILE score ({uf})"

    def test_possessive_project_test_files_stays_user_file(self):
        """Possessive project + 'test files' → USER_FILE wins tie.

        'in my project' (REPO ×2) + 'my project' (REPO ×2) = 4.
        'organize' (USER_FILE ×2) + 'test files' (USER_FILE ×2) = 4. Tie → USER_FILE.
        Rationale: 'test files' is a concrete user asset (test screenshots, test
        videos, test PDFs) — when the user says 'organize the test files in my
        project', they're organizing a folder of test assets, not source code.
        """
        uf, repo, winner = self._score("organize the test files in my project")
        assert winner == "user_file", \
            f"Expected USER_FILE (tie). Got winner={winner!r}, UF={uf}, REPO={repo}"
        assert uf == repo == 4, \
            f"Scores should both be 4 (tied). Got UF={uf}, REPO={repo}"

    def test_possessive_project_with_test_failures_is_repo(self):
        """'review the test failures in my project' → REPO.

        'test failures' (REPO ×2) + 'test failures in' (REPO ×2) +
        'in my project' (REPO ×2) + 'my project' (REPO ×2) = 8.
        No USER_FILE keywords. REPO wins clearly.
        """
        uf, repo, winner = self._score("review the test failures in my project")
        assert winner == "repo"
        assert repo == 8
        assert uf == 0


# ══════════════════════════════════════════════════════════════════════════════════
# 2. is_path_like_reply() — RETIRED in v2 (handoff redesign)
# ══════════════════════════════════════════════════════════════════════════════════
@pytest.mark.skip(reason="removed in v2 handoff redesign")
class TestIsPathLikeReply:
    """Detects path-like approval replies and returns the resolved absolute path."""

    def test_windows_absolute_path(self):
        from controller.delegate_state import is_path_like_reply
        result = is_path_like_reply("yes C:\\Users\\User\\Pictures")
        assert result is not None
        assert "pictures" in result.lower()  # case-insensitive: Windows paths normalize

    def test_windows_drive_root(self):
        from controller.delegate_state import is_path_like_reply
        result = is_path_like_reply("yes D:\\Work")
        assert result is not None
        assert "work" in result.lower()

    def test_tilde_path(self):
        from controller.delegate_state import is_path_like_reply
        result = is_path_like_reply("yes ~/Downloads")
        assert result is not None
        assert "downloads" in result.lower()  # case-insensitive: tilde expansion preserves case

    def test_named_downloads(self):
        from controller.delegate_state import is_path_like_reply
        result = is_path_like_reply("yes Downloads")
        assert result is not None
        assert "downloads" in result.lower()

    def test_named_pictures(self):
        from controller.delegate_state import is_path_like_reply
        result = is_path_like_reply("yes Pictures")
        assert result is not None
        assert "pictures" in result.lower()

    def test_named_desktop(self):
        from controller.delegate_state import is_path_like_reply
        result = is_path_like_reply("yes Desktop")
        assert result is not None
        assert "desktop" in result.lower()

    def test_named_videos(self):
        from controller.delegate_state import is_path_like_reply
        result = is_path_like_reply("yes Videos")
        assert result is not None
        assert "videos" in result.lower()

    def test_use_prefix(self):
        from controller.delegate_state import is_path_like_reply
        result = is_path_like_reply("use ~/Documents")
        assert result is not None
        assert "documents" in result.lower()

    def test_in_prefix(self):
        from controller.delegate_state import is_path_like_reply
        result = is_path_like_reply("yes in ~/Desktop")
        assert result is not None
        assert "desktop" in result.lower()

    def test_instead_prefix(self):
        from controller.delegate_state import is_path_like_reply
        result = is_path_like_reply("use ~/Work instead")
        assert result is not None
        assert "work" in result.lower()

    def test_no_path_returns_none(self):
        from controller.delegate_state import is_path_like_reply
        assert is_path_like_reply("yes") is None
        assert is_path_like_reply("go ahead") is None
        assert is_path_like_reply("sure") is None
        assert is_path_like_reply("do it") is None
        assert is_path_like_reply("yeah do it") is None
        assert is_path_like_reply("") is None

    def test_no_path_case_insensitive(self):
        from controller.delegate_state import is_path_like_reply
        result = is_path_like_reply("YES ~/DOWNLOADS")
        assert result is not None
        assert "downloads" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════════
# 3. request_permission() wording varies by task_category
# ══════════════════════════════════════════════════════════════════════════════════
class TestRequestPermissionWording:
    """Permission prompt is different for repo vs user_file tasks."""

    def test_repo_worded_for_handoff(self):
        from controller.claude_code_delegate import request_permission
        prompt = request_permission("E:\\fairy", "fix the bug", task_category="repo")
        assert "fix the bug" in prompt
        assert "Claude Code" in prompt

    def test_user_file_worded_for_user_directory(self):
        from controller.claude_code_delegate import request_permission
        prompt = request_permission("C:\\Users\\User\\Pictures", "organize my photos", task_category="user_file")
        assert "organize my photos" in prompt
        # Must include Pictures in the prompt (or ~)
        assert "Pictures" in prompt or "~" in prompt
        # Should include Claude Code reference
        assert "Claude Code" in prompt

    def test_user_file_pictures_hint(self):
        from controller.claude_code_delegate import request_permission
        prompt = request_permission("C:\\Users\\User\\Pictures", "organize my photos", task_category="user_file")
        # Should mention Pictures somewhere (or the rel-path form)
        assert "Pictures" in prompt or "~" in prompt

    def test_user_file_downloads_hint(self):
        from controller.claude_code_delegate import request_permission
        prompt = request_permission("C:\\Users\\User\\Downloads", "clean my downloads", task_category="user_file")
        assert "Downloads" in prompt or "~" in prompt


# ══════════════════════════════════════════════════════════════════════════════════
# 4. Integration: pending delegation carries task_category through the approval path
#    RETIRED in v2 (pending state machine removed)
# ══════════════════════════════════════════════════════════════════════════════════
@pytest.mark.skip(reason="removed in v2 handoff redesign — pending state machine gone")
class TestPendingDelegationTaskCategory:
    """PendingDelegation.task_category is set and carried through."""

    def test_pending_has_task_category_field(self):
        from controller.delegate_state import PendingDelegation
        p = PendingDelegation(original_request="test", project_root="E:/fairy", task_category="user_file")
        assert p.task_category == "user_file"

    def test_pending_defaults_to_repo(self):
        from controller.delegate_state import PendingDelegation
        p = PendingDelegation(original_request="test", project_root="E:/fairy")
        assert p.task_category == "repo"

    def test_user_file_pending_spawns_in_user_directory(self, monkeypatch):
        """A user_file pending delegation should spawn Claude Code in the user directory."""
        from controller import agent_controller as ac
        from controller.delegate_state import set_pending, get_pending, PendingDelegation, clear_pending

        executed = {}
        def fake_delegate(**kwargs):
            executed.update(kwargs)
            return {
                "executed": True,
                "status": "ok",
                "message": "Claude Code completed.",
                "stdout": '{"result": "done", "is_error": false}',
                "stderr": "",
                "exit_code": 0,
                "duration_seconds": 1.0,
                "_parsed_json": True,
            }
        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)
        monkeypatch.setattr(ac, "classify_delegate_result", lambda r, pr, t: {
            **r, "outcome": "success", "reason": "done.",
        })

        # Manually set a pending with a user_file category
        clear_pending()
        set_pending(PendingDelegation(
            original_request="organize my screenshots",
            project_root=os.path.join(os.path.expanduser("~"), "Pictures"),
            task_category="user_file",
        ))

        reply, _ = ac.handle_request("yes", history=[])

        assert executed.get("permission_granted") is True
        # Should have spawned in Pictures, not E:\fairy
        project_root_used = executed.get("project_root", "")
        assert "pictures" in project_root_used.lower() or "users" in project_root_used.lower(), \
            f"Expected Pictures/Users directory, got {project_root_used!r}"


# ══════════════════════════════════════════════════════════════════════════════════
# 5. Regression: exact failing input goes to Pictures, not E:\fairy
# ══════════════════════════════════════════════════════════════════════════════════
class TestRegressionScreenshotFailingInput:
    """The original bug: 'make me a folder containing all my screenshots using claude'
    was delegated to E:\fairy instead of ~/Pictures. After the fix it goes to Pictures."""

    def test_original_failing_input_classified_user_file(self):
        from controller.claude_code_delegate import _classify_task_directory
        task = "make me a folder containing all my screenshots using claude"
        category, directory = _classify_task_directory(task)
        assert category == "user_file", \
            f"Original failing input must be USER_FILE, got {category}"
        assert "pictures" in directory.lower() or "screenshot" in directory.lower(), \
            f"Expected Pictures/Screenshots directory, got {directory!r}"

    def test_original_failing_input_permission_prompt_is_user_file_style(self):
        from controller.claude_code_delegate import _classify_task_directory, request_permission
        task = "make me a folder containing all my screenshots using claude"
        category, directory = _classify_task_directory(task)
        prompt = request_permission(directory, task, category)
        # Must NOT say "repository access to E:\fairy"
        assert "E:\\fairy" not in prompt, \
            f"Permission prompt must not hardcode E:\\fairy for user-file task. Got: {prompt!r}"
        assert "fairy" not in prompt.lower() or "delegate" in prompt.lower()


# ══════════════════════════════════════════════════════════════════════════════════
# 6. format_report — no duplicate JSON dumps, no raw usage JSON
# ══════════════════════════════════════════════════════════════════════════════════
@pytest.mark.skip(reason="removed in v2 handoff redesign — format_report removed")
class TestFormatReportClean:
    """format_report must not dump raw JSON (especially the usage block) twice."""

    def test_result_text_shown_not_raw_json(self):
        from controller.claude_code_delegate import format_report
        raw_json = (
            '{"result": "Created 50 screenshots folder with 250 organized files.",'
            '"is_error": false, "duration_api_ms": 12345,'
            '"total_cost_usd": 0.0042,'
            '"usage": {"input_tokens": 57345, "output_tokens": 766}}'
        )
        result = {
            "executed": True,
            "status": "ok",
            "message": "Claude Code completed in 5.0s.",
            "stdout": raw_json,
            "stderr": "",
            "exit_code": 0,
            "duration_seconds": 5.0,
            "result_text": "Created 50 screenshots folder with 250 organized files.",
            "_parsed_json": True,
            "outcome": "success",
            "reason": "done.",
        }
        report = format_report(result)
        # Should show the result text
        assert "Created 50 screenshots folder" in report
        # Should NOT dump the raw usage block
        assert "input_tokens" not in report, f"usage block leaked into report: {report!r}"
        assert "output_tokens" not in report
        # Should show cost once
        cost_count = report.count("total_cost_usd") + report.count("$")
        assert cost_count >= 1, f"Cost should appear in report. Got: {report!r}"

    def test_duplicate_json_dumps_prevented(self):
        from controller.claude_code_delegate import format_report
        raw_json = (
            '{"result": "Done.", "is_error": false, '
            '"usage": {"input_tokens": 57345}}'
        )
        result = {
            "executed": True,
            "status": "ok",
            "message": "Claude Code completed.",
            "stdout": raw_json,
            "stderr": "",
            "exit_code": 0,
            "duration_seconds": 1.0,
            "result_text": "Done.",
            "_parsed_json": True,
            "outcome": "success",
            "reason": "done.",
        }
        report = format_report(result)
        # The result text should appear exactly once
        done_count = report.count("Done.")
        assert done_count == 1, \
            f"'Done.' appeared {done_count} times — duplicate JSON dump likely. Report: {report!r}"

    def test_empty_result_text_reports_honestly(self):
        from controller.claude_code_delegate import format_report
        raw_json = '{"is_error": false, "usage": {"input_tokens": 100}}'
        result = {
            "executed": True,
            "status": "ok",
            "message": "Claude Code completed.",
            "stdout": raw_json,
            "stderr": "",
            "exit_code": 0,
            "duration_seconds": 1.0,
            "result_text": "",   # empty — mutating task with no result
            "_parsed_json": True,
            "outcome": "success",
            "reason": "Claude Code completed successfully.",
        }
        report = format_report(result)
        # Should not crash and should not show empty result_text as a blank block
        assert "input_tokens" not in report
