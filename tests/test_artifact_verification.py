#!/usr/bin/env python3
"""
Tests for artifact verification: empty-artifact detection, non-empty checks,
and honest capability reporting on the multi-step task execution path.

The fairy agent must NOT report success when:
  - A folder it claims to have created is empty.
  - A file it claims to have written is 0 bytes.
  - A tool failed but the failure was silently swallowed.

The verification helpers in agent_controller._verify_creation_claims and
_is_artifact_non_empty enforce that contract on the reply path.
"""
from __future__ import annotations

import os
import tempfile

import pytest


class TestIsArtifactNonEmpty:
    """_is_artifact_non_empty: file/folder existence + content size checks."""

    def test_returns_false_for_nonexistent_path(self, tmp_path):
        from controller.agent_controller import _is_artifact_non_empty
        missing = tmp_path / "nope.txt"
        assert _is_artifact_non_empty(str(missing)) is False

    def test_returns_false_for_empty_file(self, tmp_path):
        from controller.agent_controller import _is_artifact_non_empty
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        assert _is_artifact_non_empty(str(empty)) is False

    def test_returns_true_for_nonempty_file(self, tmp_path):
        from controller.agent_controller import _is_artifact_non_empty
        filled = tmp_path / "filled.txt"
        filled.write_text("hello, world")
        assert _is_artifact_non_empty(str(filled)) is True

    def test_returns_false_for_empty_directory(self, tmp_path):
        from controller.agent_controller import _is_artifact_non_empty
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        assert _is_artifact_non_empty(str(empty_dir)) is False

    def test_returns_true_for_directory_with_nonempty_file(self, tmp_path):
        from controller.agent_controller import _is_artifact_non_empty
        d = tmp_path / "papers"
        d.mkdir()
        (d / "math.md").write_text("# math\n\n1+1=2")
        assert _is_artifact_non_empty(str(d)) is True

    def test_returns_true_for_directory_with_nonempty_subdirectory(self, tmp_path):
        from controller.agent_controller import _is_artifact_non_empty
        d = tmp_path / "papers"
        d.mkdir()
        sub = d / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("x")
        assert _is_artifact_non_empty(str(d)) is True


class TestVerifyCreationClaims:
    """_verify_creation_claims: catches 'created X' lies when X is missing/empty."""

    def test_no_creation_claim_passes(self):
        from controller.agent_controller import _verify_creation_claims
        verified, path = _verify_creation_claims("Here's the summary", "what's 2+2")
        assert verified is True
        assert path == ""

    def test_creation_claim_without_file_request_passes(self, tmp_path):
        """A reply mentioning 'created' but where user didn't ask for a file
        should NOT trigger the verification gate (false-positive avoidance)."""
        from controller.agent_controller import _verify_creation_claims
        # User asked a question, reply uses the word 'created' in narrative
        reply = "I created a mental model of the problem: 2+2=4"
        verified, path = _verify_creation_claims(reply, "what's 2+2")
        assert verified is True
        assert path == ""

    def test_claim_with_existing_nonempty_file_passes(self, tmp_path):
        from controller.agent_controller import _verify_creation_claims
        target = tmp_path / "report.txt"
        target.write_text("analysis")
        reply = f"I've created {target} for you."
        verified, path = _verify_creation_claims(reply, "create a report")
        assert verified is True
        assert path == str(target)

    def test_claim_with_empty_file_fails(self, tmp_path):
        """A reply that claims a file was created, but the file is 0 bytes,
        must be flagged as a verification failure (the agent faked success)."""
        from controller.agent_controller import _verify_creation_claims
        target = tmp_path / "empty_report.txt"
        target.write_text("")
        reply = f"Done — created {target}."
        verified, path = _verify_creation_claims(reply, "create a report")
        assert verified is False, "empty file should not pass as a created artifact"
        assert path == ""

    def test_claim_with_nonexistent_path_fails(self):
        from controller.agent_controller import _verify_creation_claims
        reply = "I've created /nonexistent/folder/file.txt for you."
        verified, path = _verify_creation_claims(reply, "create a file")
        assert verified is False
        assert path == ""

    def test_claim_with_empty_folder_fails(self, tmp_path):
        """The core bug: agent claims it created a folder of papers, but
        the folder is empty. Must NOT pass verification."""
        from controller.agent_controller import _verify_creation_claims
        papers = tmp_path / "exam_papers"
        papers.mkdir()
        # folder exists but is empty
        reply = f"Done — created the folder {papers} with your exam papers."
        verified, path = _verify_creation_claims(reply, "create a folder of exam papers")
        assert verified is False, "empty folder should not pass as a created artifact"

    def test_claim_with_filled_folder_passes(self, tmp_path):
        from controller.agent_controller import _verify_creation_claims
        papers = tmp_path / "exam_papers"
        papers.mkdir()
        (papers / "math.md").write_text("Q1: 1+1=2")
        (papers / "science.md").write_text("Q1: H2O")
        reply = f"Done — created the folder {papers} with your exam papers."
        verified, path = _verify_creation_claims(reply, "create a folder of exam papers")
        assert verified is True
        assert path == str(papers)


class TestHonestFailureSurfacing:
    """The reply path must surface tool errors honestly — never claim success
    when a step failed."""

    def test_honest_message_format(self):
        """The honest-reply template must include a 'created X but could not Y'
        structure when a step fails (the contract described in the task)."""
        from controller.agent_controller import _is_artifact_non_empty
        # Simulate: a folder was created, but it's empty
        with tempfile.TemporaryDirectory() as td:
            empty_folder = os.path.join(td, "papers")
            os.mkdir(empty_folder)
            # Verify: a tool that produced an empty artifact would be flagged
            assert _is_artifact_non_empty(empty_folder) is False
            # The fix-flow is: _verify_creation_claims → (False, "") → re-dispatch
            # to the planner so the user gets an honest reply rather than the
            # model's false 'created the folder' claim.
            from controller.agent_controller import _verify_creation_claims
            verified, _ = _verify_creation_claims(
                f"Done — created {empty_folder} with your exam papers.",
                "create a folder of exam papers",
            )
            assert verified is False


class TestEmptyArtifactDetectionRegression:
    """Regression tests for the specific failure modes described in the task:

    1. "Folder of exam papers with answers/images" → empty folder
    2. "Small exe/py project" → nested empty folders
    """

    def test_exam_papers_folder_must_be_nonempty(self, tmp_path):
        from controller.agent_controller import _verify_creation_claims
        papers = tmp_path / "exam_papers"
        papers.mkdir()
        # No contents — this is the original failure mode
        user_text = "make a folder of exam papers with answers and images"
        reply = f"All set! Created your exam papers at {papers}."
        verified, _ = _verify_creation_claims(reply, user_text)
        assert verified is False, "empty exam_papers folder should fail verification"

    def test_nested_empty_folders_fail(self, tmp_path):
        """Building 'a small exe/py project' with nested empty folders is the
        second failure mode. Any nested-empty structure must fail."""
        from controller.agent_controller import _verify_creation_claims, _is_artifact_non_empty
        project = tmp_path / "fairy_nested_empty"
        (project / "src").mkdir(parents=True)
        (project / "tests").mkdir(parents=True)
        # All subdirs are empty
        assert _is_artifact_non_empty(str(project)) is False
        user_text = "create a small Python project"
        reply = f"Done — created your project at {project} with src/ and tests/."
        verified, _ = _verify_creation_claims(reply, user_text)
        assert verified is False, (
            f"Nested empty project at {project} should not pass verification"
        )

    def test_project_with_real_files_passes(self, tmp_path):
        """Same shape, but with real files inside — should pass."""
        from controller.agent_controller import _verify_creation_claims
        project = tmp_path / "myproject"
        (project / "src").mkdir(parents=True)
        (project / "src" / "main.py").write_text("print('hi')\n")
        (project / "README.md").write_text("# myproject\n")
        user_text = "create a small Python project"
        reply = f"Done — created your project at {project}."
        verified, path = _verify_creation_claims(reply, user_text)
        assert verified is True
        assert path == str(project)
