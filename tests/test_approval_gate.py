#!/usr/bin/env python3
"""
Tests for controller/approval_gate.py.

Covers:
  - classify_path(): new file / overwrite / load-bearing tiers
  - _parse_terminal_decision(): all recognized inputs
  - build_terminal_prompt(): tier-specific wording
  - request_approval_terminal(): approve / deny / alt / interrupt flows
  - _redirect_to_alt(): path redirection to fairy_outputs/
  - append_audit(): log format and non-crashing
  - Gate integration with _dispatch_tool: write_file fires gate,
    denial returns error dict, approval proceeds to execute
  - Discord mode raises ApprovalRequired (not terminal prompt)
  - Load-bearing path detection (repo root, main.py, controller/, etc.)
  - Git diff preview for tracked files
  - Git-tracked detection
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the approval_gate module is importable (it lives in controller/).
# conftest.py blocks real HTTP; that's fine — we don't need network here.
_controller_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


class TestClassifyPath:
    """Unit tests for PathTier classification — no I/O beyond os.stat."""

    def _import_gate(self):
        # Fresh import so each test gets a clean module.
        if "controller.approval_gate" in sys.modules:
            del sys.modules["controller.approval_gate"]
        sys.path.insert(0, _controller_dir)
        from controller import approval_gate
        return approval_gate

    def test_new_file_tier(self):
        ag = self._import_gate()
        # A path under a non-existent directory (no file there yet).
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "brand_new_file.txt")
            tier = ag.classify_path(target)
            assert not tier.is_overwrite
            assert not tier.is_load_bearing
            assert tier.tier == "new_file"
            assert "fairy_outputs" in tier.suggested_alt

    def test_overwrite_tier(self):
        ag = self._import_gate()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.txt"
            path.write_text("hello")
            tier = ag.classify_path(str(path))
            assert tier.is_overwrite
            assert not tier.is_load_bearing
            assert tier.tier == "overwrite"
            # suggested_alt is only suppressed for load-bearing paths,
            # not for overwrites (redirecting an overwrite to fairy_outputs
            # is still useful).
            assert tier.suggested_alt is not None

    def test_load_bearing_repo_root(self):
        ag = self._import_gate()
        # Bare name in project root — "main.py" without a subdirectory.
        # We can't use a real repo path here, so test the logic with a
        # path that matches the pattern. The actual rule fires when
        # _is_under_project is True. We mock the PROJECT_DIR to test.
        with tempfile.TemporaryDirectory() as tmp:
            # Create a "repo root" structure
            root = Path(tmp)
            # Simulate "existing.py" in repo root (no subdir = repo-root pattern)
            target = root / "random.py"
            target.write_text("x")
            with patch.object(ag, "_PROJECT_DIR", str(root)):
                tier = ag.classify_path(str(target))
            assert tier.is_load_bearing
            assert tier.tier == "load_bearing"

    def test_main_py_load_bearing(self):
        ag = self._import_gate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subdir = root / "controller"
            subdir.mkdir()
            main = subdir / "main.py"
            main.write_text("x")
            with patch.object(ag, "_PROJECT_DIR", str(root)):
                tier = ag.classify_path(str(main))
            assert tier.is_load_bearing
            assert tier.load_bearing_reason == "repo entry point main.py"

    def test_controller_module_load_bearing(self):
        ag = self._import_gate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctrl = root / "controller" / "agent_controller.py"
            ctrl.parent.mkdir()
            ctrl.write_text("x")
            with patch.object(ag, "_PROJECT_DIR", str(root)):
                tier = ag.classify_path(str(ctrl))
            assert tier.is_load_bearing
            assert tier.load_bearing_reason == "controller module"

    def test_init_py_load_bearing(self):
        ag = self._import_gate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init = root / "mypkg" / "__init__.py"
            init.parent.mkdir()
            init.write_text("x")
            with patch.object(ag, "_PROJECT_DIR", str(root)):
                tier = ag.classify_path(str(init))
            assert tier.is_load_bearing

    def test_gitignore_load_bearing(self):
        ag = self._import_gate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Place .gitignore in a subdirectory (not the repo root)
            # so the "repo root write" rule doesn't fire first.
            sub = root / "config"
            sub.mkdir()
            gi = sub / ".gitignore"
            gi.write_text("*.pyc")
            with patch.object(ag, "_PROJECT_DIR", str(root)):
                tier = ag.classify_path(str(gi))
            assert tier.is_load_bearing
            assert tier.load_bearing_reason == "git ignore file"

    def test_outside_project_not_load_bearing(self):
        ag = self._import_gate()
        # A path that looks like main.py but lives outside the mocked project root.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Subdirectory under root = NOT a bare repo-root write
            other = root / "controller" / "main.py"
            other.parent.mkdir()
            other.write_text("x")
            with patch.object(ag, "_PROJECT_DIR", str(root / "fairy_project")):
                tier = ag.classify_path(str(other))
            # Not under project, so load-bearing rules don't fire
            assert not tier.is_load_bearing

    def test_empty_path(self):
        ag = self._import_gate()
        tier = ag.classify_path("")
        assert not tier.is_overwrite
        assert not tier.is_load_bearing
        assert tier.tier == "new_file"


class TestParseTerminalDecision:
    def _import_gate(self):
        if "controller.approval_gate" in sys.modules:
            del sys.modules["controller.approval_gate"]
        sys.path.insert(0, _controller_dir)
        from controller import approval_gate
        return approval_gate

    def test_approve_variants(self):
        ag = self._import_gate()
        for inp in ("allow", "Allow", "✅", "yes", "Yes", "y", "ok", "sure", "do it"):
            assert ag._parse_terminal_decision(inp) == ag.OUTCOME_APPROVED, f"failed on {inp!r}"

    def test_deny_variants(self):
        ag = self._import_gate()
        for inp in ("deny", "DENY", "❌", "no", "No", "n", "nope", "nah"):
            assert ag._parse_terminal_decision(inp) == ag.OUTCOME_DENIED, f"failed on {inp!r}"

    def test_alt_variants(self):
        ag = self._import_gate()
        for inp in ("alt", "ALT", "redirect", "fairy_outputs"):
            assert ag._parse_terminal_decision(inp) == "alt", f"failed on {inp!r}"

    def test_unknown_returns_unknown(self):
        ag = self._import_gate()
        for inp in ("maybe", "", "  ", "yes please"):
            assert ag._parse_terminal_decision(inp) == "unknown", f"failed on {inp!r}"


class TestBuildTerminalPrompt:
    def _import_gate(self):
        if "controller.approval_gate" in sys.modules:
            del sys.modules["controller.approval_gate"]
        sys.path.insert(0, _controller_dir)
        from controller import approval_gate
        return approval_gate

    def _fake_tier(self, is_lb=False, is_overwrite=False, suggested=None):
        from controller import approval_gate as _ag
        return _ag.PathTier(
            raw_path="/tmp/test.txt",
            normalized="/tmp/test.txt",
            is_load_bearing=is_lb,
            load_bearing_reason="test reason" if is_lb else None,
            is_overwrite=is_overwrite,
            is_directory=False,
            suggested_alt=suggested,
        )

    def test_new_file_no_warning(self):
        ag = self._import_gate()
        tier = self._fake_tier(is_lb=False, is_overwrite=False)
        prompt = ag.build_terminal_prompt(
            user_name="Master", tool="write_file",
            args={"path": "/tmp/test.txt", "content": "hello"},
            tier=tier,
        )
        assert "⚠" not in prompt
        assert "allow" in prompt.lower()
        assert "deny" in prompt.lower()

    def test_overwrite_warning(self):
        ag = self._import_gate()
        tier = self._fake_tier(is_lb=False, is_overwrite=True)
        prompt = ag.build_terminal_prompt(
            user_name="Master", tool="write_file",
            args={"path": "/tmp/test.txt", "content": "overwrite!"},
            tier=tier,
        )
        assert "⚠" in prompt
        assert "OVERWRITE" in prompt

    def test_load_bearing_warning(self):
        ag = self._import_gate()
        tier = self._fake_tier(is_lb=True, is_overwrite=True)
        with patch.object(ag, "_PROJECT_DIR", "/tmp/fairy"):
            with patch.object(ag, "_git_diff_preview", lambda *a, **kw: ""):
                with patch.object(ag, "_git_tracked", lambda *a, **kw: False):
                    prompt = ag.build_terminal_prompt(
                        user_name="Master", tool="write_file",
                        args={"path": "/tmp/fairy/main.py", "content": "overwrite!"},
                        tier=tier,
                    )
        assert "⚠" in prompt
        assert "load-bearing" in prompt.lower()
        assert "test reason" in prompt  # load_bearing_reason shown

    def test_alt_suggestion_shown_for_new_file(self):
        ag = self._import_gate()
        tier = self._fake_tier(is_lb=False, is_overwrite=False, suggested="/fairy/fairy_outputs/test.txt")
        prompt = ag.build_terminal_prompt(
            user_name="Master", tool="write_file",
            args={"path": "/fairy/something.txt", "content": "hi"},
            tier=tier,
        )
        assert "tip" in prompt.lower() or "fairy_outputs" in prompt.lower()


class TestAppendAudit:
    def _import_gate(self):
        if "controller.approval_gate" in sys.modules:
            del sys.modules["controller.approval_gate"]
        sys.path.insert(0, _controller_dir)
        from controller import approval_gate
        return approval_gate

    def test_append_audit_writes_json_line(self):
        ag = self._import_gate()
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_path = f.name
        try:
            with patch.object(ag, "AUDIT_LOG_PATH", log_path):
                ag.append_audit(
                    user_name="Master",
                    user_id="terminal",
                    tool="write_file",
                    args={"path": "/tmp/x.txt"},
                    outcome="approved",
                    tier="new_file",
                    path="/tmp/x.txt",
                )
            with open(log_path) as f:
                line = f.read().strip()
            record = json.loads(line)
            assert record["outcome"] == "approved"
            assert record["user"] == "Master"
            assert record["tool"] == "write_file"
            assert record["tier"] == "new_file"
            assert record["path"] == "/tmp/x.txt"
        finally:
            os.unlink(log_path)

    def test_append_audit_does_not_raise(self):
        ag = self._import_gate()
        # A log path that doesn't exist and can't be created should not
        # raise into the caller's flow.
        with patch.object(ag, "AUDIT_LOG_PATH", "/nonexistent/path/that/cant/be/created/audit.log"):
            ag.append_audit(
                user_name="Master", user_id="terminal",
                tool="write_file", args={},
                outcome="approved",
            )  # must not raise


class TestRedirectToAlt:
    def _import_gate(self):
        if "controller.approval_gate" in sys.modules:
            del sys.modules["controller.approval_gate"]
        sys.path.insert(0, _controller_dir)
        from controller import approval_gate
        return approval_gate

    def test_path_redirected(self):
        ag = self._import_gate()
        with tempfile.TemporaryDirectory() as tmp:
            alt = os.path.join(tmp, "fairy_outputs", "test.txt")
            tier = ag.PathTier(
                raw_path="/dangerous.txt", normalized="/dangerous.txt",
                is_load_bearing=False, load_bearing_reason=None,
                is_overwrite=False, is_directory=False, suggested_alt=alt,
            )
            args = {"path": "/dangerous.txt", "content": "hello"}
            result = ag._redirect_to_alt(args, tier)
            assert result["path"] == alt
            assert result["content"] == "hello"

    def test_src_redirected_for_move(self):
        ag = self._import_gate()
        with tempfile.TemporaryDirectory() as tmp:
            alt = os.path.join(tmp, "fairy_outputs", "old.txt")
            tier = ag.PathTier(
                raw_path="/old.txt", normalized="/old.txt",
                is_load_bearing=False, load_bearing_reason=None,
                is_overwrite=False, is_directory=False, suggested_alt=alt,
            )
            args = {"src": "/old.txt", "dst": "/new.txt"}
            result = ag._redirect_to_alt(args, tier)
            assert result["src"] == alt


class TestRequestApprovalTerminal:
    """Test the terminal approval flow with mocked stdin/stdout."""

    def _import_gate(self):
        if "controller.approval_gate" in sys.modules:
            del sys.modules["controller.approval_gate"]
        sys.path.insert(0, _controller_dir)
        from controller import approval_gate
        return approval_gate

    def test_approved_returns_approved(self):
        ag = self._import_gate()
        output_lines = []
        outcome, args = ag.request_approval_terminal(
            user_name="Master", tool="write_file",
            args={"path": "/tmp/ok.txt", "content": "hello"},
            input_fn=lambda _: "allow",
            output_fn=lambda l: output_lines.append(l),
        )
        assert outcome == ag.OUTCOME_APPROVED
        assert args["path"] == "/tmp/ok.txt"
        assert any("allow" in l.lower() for l in output_lines)

    def test_denied_returns_denied(self):
        ag = self._import_gate()
        outcome, args = ag.request_approval_terminal(
            user_name="Master", tool="write_file",
            args={"path": "/tmp/ok.txt", "content": "hello"},
            input_fn=lambda _: "deny",
            output_fn=lambda _: None,
        )
        assert outcome == ag.OUTCOME_DENIED
        assert args["content"] == "hello"  # args returned unchanged

    def test_alt_redirects_path(self):
        ag = self._import_gate()
        with tempfile.TemporaryDirectory() as tmp:
            alt = os.path.join(tmp, "fairy_outputs", "ok.txt")
            # Use a fake project dir so suggested_alt is populated.
            fake_project = tmp
            with patch.object(ag, "_PROJECT_DIR", fake_project):
                with patch.object(ag, "_suggest_alt_path", lambda p: alt):
                    tier = ag.PathTier(
                        raw_path="/tmp/ok.txt", normalized="/tmp/ok.txt",
                        is_load_bearing=False, load_bearing_reason=None,
                        is_overwrite=False, is_directory=False, suggested_alt=alt,
                    )
                    with patch.object(ag, "classify_path", lambda p: tier):
                        outcome, args = ag.request_approval_terminal(
                            user_name="Master", tool="write_file",
                            args={"path": "/tmp/ok.txt", "content": "hello"},
                            input_fn=lambda _: "alt",
                            output_fn=lambda _: None,
                        )
        assert outcome == ag.OUTCOME_APPROVED
        assert args["path"] == alt

    def test_ctrl_c_denied(self):
        ag = self._import_gate()
        outcome, args = ag.request_approval_terminal(
            user_name="Master", tool="write_file",
            args={"path": "/tmp/ok.txt", "content": "hello"},
            input_fn=lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
            output_fn=lambda _: None,
        )
        assert outcome == ag.OUTCOME_DENIED

    def test_unknown_is_denied(self):
        ag = self._import_gate()
        outcome, _ = ag.request_approval_terminal(
            user_name="Master", tool="write_file",
            args={"path": "/tmp/ok.txt", "content": "hello"},
            input_fn=lambda _: "what do you think",
            output_fn=lambda _: None,
        )
        assert outcome == ag.OUTCOME_DENIED

    def test_audit_logged_on_request(self):
        ag = self._import_gate()
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            log_path = f.name
        try:
            with patch.object(ag, "AUDIT_LOG_PATH", log_path):
                with patch.object(ag, "classify_path", lambda p: ag.PathTier(
                    raw_path=p, normalized=p,
                    is_load_bearing=False, load_bearing_reason=None,
                    is_overwrite=False, is_directory=False, suggested_alt=None,
                )):
                    ag.request_approval_terminal(
                        user_name="Master", tool="write_file",
                        args={"path": "/tmp/ok.txt", "content": "hello"},
                        input_fn=lambda _: "deny",
                        output_fn=lambda _: None,
                    )
            with open(log_path) as f:
                lines = [json.loads(l) for l in f]
            # First entry is the "requested" notification, second is the
            # "denied" decision.
            assert len(lines) == 2
            assert lines[0]["outcome"] == "requested"
            assert lines[1]["outcome"] == "denied"
        finally:
            os.unlink(log_path)


class TestDispatchGateIntegration:
    """Integration: wire the gate into a fresh agent_controller and call."""

    def test_write_file_denied_blocks_execution(self):
        # Ensure the filesystem is NOT touched when denied.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "denied_file.txt"
            assert not target.exists()

            # Patch the gate's terminal approval to deny. The function is
            # called from agent_controller._dispatch_tool; monkeypatching
            # the function directly is cleaner than reaching into
            # Python's `input` built-in.
            from controller import agent_controller as ac
            from controller import approval_gate as ag

            def _deny(**kw):
                return ag.OUTCOME_DENIED, dict(kw.get("args", {}))

            with patch.object(ag, "request_approval_terminal", side_effect=_deny):
                result = ac._dispatch_tool("write_file", {
                    "path": str(target),
                    "content": "should not be written",
                    "allow_outside_project": True,
                })

            assert not target.exists()
            assert isinstance(result, dict)
            assert result.get("gate") == "denied" or result.get("error", "").startswith("denied by approval gate")

    def test_write_file_approved_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "approved_file.txt"
            assert not target.exists()

            from controller import agent_controller as ac
            from controller import approval_gate as ag

            def _approve(**kw):
                return ag.OUTCOME_APPROVED, dict(kw.get("args", {}))

            with patch.object(ag, "request_approval_terminal", side_effect=_approve):
                result = ac._dispatch_tool("write_file", {
                    "path": str(target),
                    "content": "hello world",
                    "allow_outside_project": True,
                })

            assert target.exists()
            assert target.read_text(encoding="utf-8") == "hello world"
            assert isinstance(result, dict)
            assert result.get("ok") is True

    def test_unknown_tool_not_affected(self):
        from controller import agent_controller as ac
        # Make sure non-file-mutating tools still work.
        # _dispatch_tool returns an error for unknown tools.
        result = ac._dispatch_tool("not_a_real_tool_name_xyz", {})
        assert isinstance(result, dict)
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_mkdir_approved_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "newdir"
            assert not target.exists()

            from controller import agent_controller as ac
            from controller import approval_gate as ag

            def _approve(**kw):
                return ag.OUTCOME_APPROVED, dict(kw.get("args", {}))

            with patch.object(ag, "request_approval_terminal", side_effect=_approve):
                result = ac._dispatch_tool("mkdir", {
                    "path": str(target),
                    "allow_outside_project": True,
                })

            assert target.exists()
            assert target.is_dir()
            assert result.get("ok") is True

    def test_delete_approved_removes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "to_delete.txt"
            target.write_text("bye")
            assert target.exists()

            from controller import agent_controller as ac
            from controller import approval_gate as ag

            def _approve(**kw):
                return ag.OUTCOME_APPROVED, dict(kw.get("args", {}))

            with patch.object(ag, "request_approval_terminal", side_effect=_approve):
                result = ac._dispatch_tool("delete", {
                    "path": str(target),
                    "allow_outside_project": True,
                })

            assert not target.exists()
            assert result.get("ok") is True

    def test_path_outside_project_blocked(self):
        """Even with approval, a path outside the project root is rejected
        unless allow_outside_project=True. The gate doesn't get a chance
        to approve this because _execute_file_mutation does the sandbox
        check itself."""
        from controller import agent_controller as ac
        from controller import approval_gate as ag

        def _approve(**kw):
            return ag.OUTCOME_APPROVED, dict(kw.get("args", {}))

        with patch.object(ag, "request_approval_terminal", side_effect=_approve):
            # Try to write to a path outside the project root without
            # allow_outside_project. Should return ok=False with a
            # clear error.
            result = ac._dispatch_tool("write_file", {
                "path": "C:/Windows/System32/drivers/etc/hosts",
                "content": "evil",
            })
        assert isinstance(result, dict)
        assert result.get("ok") is False
        assert "outside" in str(result.get("error", "")).lower()


class TestDiscordModeRaisesApprovalRequired:
    """When Discord front-end is active, the gate raises ApprovalRequired
    instead of blocking on a terminal prompt."""

    def test_raises_when_discord_active(self):
        if "controller.approval_gate" in sys.modules:
            del sys.modules["controller.approval_gate"]
        sys.path.insert(0, _controller_dir)
        from controller import approval_gate as ag

        ag._DISCORD_ACTIVE = True
        try:
            with pytest.raises(ag.ApprovalRequired) as exc_info:
                ag.request_approval_discord(
                    user_name="Master", user_id=123,
                    tool="write_file", args={"path": "/tmp/x.txt"},
                )
            req = exc_info.value.request_info
            assert req["tool"] == "write_file"
            assert req["user_name"] == "Master"
        finally:
            ag._DISCORD_ACTIVE = False


class TestGitDiffPreview:
    def test_git_diff_returns_empty_on_bad_path(self):
        if "controller.approval_gate" in sys.modules:
            del sys.modules["controller.approval_gate"]
        sys.path.insert(0, _controller_dir)
        from controller import approval_gate as ag

        # No git repo → empty string.
        with tempfile.TemporaryDirectory() as tmp:
            result = ag._git_diff_preview(tmp, os.path.join(tmp, "x.txt"))
            assert result == ""

    def test_git_tracked_false_for_nonexistent(self):
        if "controller.approval_gate" in sys.modules:
            del sys.modules["controller.approval_gate"]
        sys.path.insert(0, _controller_dir)
        from controller import approval_gate as ag

        with tempfile.TemporaryDirectory() as tmp:
            result = ag._git_tracked(tmp, os.path.join(tmp, "nonexistent.py"))
            assert result is False


class TestFormatArgsSummary:
    def _import_gate(self):
        if "controller.approval_gate" in sys.modules:
            del sys.modules["controller.approval_gate"]
        sys.path.insert(0, _controller_dir)
        from controller import approval_gate
        return approval_gate

    def test_write_file_summary(self):
        ag = self._import_gate()
        s = ag._format_args_summary("write_file", {"path": "/tmp/x.txt", "content": "hello"})
        assert "/tmp/x.txt" in s
        assert "5 chars" in s  # len("hello")

    def test_mkdir_summary(self):
        ag = self._import_gate()
        s = ag._format_args_summary("mkdir", {"path": "/tmp/newdir"})
        assert "/tmp/newdir" in s

    def test_delete_summary(self):
        ag = self._import_gate()
        s = ag._format_args_summary("delete", {"path": "/tmp/old.txt"})
        assert "/tmp/old.txt" in s

    def test_move_summary(self):
        ag = self._import_gate()
        s = ag._format_args_summary("move", {"src": "/a", "dst": "/b"})
        assert "/a" in s
        assert "/b" in s

    def test_zip_create_summary(self):
        ag = self._import_gate()
        s = ag._format_args_summary("zip_create", {"archive": "/a.zip", "src": "/b"})
        assert "/a.zip" in s
        assert "/b" in s
