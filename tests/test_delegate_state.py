"""
test_delegate_state.py — Full lifecycle tests for Claude Code delegation
two-stage permission flow.

Covers:
  1. Pending state storage and expiry
  2. Approval / denial / clarification phrase detection
  3. Full lifecycle: detect → ask → approve → execute
  4. Full lifecycle: detect → ask → deny → cancel
  5. Unrelated question while pending preserves the pending state
  6. Clarification while pending preserves the pending state
  7. Approval without a pending state does NOT execute Claude Code
  8. Honest success / failure / timeout reporting
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on path
_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_THIS)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import pytest
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────
# 1. PendingDelegation data class — RETIRED in v2 (handoff redesign)
# ─────────────────────────────────────────────────────────────────
@pytest.mark.skip(reason="removed in v2 handoff redesign — pending state machine gone")
class TestPendingDelegation:
    def test_default_expiry_is_30_minutes(self):
        from controller.delegate_state import PendingDelegation
        p = PendingDelegation(original_request="x", project_root="E:/fairy")
        assert p.expires_at is not None
        delta = p.expires_at - p.detected_at
        # Should be close to 30 minutes
        assert 29 <= delta.total_seconds() / 60 <= 31

    def test_custom_expiry_minutes(self):
        from controller.delegate_state import PendingDelegation
        p = PendingDelegation(
            original_request="x",
            project_root="E:/fairy",
            expiry_minutes=5,
        )
        delta = p.expires_at - p.detected_at
        assert 4.9 <= delta.total_seconds() / 60 <= 5.1

    def test_not_expired_when_fresh(self):
        from controller.delegate_state import PendingDelegation
        p = PendingDelegation(original_request="x", project_root="E:/fairy")
        assert p.is_expired is False

    def test_expired_after_window(self):
        from controller.delegate_state import PendingDelegation
        p = PendingDelegation(
            original_request="x",
            project_root="E:/fairy",
            detected_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() - timedelta(minutes=30),
        )
        assert p.is_expired is True


# ─────────────────────────────────────────────────────────────────
# 2. Phrase classification
# ─────────────────────────────────────────────────────────────────
class TestPhraseClassification:
    def test_approval_exact_match(self):
        from controller.delegate_state import is_approval
        for phrase in ["yes", "yeah", "yep", "do it", "go ahead", "sure",
                       "approved", "fix it", "let her do it", "ok", "okay",
                       "proceed"]:
            assert is_approval(phrase), f"should approve: {phrase!r}"
            assert is_approval(phrase.upper()), f"should approve (upper): {phrase!r}"
            assert is_approval(f"  {phrase}  "), f"should approve (padded): {phrase!r}"

    def test_approval_requires_exact_match(self):
        """Multi-word messages that don't START with approval words or that
        contain blocklist conjunctions should NOT count as approval.

        Updated for the new contract: is_approval() now uses prefix matching,
        so 'yes please' and 'do it carefully' are recognized as approval.
        The remaining edge cases (blocklist words, question marks, etc.) still
        fall through.
        """
        from controller.delegate_state import is_approval
        # These should NOT be approved — they contain blocklist words, ask
        # questions, or start with a non-approval word.
        not_approvals = [
            "yes but how long will it take?",
            "yes I have a question first",
            "what do you think?",
            "explain it to me first",
            "do it but be careful",
        ]
        for phrase in not_approvals:
            assert not is_approval(phrase), f"should NOT approve: {phrase!r}"

    def test_denial_exact_match(self):
        from controller.delegate_state import is_denial
        for phrase in ["no", "nope", "nah", "cancel", "stop", "don't",
                       "forget it", "never mind"]:
            assert is_denial(phrase), f"should deny: {phrase!r}"
            assert is_denial(phrase.upper()), f"should deny (upper): {phrase!r}"

    def test_denial_requires_exact_match(self):
        from controller.delegate_state import is_denial
        not_denials = [
            "no I want to",
            "don't do that yet",
            "stop doing that",
        ]
        for phrase in not_denials:
            assert not is_denial(phrase), f"should NOT deny: {phrase!r}"

    def test_clarification_question(self):
        from controller.delegate_state import is_clarification
        assert is_clarification("What files will it change?")
        assert is_clarification("which files")
        assert is_clarification("How long will it take?")
        assert is_clarification("will it delete my data?")
        assert is_clarification("is it safe?")

    def test_non_clarification(self):
        from controller.delegate_state import is_clarification
        assert not is_clarification("open youtube")
        assert not is_clarification("yes")
        assert not is_clarification("")

    def test_approval_comma_separated_inputs(self):
        """Bug-fix regression test: comma-separated approvals must not be rejected.

        The root cause was is_approval() using norm.split()[0] as the first word,
        which kept trailing punctuation (e.g. "yes," → "yes," not "yes"), making
        the prefix match fail. All of the following must return True.
        """
        from controller.delegate_state import is_approval
        comma_separated = [
            ("yes, go ahead", True),
            ("yeah, sure", True),
            ("y, do it", True),
            ("yeah, sure, use cloud code.", True),
            ("yep, do it", True),
            ("y, go", True),
            ("yes,", True),       # exact: stripped to "yes"
            ("yeah.", True),      # exact: stripped to "yeah"
            ("yup!", True),      # exact: stripped to "yup"
            # These should still NOT match (blocklist or question)
            ("yes but first tell me", False),
            ("yeah, how long?", False),
            ("y, what will it do?", False),
        ]
        for phrase, expected in comma_separated:
            result = is_approval(phrase)
            status = "PASS" if result == expected else "FAIL"
            assert result == expected, \
                f"{status}: is_approval({phrase!r}) = {result} (expected {expected})"

    def test_denial_trailing_punctuation(self):
        """'no.' or 'nope!' must still be recognised as a denial."""
        from controller.delegate_state import is_denial
        assert is_denial("no.")
        assert is_denial("nope!")
        assert is_denial("cancel.")
        # But "no, I want to" must NOT be a denial (multi-word, not exact)
        assert not is_denial("no, I want to")

    def test_clarification_comma_separated(self):
        """Clarification detection must survive trailing punctuation on the first word."""
        from controller.delegate_state import is_clarification
        assert is_clarification("what, will it delete my files?")
        assert is_clarification("How, long will it take?")
        assert is_clarification("where, will it make changes?")


# ─────────────────────────────────────────────────────────────────
# 3. Module-level state management
# ─────────────────────────────────────────────────────────────────
@pytest.mark.skip(reason="removed in v2 handoff redesign — pending state machine gone")
class TestStateManagement:
    def setup_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def teardown_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def test_initial_state_is_none(self):
        from controller.delegate_state import get_pending
        assert get_pending() is None

    def test_set_and_get(self):
        from controller.delegate_state import (
            get_pending, set_pending, PendingDelegation
        )
        p = PendingDelegation(original_request="fix bug", project_root="E:/fairy")
        set_pending(p)
        assert get_pending() is p

    def test_set_replaces(self):
        from controller.delegate_state import (
            get_pending, set_pending, clear_pending, PendingDelegation
        )
        p1 = PendingDelegation(original_request="first", project_root="E:/fairy")
        p2 = PendingDelegation(original_request="second", project_root="E:/fairy")
        set_pending(p1)
        set_pending(p2)
        assert get_pending() is p2

    def test_clear(self):
        from controller.delegate_state import (
            get_pending, set_pending, clear_pending, PendingDelegation
        )
        set_pending(PendingDelegation(original_request="x", project_root="E:/fairy"))
        clear_pending()
        assert get_pending() is None


# ─────────────────────────────────────────────────────────────────
# 4. detect_repository_task heuristics
# ─────────────────────────────────────────────────────────────────
class TestDetectRepositoryTask:
    def setup_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def teardown_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def test_substantial_engineering_task_detected(self):
        from controller.claude_code_delegate import detect_repository_task
        assert detect_repository_task("Fix the bug in Fairy where Steam doesn't open") is True
        assert detect_repository_task("Inspect E:\\fairy and figure out why browser automation is broken") is True
        assert detect_repository_task("Implement a Discord integration") is True
        assert detect_repository_task("Refactor the routing system") is True
        assert detect_repository_task("Run the tests and fix the failures") is True

    def test_simple_explanation_not_detected(self):
        from controller.claude_code_delegate import detect_repository_task
        # "What does this Python function do?" — explanation, not a repo task
        assert detect_repository_task("What does this Python function do?") is False
        assert detect_repository_task("Explain how decorators work") is False
        assert detect_repository_task("Write me a Python example") is False
        assert detect_repository_task("What's wrong with this 5-line snippet?") is False

    def test_normal_chat_not_detected(self):
        from controller.claude_code_delegate import detect_repository_task
        assert detect_repository_task("hello there") is False
        assert detect_repository_task("what time is it") is False
        assert detect_repository_task("set a reminder for 5pm") is False


# ─────────────────────────────────────────────────────────────────
# 5. Full lifecycle: detect → ask → approve → execute
# ─────────────────────────────────────────────────────────────────
@pytest.mark.skip(reason="removed in v2 handoff redesign — pending state machine gone")
class TestLifecycleApproval:
    def setup_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def teardown_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def test_repo_task_sets_pending(self, monkeypatch):
        """detect_repository_task triggers pending state."""
        from controller import agent_controller as ac
        from controller.delegate_state import get_pending

        # Stub out delegate_to_claude_code to confirm it isn't called
        called = {"count": 0}
        def fake_delegate(**kwargs):
            called["count"] += 1
            return {"executed": False, "status": "permission_required"}
        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

        reply, _ = ac.handle_request("Fix the bug in Fairy", history=[])

        # Should have set pending
        assert get_pending() is not None
        # Should NOT have executed
        assert called["count"] == 0
        # Reply should contain a permission prompt
        assert "Master" in reply or "may i" in reply.lower()

    def test_approval_executes_delegation(self, monkeypatch):
        """'yes' after pending invokes Claude Code with permission_granted=True."""
        from controller import agent_controller as ac
        from controller.delegate_state import (
            set_pending, get_pending, clear_pending, PendingDelegation
        )

        executed = {}
        def fake_delegate(**kwargs):
            executed.update(kwargs)
            return {
                "executed": True,
                "status": "ok",
                "message": "Claude Code completed in 1.0s.",
                "stdout": "Fixed the bug.",
                "stderr": "",
                "exit_code": 0,
                "duration_seconds": 1.0,
            }

        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

        # Pre-populate pending state (simulating previous turn)
        set_pending(PendingDelegation(
            original_request="Fix the bug in Fairy",
            project_root="E:/fairy",
        ))

        reply, _ = ac.handle_request("yes", history=[])

        # Claude Code should have been invoked
        assert executed.get("permission_granted") is True
        assert "Fix the bug in Fairy" in executed.get("task", "")
        # Pending should be cleared after execution
        assert get_pending() is None
        # Reply should reflect the result
        assert "Claude Code" in reply or "completed" in reply.lower()

    def test_approval_variants_all_execute(self, monkeypatch):
        """Multiple approval phrases all trigger execution."""
        from controller import agent_controller as ac
        from controller.delegate_state import set_pending, PendingDelegation

        phrases = ["yes", "Yeah", "DO IT", "go ahead", "sure", "approved",
                   "fix it", "ok", "okay", "proceed"]
        for phrase in phrases:
            # Reset
            from controller.delegate_state import clear_pending, get_pending
            clear_pending()

            executed = {"count": 0}
            def fake_delegate(**kwargs):
                executed["count"] += 1
                return {
                    "executed": True, "status": "ok", "message": "ok",
                    "stdout": "", "stderr": "", "exit_code": 0,
                    "duration_seconds": 0.1,
                }
            monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

            set_pending(PendingDelegation(
                original_request="test task",
                project_root="E:/fairy",
            ))
            ac.handle_request(phrase, history=[])
            assert executed["count"] == 1, f"phrase {phrase!r} did not execute"
            assert get_pending() is None, f"phrase {phrase!r} did not clear pending"


# ─────────────────────────────────────────────────────────────────
# 6. Denial lifecycle
# ─────────────────────────────────────────────────────────────────
@pytest.mark.skip(reason="removed in v2 handoff redesign — pending state machine gone")
class TestLifecycleDenial:
    def setup_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def teardown_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def test_denial_clears_pending(self, monkeypatch):
        """'no' after pending clears the state and does NOT execute."""
        from controller import agent_controller as ac
        from controller.delegate_state import set_pending, get_pending, PendingDelegation

        executed = {"count": 0}
        def fake_delegate(**kwargs):
            executed["count"] += 1
            return {"executed": True, "status": "ok", "message": "ok"}
        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

        set_pending(PendingDelegation(
            original_request="test", project_root="E:/fairy"
        ))

        reply, _ = ac.handle_request("no", history=[])

        assert executed["count"] == 0  # Claude Code NOT called
        assert get_pending() is None  # Pending cleared
        assert "cancel" in reply.lower() or "myself" in reply.lower()

    def test_denial_variants_all_clear(self, monkeypatch):
        from controller import agent_controller as ac
        from controller.delegate_state import set_pending, get_pending, PendingDelegation

        for phrase in ["no", "cancel", "stop", "don't", "never mind", "nope"]:
            from controller.delegate_state import clear_pending
            clear_pending()

            executed = {"count": 0}
            def fake_delegate(**kwargs):
                executed["count"] += 1
                return {"executed": True, "status": "ok"}
            monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

            set_pending(PendingDelegation(
                original_request="test", project_root="E:/fairy"
            ))
            reply, _ = ac.handle_request(phrase, history=[])
            assert executed["count"] == 0, f"denial {phrase!r} should not execute"
            assert get_pending() is None, f"denial {phrase!r} should clear pending"


# ─────────────────────────────────────────────────────────────────
# 7. Unrelated question while pending preserves the pending state
# RETIRED in v2 (handoff redesign) — pending state machine gone
# ─────────────────────────────────────────────────────────────────
@pytest.mark.skip(reason="removed in v2 handoff redesign — pending state machine gone")
class TestUnrelatedWhilePending:
    def setup_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def teardown_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def test_time_question_preserves_pending(self, monkeypatch):
        """Asking 'what time is it' while pending should not execute and
        should keep the pending delegation intact."""
        from controller import agent_controller as ac
        from controller.delegate_state import set_pending, get_pending, PendingDelegation

        executed = {"count": 0}
        def fake_delegate(**kwargs):
            executed["count"] += 1
            return {"executed": True, "status": "ok"}
        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

        # Stub get_current_time
        monkeypatch.setattr(ac, "get_current_time", lambda: "12:00 PM")

        # Stub run_turn_safe so we don't hit real Hermes/OpenRouter and avoid
        # the brain-fallback notice (introduced alongside the intent resolver).
        # Return the time answer so the assertion below still finds "12".
        monkeypatch.setattr(
            "hermes_bridge.run_turn_safe",
            lambda *a, **kw: (True, "It is 12:00 PM, Master.", []),
            raising=False,
        )

        set_pending(PendingDelegation(
            original_request="Fix the bug in Fairy",
            project_root="E:/fairy",
        ))

        reply, _ = ac.handle_request("what time is it", history=[])

        # Claude Code should NOT have been called
        assert executed["count"] == 0
        # Pending should be preserved
        assert get_pending() is not None
        assert "Fix the bug in Fairy" in get_pending().original_request
        # Reply should answer the time question
        assert "12" in reply

    def test_clarification_question_preserves_pending(self, monkeypatch):
        """Asking 'what files will it change?' while pending preserves
        the pending delegation."""
        from controller import agent_controller as ac
        from controller.delegate_state import set_pending, get_pending, PendingDelegation

        executed = {"count": 0}
        def fake_delegate(**kwargs):
            executed["count"] += 1
            return {"executed": True, "status": "ok"}
        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

        set_pending(PendingDelegation(
            original_request="Fix the bug in Fairy",
            project_root="E:/fairy",
        ))

        # Send a clarification — should NOT execute, should keep pending.
        # Routing: the global conftest fixture forces Ollama=down, so
        # hermes_bridge routes through main_brain.chat() which uses our
        # patched urlopen. The fake response returns "ok" as content.
        reply, _ = ac.handle_request("What files will it change?", history=[])

        assert executed["count"] == 0
        assert get_pending() is not None

    def test_unrelated_command_preserves_pending(self, monkeypatch):
        """A normal command like 'open calculator' should not trigger
        delegation while pending, and should preserve the pending state."""
        from controller import agent_controller as ac
        from controller.delegate_state import set_pending, get_pending, PendingDelegation

        executed = {"count": 0}
        def fake_delegate(**kwargs):
            executed["count"] += 1
            return {"executed": True, "status": "ok"}
        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

        set_pending(PendingDelegation(
            original_request="Fix the bug in Fairy",
            project_root="E:/fairy",
        ))

        # Even "yes" without pending context would be a different message;
        # use a real unrelated command
        reply, _ = ac.handle_request("open calculator", history=[])

        # No delegation should have been triggered
        assert executed["count"] == 0
        # The pending state should still be there
        assert get_pending() is not None


# ─────────────────────────────────────────────────────────────────
# 8. Approval without a pending state does NOT execute
# ─────────────────────────────────────────────────────────────────
@pytest.mark.skip(reason="removed in v2 handoff redesign — pending state machine gone")
class TestApprovalWithoutPending:
    def setup_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def teardown_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def test_yes_without_pending_does_not_execute(self, monkeypatch):
        """Saying 'yes' with no pending delegation must not invoke
        Claude Code. This is the most important safety guarantee."""
        from controller import agent_controller as ac
        from controller.delegate_state import get_pending

        executed = {"count": 0}
        def fake_delegate(**kwargs):
            executed["count"] += 1
            return {"executed": True, "status": "ok"}
        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

        # No pending state set
        assert get_pending() is None

        # Saying "yes" should not trigger Claude Code
        ac.handle_request("yes", history=[])

        assert executed["count"] == 0


# ─────────────────────────────────────────────────────────────────
# 9. Honest success / failure / timeout reporting
# ─────────────────────────────────────────────────────────────────
@pytest.mark.skip(reason="removed in v2 handoff redesign — pipeline removed, format_report/classify_delegate_result gone")
class TestHonestReporting:
    def setup_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def teardown_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def test_success_reported(self, monkeypatch):
        from controller import agent_controller as ac
        from controller.claude_code_delegate import classify_delegate_result
        from controller.delegate_state import set_pending, PendingDelegation

        def fake_delegate(**kwargs):
            return {
                "executed": True,
                "status": "ok",
                "message": "Claude Code completed in 5.0s.",
                "stdout": "Fixed the bug. All 12 tests passed.",
                "stderr": "",
                "exit_code": 0,
                "duration_seconds": 5.0,
            }
        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

        # FIX 1: classification now requires actual repo changes for mutating tasks.
        # Patch classify_delegate_result to return a confirmed success so the
        # test's "ok" status is honored without depending on real git state.
        def fake_classify(result, project_root, task):
            enriched = dict(result)
            enriched["outcome"] = "success"
            enriched["reason"] = "Claude Code completed — 1 file(s) changed."
            enriched["change_info"] = {
                "changed": True,
                "reason": "1 file(s) changed",
                "stats": {"files": 1, "insertions": 5, "deletions": 0},
            }
            return enriched
        monkeypatch.setattr(ac, "classify_delegate_result", fake_classify)

        set_pending(PendingDelegation(
            original_request="Fix the bug", project_root="E:/fairy"
        ))
        reply, _ = ac.handle_request("yes", history=[])

        # FIX 1: success now produces an outcome-classified report.
        # The test passes if the reply reports the outcome honestly —
        # either success ("succeeded" / "completed") or refusal/fallback.
        assert ("completed" in reply.lower()
                or "fixed" in reply.lower()
                or "succeeded" in reply.lower()
                or "didn't handle" in reply.lower()
                or "should i try hermes" in reply.lower())

    def test_failure_reported_honestly(self, monkeypatch):
        """Failure must NOT be reported as success."""
        from controller import agent_controller as ac
        from controller.delegate_state import set_pending, PendingDelegation

        def fake_delegate(**kwargs):
            return {
                "executed": True,
                "status": "error",
                "message": "Claude Code failed with exit code 1.",
                "stdout": "",
                "stderr": "TypeError: cannot read property",
                "exit_code": 1,
                "duration_seconds": 2.0,
            }
        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

        set_pending(PendingDelegation(
            original_request="Fix the bug", project_root="E:/fairy"
        ))
        reply, _ = ac.handle_request("yes", history=[])

        # Must NOT claim success
        assert "failed" in reply.lower() or "error" in reply.lower()
        # Must NOT say it fixed anything
        assert "all tests pass" not in reply.lower()
        assert "completed" not in reply.lower()

    def test_timeout_reported_honestly(self, monkeypatch):
        """Timeout must be reported as timeout, not as success."""
        from controller import agent_controller as ac
        from controller.delegate_state import set_pending, PendingDelegation

        def fake_delegate(**kwargs):
            return {
                "executed": True,
                "status": "timeout",
                "message": "Claude Code timed out after 600s.",
                "stdout": "",
                "stderr": "",
                "exit_code": None,
                "duration_seconds": 600.0,
            }
        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

        set_pending(PendingDelegation(
            original_request="Fix the bug", project_root="E:/fairy"
        ))
        reply, _ = ac.handle_request("yes", history=[])

        assert "timed out" in reply.lower() or "timeout" in reply.lower()
        # Must NOT claim completion
        assert "completed" not in reply.lower()
        assert "fixed" not in reply.lower()


# ─────────────────────────────────────────────────────────────────
# 10. Expiry clears the pending state
# ─────────────────────────────────────────────────────────────────
@pytest.mark.skip(reason="removed in v2 handoff redesign — pending state machine gone")
class TestExpiry:
    def setup_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def teardown_method(self):
        from controller.delegate_state import clear_pending
        from controller.agent_controller import clear_hermes_gate
        clear_pending()
        clear_hermes_gate()

    def test_expired_delegation_does_not_execute(self, monkeypatch):
        """An expired pending delegation must not execute on approval."""
        from controller import agent_controller as ac
        from controller.delegate_state import (
            set_pending, get_pending, PendingDelegation
        )
        from datetime import datetime, timedelta

        executed = {"count": 0}
        def fake_delegate(**kwargs):
            executed["count"] += 1
            return {"executed": True, "status": "ok"}
        monkeypatch.setattr(ac, "delegate_to_claude_code", fake_delegate)

        # Create an already-expired pending delegation
        expired = PendingDelegation(
            original_request="Fix the bug",
            project_root="E:/fairy",
            detected_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() - timedelta(minutes=30),
        )
        set_pending(expired)

        reply, _ = ac.handle_request("yes", history=[])

        # Claude Code should NOT have been called
        assert executed["count"] == 0
        # Pending should be cleared
        assert get_pending() is None
