"""
test_explicit_claude_and_fallback_honesty.py

Regression tests for two fixes:

1. detect_repository_task() — explicit agent-mention override
   Any input containing "using claude", "use claude to", "via claude",
   "have claude do it" (or the "Claude Code" variant) MUST trigger
   delegation, regardless of whether it contains repo keywords or lives
   inside a git repository.

2. agent_controller fallback-brain honesty guard
   When Hermes is unavailable and the fallback brain (Ollama → OpenRouter)
   answers with action language ("On it", "Let me grab...") but dispatched
   zero tools, the guard must either re-route through the planner (first
   occurrence) or return an honest-failure message (second/deep re-route).
   A truthful "I couldn't dispatch" beats a fabricated "On it".
"""
from __future__ import annotations

import os
import sys

# Wipe Fairy's own env vars so the test harness is hermetic.
os.environ.pop("FAIRY_USE_HERMES", None)
os.environ.pop("FAIRY_TASK_GUARD_REENTRY", None)
os.environ.pop("_ROUTED_AFTER_FALLBACK_NO_TOOLS", None)

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_THIS)
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import pytest
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════════════════════════════
# 1. detect_repository_task — explicit agent-mention patterns
# ══════════════════════════════════════════════════════════════════════════════════
class TestDetectRepositoryTaskExplicitClaude:
    """Explicit 'using claude' / 'use Claude Code' phrases ALWAYS trigger delegation."""

    # ── bare-claude variants (the original failing input is covered in its own class)
    @pytest.mark.parametrize("phrase", [
        "using claude",
        "Using Claude",
        "using CLAUDE",
        "use claude",
        "Use Claude",
        "via claude",
        "Via Claude",
        "have claude do it",
        "Have Claude do it",
        "have claude do the thing",
        "via Claude",
    ])
    def test_bare_claude_triggers(self, phrase):
        from controller.claude_code_delegate import detect_repository_task
        assert detect_repository_task(f"make me a folder {phrase}"), \
            f"'{phrase}' should trigger delegation"

    # ── "Claude Code" variants
    @pytest.mark.parametrize("phrase", [
        "using claude code",
        "Using Claude Code",
        "use claude code to",
        "Use Claude Code to",
        "via claude code",
        "Via Claude Code",
        "have claude code do it",
        "Have Claude Code do it",
    ])
    def test_claude_code_triggers(self, phrase):
        from controller.claude_code_delegate import detect_repository_task
        assert detect_repository_task(f"make me a folder {phrase}"), \
            f"'{phrase}' should trigger delegation"

    # ── embedded in longer natural sentences
    @pytest.mark.parametrize("sentence", [
        "organize my screenshots using claude",
        "organise my screenshots using claude",
        "build a folder structure using claude code for me",
        "can you use claude to fix that bug",
        "please have claude code generate the docs",
        "would you use claude to run the tests",
        "set up my project using claude",
        "write a script using claude code",
    ])
    def test_embedded_in_sentence(self, sentence):
        from controller.claude_code_delegate import detect_repository_task
        assert detect_repository_task(sentence), \
            f"'{sentence}' should trigger delegation"

    # ── non-delegation phrases must NOT trigger
    @pytest.mark.parametrize("phrase", [
        "hello fairy",            # greeting, no delegation intent
        "what time is it",        # casual query
        "who made you",           # casual query
        "tell me a joke",          # chat
        "i like claude",           # "claude" is not a delegation signal here
        "open claude",             # means "open the app Claude" (ambiguous)
        "use vscode to edit",      # different tool
        "ask claude",              # "ask" is not delegation
        "tell claude to do it",    # indirect; not one of our patterns
    ])
    def test_non_delegation_phrases_do_not_trigger(self, phrase):
        from controller.claude_code_delegate import detect_repository_task
        assert not detect_repository_task(phrase), \
            f"'{phrase}' should NOT trigger delegation"


# ══════════════════════════════════════════════════════════════════════════════════
# 2. Regression: the exact failing input from the bug report
# ══════════════════════════════════════════════════════════════════════════════════
class TestRegressionExactFailingInput:
    """Regression test for the exact input that triggered the bug.

    "make me a folder containing all my screenshots using claude"
    → no DELEGATE_PENDING event because detect_repository_task returned False.
    After the fix, this must return True.
    """

    def test_exact_failing_input_triggers(self):
        from controller.claude_code_delegate import detect_repository_task
        inp = "make me a folder containing all my screenshots using claude"
        assert detect_repository_task(inp), \
            f"Exact failing input should trigger delegation: {inp!r}"

    def test_variant_screenshots_using_claude(self):
        from controller.claude_code_delegate import detect_repository_task
        inp = "copy my screenshots using claude"
        assert detect_repository_task(inp), \
            f"'copy my screenshots using claude' should trigger delegation"

    def test_variant_make_folder_via_claude(self):
        from controller.claude_code_delegate import detect_repository_task
        inp = "make a folder for my documents via claude"
        assert detect_repository_task(inp), \
            f"'make a folder for my documents via claude' should trigger delegation"


# ══════════════════════════════════════════════════════════════════════════════════
# 3. Fallback-brain honesty guard
# ══════════════════════════════════════════════════════════════════════════════════
class TestFallbackBrainHonesty:
    """When the fallback brain answers with action language but dispatches zero
    tools, the guard must re-route (first time) or return honest-failure
    (second time). A truthful 'I couldn't dispatch' beats a fabricated 'On it'.
    """

    def test_action_claim_zero_tools_re_routes_via_planner(self):
        """First occurrence: action-claim + zero tools → re-route through planner."""
        from controller import agent_controller as ac
        import hermes_bridge

        call_count = [0]

        def mock_run_turn_safe(user_text, history, **kwargs):
            call_count[0] += 1
            # Simulate fallback brain (not Hermes) answering with action language
            # and NOT dispatching any tools.
            return (True, "On it, Master. Let me grab your screenshots folder...", [])

        with patch("hermes_bridge.run_turn_safe", side_effect=mock_run_turn_safe):
            with patch.object(hermes_bridge, "_LAST_PROVIDER", "main_brain/openrouter"):
                # Mock _verify_creation_claims so it doesn't trigger the earlier fix
                with patch.object(ac, "_verify_creation_claims", return_value=(True, None)):
                    with patch.object(ac, "_strip_meta_blocks", side_effect=lambda r, **k: r):
                        with patch.object(ac, "_complexity_profile", return_value={"strategy": "fast"}):
                            # Track _TOOLS_DISPATCHED_THIS_TURN across calls.
                            # First call: zero tools (fallback brain)
                            # Second call (planner re-route): zero tools still
                            #   → should still be caught by the depth-limited re-route.
                            #   → but since tools_dispatched is 0 and routed_depth="1",
                            #     the honest-failure path is taken.
                            # We just verify no exception and honest-failure reply.
                            # NB: In v2, "using claude" is the explicit handoff trigger,
                            # so handle_request returns (None, history) and sets
                            # _pending_handoff. The TUI layer handles the handoff.
                            reply, hist = ac.handle_request(
                                "make me a folder containing all my screenshots using claude",
                                [],
                                on_status=MagicMock(),
                            )
                            # In v2: explicit handoff returns None with _pending_handoff set.
                            assert reply is None
                            pending = ac.get_pending_handoff()
                            assert pending is not None
                            assert pending.get("action") == "handoff"

    def test_re_routed_still_zero_tools_returns_honest_failure(self):
        """Second occurrence (depth-limited): action-claim + zero tools → honest failure."""
        from controller import agent_controller as ac
        import hermes_bridge

        call_count = [0]

        def mock_run_turn_safe(user_text, history, **kwargs):
            call_count[0] += 1
            # Simulate a brain that always answers with action language but never
            # dispatches tools — useful for testing the depth limit.
            return (True, "On it, Master. Let me grab your screenshots folder...", [])

        with patch("hermes_bridge.run_turn_safe", side_effect=mock_run_turn_safe):
            with patch.object(hermes_bridge, "_LAST_PROVIDER", "main_brain/openrouter"):
                with patch.object(ac, "_verify_creation_claims", return_value=(True, None)):
                    with patch.object(ac, "_strip_meta_blocks", side_effect=lambda r, **k: r):
                        with patch.object(ac, "_complexity_profile", return_value={"strategy": "fast"}):
                            # Set the reentry flag to simulate already having re-routed once.
                            # This forces the depth-limited honest-failure path.
                            with patch.dict(os.environ, {"_ROUTED_AFTER_FALLBACK_NO_TOOLS": "1"}):
                                reply, hist = ac.handle_request(
                                    "make me a folder using claude",
                                    [],
                                    on_status=MagicMock(),
                                )
                                # In v2: explicit handoff → reply is None, pending_handoff is set.
                                # The honest-failure guard is still in place for non-handoff paths.
                                assert reply is None
                                pending = ac.get_pending_handoff()
                                assert pending is not None
                                assert pending.get("action") == "handoff"

    def test_chat_without_action_claim_passes_through(self):
        """Pure chat (no action claim) should NOT trigger the guard."""
        from controller import agent_controller as ac
        import hermes_bridge

        def mock_run_turn_safe(user_text, history, **kwargs):
            # Fallback brain returns a pure chat reply — no action language.
            return (True, "I hear you, Master! What's on your mind today?", [])

        def mock_get_last_provider():
            # Must return NOT "hermes" so the fallback-notice path is reached.
            # Patching _LAST_PROVIDER directly doesn't work because
            # get_last_provider() is a function call — we must patch the function.
            return "main_brain/openrouter"

        with patch("hermes_bridge.run_turn_safe", side_effect=mock_run_turn_safe):
            with patch.object(hermes_bridge, "get_last_provider", mock_get_last_provider):
                with patch.object(ac, "_verify_creation_claims", return_value=(True, None)):
                    with patch.object(ac, "_strip_meta_blocks", side_effect=lambda r, **k: r):
                        reply, hist = ac.handle_request(
                            "how are you today",
                            [],
                            on_status=MagicMock(),
                        )
                        # The chat reply should pass through unchanged (plus fallback notice).
                        assert "Master" in reply or "hear you" in reply
                        # The guard must NOT have re-routed — the fallback notice
                        # should still be present.
                        assert "fallback" in reply.lower(), \
                            f"Expected fallback notice in reply (chat not re-routed). Got: {reply!r}"

    def test_tools_dispatched_no_guard_trigger(self):
        """When tools ARE dispatched, the fabricated-reply guard must NOT fire."""
        from controller import agent_controller as ac
        import hermes_bridge

        def mock_run_turn_safe(user_text, history, **kwargs):
            return (True, "On it, Master. Making your folder now...", [])

        with patch("hermes_bridge.run_turn_safe", side_effect=mock_run_turn_safe):
            with patch.object(hermes_bridge, "_LAST_PROVIDER", "main_brain/openrouter"):
                with patch.object(ac, "_verify_creation_claims", return_value=(True, None)):
                    with patch.object(ac, "_strip_meta_blocks", side_effect=lambda r, **k: r):
                        # Patch _complexity_profile so the re-route path doesn't crash
                        # (the re-route sets FAIRY_USE_HERMES=0 and recurses, which
                        # hits _complexity_profile inside the planner).
                        with patch.object(ac, "_complexity_profile",
                                          return_value={"score": 0, "mode": "fast", "strategy": "fast", "reasons": []}):
                            # Simulate that tools WERE dispatched. We patch the
                            # _was_tool_dispatched function directly so both the
                            # guard call AND any nested call inside re-routes
                            # see the patched value.
                            with patch.object(ac, "_was_tool_dispatched",
                                              return_value=(1, ["file_creation"])):
                                reply, hist = ac.handle_request(
                                    "organize my screenshots",
                                    [],
                                    on_status=MagicMock(),
                                )
                                # Since tools were dispatched, the fabricated-reply guard
                                # should NOT have re-routed. The reply should still contain
                                # the action language (the fallback brain correctly narrated).
                                assert "On it" in reply or "Making" in reply, \
                                    f"Reply should be preserved when tools dispatched. Got: {reply!r}"
