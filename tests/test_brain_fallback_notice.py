"""Tests for brain-fallback notice: no silent degradation when Hermes is unavailable."""
import os
import sys

os.environ.pop("FAIRY_USE_HERMES", None)
os.environ.pop("FAIRY_TASK_GUARD_REENTRY", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock


class TestBrainFallbackNotice:
    """When Hermes is unavailable and main_brain answers, a polite notice appears."""

    def test_fallback_notice_in_reply_when_hermes_skipped(self):
        """When run_turn_safe returns success=True but get_last_provider()
        is not "hermes" (i.e., main_brain answered instead), handle_request
        must surface a fallback notice in the reply text."""
        from controller import agent_controller as ac
        import hermes_bridge

        with patch("hermes_bridge.run_turn_safe") as mock_run:
            # main_brain answered successfully (success=True), but
            # the provider sentinel is "main_brain/openrouter"
            mock_run.return_value = (True, "I answered via fallback.", [])
            with patch.object(hermes_bridge, "_LAST_PROVIDER", "main_brain/openrouter"):
                reply, _ = ac.handle_request("hello fairy", [])
                assert isinstance(reply, str)
                # The fallback notice must appear — no silent degradation.
                assert any(
                    token in reply.lower()
                    for token in ("fallback", "busy", "answering")
                ), f"Expected fallback notice in reply, got: {reply!r}"

    def test_no_notice_when_hermes_answered(self):
        """When Hermes answers normally (provider == "hermes"), no fallback notice."""
        from controller import agent_controller as ac
        import hermes_bridge

        with patch("hermes_bridge.run_turn_safe") as mock_run:
            mock_run.return_value = (True, "Hello, Master!", [])
            with patch.object(hermes_bridge, "_LAST_PROVIDER", "hermes"):
                reply, _ = ac.handle_request("hello fairy", [])
                assert isinstance(reply, str)
                # No fallback notice when Hermes answered successfully
                assert "fallback" not in reply.lower(), \
                    f"Unexpected fallback notice when Hermes succeeded: {reply!r}"

    def test_fallback_notice_provides_provider_info(self):
        """The fallback notice must indicate WHICH provider answered."""
        from controller import agent_controller as ac
        import hermes_bridge

        with patch("hermes_bridge.run_turn_safe") as mock_run:
            mock_run.return_value = (True, "Fallback answer.", [])
            with patch.object(hermes_bridge, "_LAST_PROVIDER", "main_brain/openrouter"):
                reply, _ = ac.handle_request("hello fairy", [])
                # Notice should mention the provider (ollama or openrouter or gemma)
                assert any(
                    p in reply.lower()
                    for p in ("ollama", "openrouter", "gemma", "main_brain")
                ), f"Notice should mention provider, got: {reply!r}"
