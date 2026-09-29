import json

import pytest
import urllib.request
from unittest.mock import patch

from controller.agent_controller import _structured_tool_decision, TOOLS


def test_planner_can_choose_browser_tool_for_open_request():
    """_structured_tool_decision calls _brain (→ main_brain.chat → urlopen).

    The conftest installs a generic "ok" fake that would make the JSON parse
    fail. This test provides its own urlopen patch returning a valid tool-
    decision JSON so the function exercises the full parse/return path.
    """
    def fake_urlopen(req, timeout=None):
        # Return a minimal valid chat/completions response whose content is
        # a JSON tool decision (mirrors what a real LLM would emit).
        body = json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"tool": "navigate_to", "arguments": {"url": "https://youtube.com"}, "reason": "user wants to open youtube"}),
                },
                "finish_reason": "stop",
            }],
            "created": 0,
        }).encode()
        class _Fake:
            def __init__(self, b): self._body = b
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return self._body
        return _Fake(body)

    with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
        decision = _structured_tool_decision("open youtube", TOOLS, [])

    assert isinstance(decision, dict)
    assert decision.get("tool") == "navigate_to"
    assert decision.get("arguments", {}).get("url") == "https://youtube.com"
    assert "error" not in decision


def test_prompt_builder_handles_nested_tool_schema():
    from core.prompts import build_action_decision_prompt
    tool = {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Open a URL in the browser.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
        },
    }
    prompt = build_action_decision_prompt("open youtube", [tool])
    assert "navigate_to" in prompt
    assert "Open a URL in the browser." in prompt
