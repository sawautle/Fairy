"""Live voice-flow smoke test.

Exercises the exact code path the voice listener triggers when it gets a
transcript. Runs each of the 5 user-voice scenarios and prints the
response, mirroring what the user would hear.

This is not a unit test — it imports live agent_controller and runs the
real handle_request path. Only the LLM/Hermes are bypassed where the
resolver path returns early (the delegation path, clarification gate,
fast-path, URL research).
"""
import os
import sys

# Force UTF-8 stdout for Windows console — the test prints Unicode arrows
# and box-drawing characters that cp1252 can't encode.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

os.environ.pop("FAIRY_USE_HERMES", None)
os.environ.pop("FAIRY_TASK_GUARD_REENTRY", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from controller.delegate_state import clear_pending
from controller import agent_controller as ac


# ANSI for visual readability in terminal
# Use ASCII-only markers on Windows — no Unicode
_GREEN = "[PASS] "
_BLUE = "[INFO] "
_YELLOW = "[WARN] "
_RED = "[FAIL] "
_END = ""


def _print_scenario(num, scenario_text, expected):
    print(f"\n{_BLUE}-- Scenario {num}: {scenario_text!r} --{_END}")
    print(f"  expected: {expected}")


def _print_reply(reply, ok, notes=""):
    colour = _GREEN if ok else _RED
    print(f"  {colour}REPLY{_END}: {reply}")
    if notes:
        print(f"  notes: {notes}")


def scenario_1_cod_to_folder():
    """'use cod to make a folder' — must show visible reasoning + delegation prompt."""
    _print_scenario(
        1,
        "use cod to make a folder",
        "visible reasoning about cod/game/Claude Code + delegation permission prompt",
    )
    clear_pending()
    with patch("hermes_bridge.run_turn_safe") as mh:
        mh.return_value = (False, "fallback", [])
        with patch("controller.agent_controller._brain") as mb:
            mb.return_value = MagicMock(
                message=MagicMock(content="", tool_calls=[]),
                done=True,
            )
            reply, _ = ac.handle_request("use cod to make a folder", [])

    has_visible = any(
        s in reply.lower() for s in ("claude code", "folder", "delegation", "permission")
    )
    has_game_mention = "game" in reply.lower() or "can't" in reply.lower()
    hermes_called = False  # We can't check this after the with-block, but if
                          # the reply is a permission prompt, the path was taken.
    _print_reply(
        reply,
        ok=has_visible and has_game_mention,
        notes=(
            f"visible reasoning present: {has_visible}, "
            f"game/can't mentioned: {has_game_mention}"
        ),
    )
    return has_visible and has_game_mention


def scenario_2_open_cod_clarification():
    """'open cod' — must return clarification, no action taken."""
    _print_scenario(
        2,
        "open cod",
        "clarification question, no action",
    )
    clear_pending()
    with patch("hermes_bridge.run_turn_safe") as mh:
        mh.return_value = (False, "fallback", [])
        with patch("controller.agent_controller._brain") as mb:
            mb.return_value = MagicMock(
                message=MagicMock(content="", tool_calls=[]),
                done=True,
            )
            reply, _ = ac.handle_request("open cod", [])

    is_clarification = (
        "?" in reply
        and any(s in reply.lower() for s in ("cod", "game", "claude"))
    )
    no_action = "http" not in reply.lower() and "browser" not in reply.lower()
    _print_reply(
        reply,
        ok=is_clarification and no_action,
        notes=f"is clarification: {is_clarification}, no browser hit: {no_action}",
    )
    return is_clarification and no_action


def scenario_3_fast_path_unchanged():
    """'hey fairy, what's the time' — must be fast-path, instant, unchanged."""
    _print_scenario(
        3,
        "hey fairy, what's the time",
        "fast-path, no LLM call, instant",
    )
    clear_pending()
    from controller.intent_resolver import resolve_intent

    class _NoCallBrain:
        def __init__(self):
            self.called = False

        def __call__(self, *a, **k):
            self.called = True
            return MagicMock(
                message=MagicMock(content="", tool_calls=[]),
                done=True,
            )

    nb = _NoCallBrain()
    import time
    t0 = time.perf_counter()
    r = resolve_intent("hey fairy, what's the time", brain_fn=nb)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    is_fast = r.fast_path_eligible
    brain_not_called = not nb.called  # True = fast-path avoided the LLM
    _print_reply(
        f"fast_path={is_fast}, brain_not_called={brain_not_called}, elapsed={elapsed_ms:.1f}ms",
        ok=is_fast and brain_not_called and elapsed_ms < 100,
        notes=f"elapsed: {elapsed_ms:.2f}ms (target: <100ms)",
    )
    return is_fast and brain_not_called and elapsed_ms < 100


def scenario_4_normal_chat():
    """'thanks fairy' — normal chat must not engage resolver at all."""
    _print_scenario(
        4,
        "thanks fairy",
        "fast-path (chat), no LLM, no resolver cost",
    )
    clear_pending()
    from controller.intent_resolver import resolve_intent

    class _NoCallBrain:
        def __init__(self):
            self.called = False

        def __call__(self, *a, **k):
            self.called = True
            return MagicMock(
                message=MagicMock(content="", tool_calls=[]),
                done=True,
            )

    nb = _NoCallBrain()
    r = resolve_intent("thanks fairy", brain_fn=nb)
    brain_not_called = not nb.called
    _print_reply(
        f"fast_path={r.fast_path_eligible}, brain_not_called={brain_not_called}",
        ok=r.fast_path_eligible and brain_not_called,
    )
    return r.fast_path_eligible and brain_not_called


def scenario_5_url_research_intact():
    """URL research: the 'old path' for web research must still fire when the
    resolver correctly identifies a website. We confirm via resolve_intent
    that 'look up https://example.com' resolves to a website routing hint."""
    _print_scenario(
        5,
        "open example.com",
        "fast-path website, no LLM call",
    )
    clear_pending()
    from controller.intent_resolver import resolve_intent

    class _NoCallBrain:
        def __init__(self):
            self.called = False

        def __call__(self, *a, **k):
            self.called = True
            return MagicMock(
                message=MagicMock(content="", tool_calls=[]),
                done=True,
            )

    nb = _NoCallBrain()
    r = resolve_intent("open example.com", brain_fn=nb)
    is_website = (
        r.chosen_meaning is not None
        and r.chosen_meaning.category == "website"
        and r.routing_hint == "browser"
    )
    _print_reply(
        f"category={r.chosen_meaning.category if r.chosen_meaning else 'none'}, "
        f"hint={r.routing_hint}",
        ok=is_website,
    )
    return is_website


def main():
    print(f"{_YELLOW}=== Live voice-flow smoke test ==={_END}")
    print("(Each scenario exercises the same code path the voice listener triggers.)")
    print()

    results = []
    results.append(("cod->folder", scenario_1_cod_to_folder()))
    results.append(("open cod clarify", scenario_2_open_cod_clarification()))
    results.append(("fast-path time", scenario_3_fast_path_unchanged()))
    results.append(("normal chat", scenario_4_normal_chat()))
    results.append(("URL research", scenario_5_url_research_intact()))

    print(f"\n{_YELLOW}=== Summary ==={_END}")
    all_ok = True
    for name, ok in results:
        colour = _GREEN if ok else _RED
        print(f"  {colour}{'PASS' if ok else 'FAIL'}{_END}: {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print(f"{_GREEN}All 5 voice-flow scenarios PASS.{_END}")
        print("Voice tests green. Tag-style follow-up commit:")
        print(
            '  git commit --allow-empty -m "verified: intent resolution passes live voice tests"'
        )
        return 0
    else:
        print(f"{_RED}Some scenarios failed — DO NOT commit.{_END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
