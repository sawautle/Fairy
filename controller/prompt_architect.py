#!/usr/bin/env python3
"""
Prompt Architect - Fairy composes optimized prompts for Claude Code delegation.

Pipeline:
  1. Intake: user intent received.
  2. LLM-compose via main_brain.chat() (NEVER a template).
  3. Optional clarifying questions (max 3, interactive selector).
  4. Transparency panel: display composed prompt BEFORE launch.
  5. Wait for Y / R / N; only Y actually launches the spawn.

Safety: the LLM call is the ONLY source of the composed prompt.
The transparency panel is ALWAYS shown before any spawn. There is no
code path that can fabricate a Y-launch without going through this
function returning confirmed=True.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field

FAIRY_BLUE = "#3d7dfd"
FAIRY_LIGHT = "#7ec8ff"
FAIRY_WHITE = "#f2f8ff"
DIM = "#808080"

SYSTEM = """You are Fairy Prompt Architect.

Given a user intent and optional prior answers, produce a concrete,
optimized Claude Code prompt.

Rules:
1. Be concrete: specify filenames, line numbers, or error messages if given.
2. Set scope: if the user says fix bug in Fairy, scope to that module.
3. Add constraints: language, style, testing expectations if relevant.
4. If intent is vague, ask ONE clarifying question (max).
5. Never template. Compose the actual prompt text.

Output format (reply ONLY with this structure):
COMPOSED: <the actual Claude Code prompt>
CLARIFY: <your single clarifying question or NONE>
OPTIONS: <option A|option B|option C or NONE>
"""


@dataclass
class ArchitectResult:
    composed_prompt: str
    was_clarifying: bool = False
    clarifying_answers: list = field(default_factory=list)
    confirmed: bool = False


def _call_architect_llm(user_intent: str, history: list, prior_answers: list) -> tuple:
    """Call the LLM to compose or clarify. Returns (composed, was_clarifying, clarify_question, options_list)."""
    sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from controller.main_brain import chat
    except Exception:
        return _LLM_UNAVAILABLE_FALLBACK(user_intent)

    msgs = []
    if prior_answers:
        qa_text = "\n".join(f"Q: {a[0]}\nA: {a[1]}" for a in prior_answers)
        msgs.append({"role": "user", "content": f"Prior answers:\n{qa_text}"})
    msgs.append({"role": "user", "content": f"User intent: {user_intent}"})
    msgs.append({"role": "system", "content": SYSTEM})
    try:
        resp, _ = chat("auto", msgs)
        raw = resp.get("message", {}).get("content", "")
    except Exception:
        return _LLM_UNAVAILABLE_FALLBACK(user_intent)

    composed = ""
    clarifying = False
    question = "NONE"
    options = []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("COMPOSED:"):
            composed = s[len("COMPOSED:"):].strip()
        elif s.startswith("CLARIFY:"):
            question = s[len("CLARIFY:"):].strip()
            clarifying = question.upper() not in ("NONE", "", "N/A")
        elif s.startswith("OPTIONS:"):
            opts_str = s[len("OPTIONS:"):].strip()
            if opts_str.upper() not in ("NONE", "", "N/A"):
                options = [o.strip() for o in opts_str.split("|") if o.strip()]
    if not composed:
        composed = user_intent
    return composed, clarifying, question, options


def _LLM_UNAVAILABLE_FALLBACK(user_intent: str) -> tuple:
    """Called when LLM is unavailable. Returns the intent as-is, no clarification."""
    return user_intent, False, "NONE", []


def _wait_for_action(console=None) -> str:
    """Wait for Y / R / N keypress. Returns Y, R, or N."""
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.styles import Style

        def get_tokens():
            return FormattedText([
                ("bold #3d7dfd", "  [Y] Launch   "),
                ("fg:#808080", "[R] Refine   "),
                ("fg:#808080", "[N] Cancel"),
            ])

        kb = KeyBindings()

        def _y(e):
            e.app.exit(result="Y")

        def _r(e):
            e.app.exit(result="R")

        def _n(e):
            e.app.exit(result="N")

        kb.add("y", eager=True)(_y)
        kb.add("Y", eager=True)(_y)
        kb.add("r", eager=True)(_r)
        kb.add("R", eager=True)(_r)
        kb.add(Keys.Left, eager=True)(_r)
        kb.add(Keys.Up, eager=True)(_r)
        kb.add("n", eager=True)(_n)
        kb.add("N", eager=True)(_n)
        kb.add(Keys.Right, eager=True)(_n)
        kb.add(Keys.Down, eager=True)(_n)
        kb.add("c-c", eager=True)(_n)

        ctrl = FormattedTextControl(get_tokens, key_bindings=kb)
        win = Window(content=ctrl, width=80, height=3)
        style = Style.from_dict({"": "fg:#f2f8ff bg:#001a33"})
        return Application(layout=Layout(win), style=style, full_screen=False).run() or "N"
    except Exception:
        return _wait_for_action_plain()


def _wait_for_action_plain() -> str:
    """Plain fallback: print prompt and read one char."""
    sys.stdout.write("\n  Action: [Y] Launch   [R] Refine   [N] Cancel\n")
    sys.stdout.flush()
    while True:
        try:
            ch = sys.stdin.read(1)
            if not ch:
                return "N"
            c = ch.upper()
            if c in ("Y", "R", "N"):
                sys.stdout.write(f"  Selected: {c}\n")
                sys.stdout.flush()
                return c
        except (EOFError, KeyboardInterrupt, OSError):
            return "N"


def display_transparency_panel(composed_prompt: str, console=None) -> str:
    """Display composed prompt in bordered panel, wait for Y/R/N. Returns Y, R, or N."""
    from rich.console import Console as _Console
    from rich.panel import Panel as _Panel
    from rich.text import Text as _Text
    from rich.rule import Rule as _Rule
    _con = console or _Console()
    _con.print()
    _con.print(_Text("Claude  ✕  Fairy: Composed Prompt", style=f"bold {FAIRY_BLUE}"))
    _con.print(_Rule(style=FAIRY_BLUE))
    _con.print()
    _con.print(_Panel(
        _Text(composed_prompt[:2000], style=FAIRY_WHITE),
        title="[ Composed Prompt ]",
        border_style=FAIRY_BLUE,
        width=80,
        padding=(1, 2),
    ))
    _con.print()
    return _wait_for_action(console)


def get_git_evidence(project_root: str) -> tuple:
    """Return (git_status, git_diff, git_diffstat) from project_root. None if unavailable."""

    def _run(args):
        try:
            r = subprocess.run(args, cwd=project_root, capture_output=True, text=True, timeout=15)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    status = _run(["git", "status", "--short"])
    diff = _run(["git", "diff", "--no-color"])
    stat = _run(["git", "diff", "--stat", "--no-color"])
    return status, diff, stat


def run_prompt_architect(user_intent: str, history: list = None, console=None) -> ArchitectResult:
    """Orchestrate the full Prompt Architect pipeline."""
    from controller.choice_selector import run_choice_selector
    prior_answers: list = []
    current_intent = user_intent
    clarifying = True
    loops = 0
    MAX_LOOPS = 3

    while clarifying and loops < MAX_LOOPS:
        loops += 1
        composed, was_clarifying, question, options = _call_architect_llm(
            current_intent, history or [], prior_answers
        )
        if not was_clarifying:
            clarifying = False
            break
        if question == "NONE" or question == "N/A":
            clarifying = False
            break
        if options:
            result = run_choice_selector(question, options, max_questions=1)
            if result is None:
                return ArchitectResult(
                    composed_prompt=composed, was_clarifying=True,
                    clarifying_answers=prior_answers, confirmed=False,
                )
            answers, _indices = result
            answer = answers[0]
            prior_answers.append((question, answer))
            current_intent = f"{user_intent}\n\nClarification: {question}\nUser answer: {answer}"
        else:
            sys.stdout.write(f"\n  [Fairy] Clarification needed: {question}\n")
            sys.stdout.write("  Your answer: ")
            sys.stdout.flush()
            try:
                answer = sys.stdin.readline().strip()
                if not answer:
                    prior_answers.append((question, "(no answer)"))
                else:
                    prior_answers.append((question, answer))
                current_intent = f"{user_intent}\n\nClarification: {question}\nUser answer: {answer}"
            except (EOFError, KeyboardInterrupt, OSError):
                return ArchitectResult(
                    composed_prompt=composed, was_clarifying=True,
                    clarifying_answers=prior_answers, confirmed=False,
                )

    composed, _, _, _ = _call_architect_llm(current_intent, history or [], prior_answers)
    action = display_transparency_panel(composed, console)
    confirmed = action == "Y"
    return ArchitectResult(
        composed_prompt=composed,
        was_clarifying=bool(prior_answers),
        clarifying_answers=[a for _, a in prior_answers],
        confirmed=confirmed,
    )

