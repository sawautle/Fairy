#!/usr/bin/env python3
"""
TTY letter/arrow-key selector for Prompt Architect clarifying questions.

Uses prompt_toolkit RadioList-style key bindings to present numbered/lettered
options with Fairy's blue theme. Synchronous call between turns — the main
fairy_prompt session is not active while this runs (blocked at call site until
user selects). No simultaneous render conflict by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

_HAVE_PT = True
try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout.containers import Window, HSplit
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style
    from prompt_toolkit.layout import Layout
except Exception:
    _HAVE_PT = False

# Fairy's palette
_FAIRY_BLUE = "#3d7dfd"
_FAIRY_LIGHT = "#7ec8ff"
_FAIRY_WHITE = "#f2f8ff"
_DIM = "#808080"
_OPTION_LETTERS = [chr(ord("A") + i) for i in range(26)]


@dataclass
class ChoiceResult:
    selected_index: int
    selected_letter: str


def run_choice_selector(
    question: str,
    options: list[str],
    *,
    max_questions: int = 3,
) -> tuple[list[str], list[int]] | None:
    """
    Present clarifying questions with lettered options until the request is clear
    or max_questions is reached.

    Returns None on Ctrl+C. Otherwise returns (answers, indices) where answers
    are the selected option texts and indices are the 0-based selected indices.
    """
    if not options:
        return None
    # Single-option lists auto-resolve; no TTY interaction needed.
    if len(options) == 1:
        return [options[0]], [0]

    if not _HAVE_PT:
        return _plain_fallback(question, options)

    all_answers: list[str] = []
    all_indices: list[int] = []

    while len(all_answers) < max_questions:
        selected = _interactive_select(question, options)
        if selected is None:
            return None
        all_answers.append(options[selected.selected_index])
        all_indices.append(selected.selected_index)
        # for phase 2: one question per call is sufficient
        break

    return all_answers, all_indices


def _interactive_select(
    question: str,
    options: list[str],
) -> ChoiceResult | None:
    """Run a single interactive selector via prompt_toolkit."""
    if not _HAVE_PT:
        return _plain_fallback_single(question, options)

    selected_index = [0]  # mutable container for closures

    def get_text():
        lines: list[tuple[str, str]] = []
        lines.append(("", "\n"))
        lines.append((f"bold {_FAIRY_LIGHT}", f"  {question}"))
        lines.append(("", "\n\n"))
        for i, opt in enumerate(options):
            letter = _OPTION_LETTERS[i]
            is_sel = (i == selected_index[0])
            if is_sel:
                lines.append(("", "  "))
                lines.append((f"bold {_FAIRY_BLUE}", f"  ► [{letter}] {opt}"))
                lines.append(("", "\n"))
            else:
                lines.append(("", "   "))
                lines.append((f"fg:{_DIM}", f"    [{letter}] {opt}"))
                lines.append(("", "\n"))
        lines.append(("", "  "))
        lines.append((f"fg:{_DIM}", "  ↑↓ navigate  ·  Enter select  ·  Ctrl+C abort"))
        lines.append(("", "\n"))
        return lines

    kb = KeyBindings()

    @kb.add("c-c", eager=True)
    def _abort(event):
        event.app.exit(result=None)

    @kb.add(Keys.Up)
    def _up(event):
        selected_index[0] = (selected_index[0] - 1) % len(options)

    @kb.add(Keys.Down)
    def _down(event):
        selected_index[0] = (selected_index[0] + 1) % len(options)

    @kb.add(Keys.Enter)
    def _enter(event):
        idx = selected_index[0]
        event.app.exit(result=ChoiceResult(
            selected_index=idx,
            selected_letter=_OPTION_LETTERS[idx],
        ))

    # Letter shortcuts
    for i in range(min(len(options), 26)):
        ltr = _OPTION_LETTERS[i]

        @kb.add(ltr, eager=True)
        def _letter(event, idx=i, ltr=ltr):
            selected_index[0] = idx
            event.app.exit(result=ChoiceResult(selected_index=idx, selected_letter=ltr))

    style = Style.from_dict({"": f"fg:{_FAIRY_WHITE} bg:#001a33"})
    control = FormattedTextControl(get_text, key_bindings=kb)
    window = Window(content=control, width=80)
    layout = Layout(HSplit([window]))
    app = Application(layout=layout, style=style, full_screen=False)
    return app.run()


def _plain_fallback(question: str, options: list[str]) -> tuple[list[str], list[int]] | None:
    print(f"\n  {question}")
    for i, opt in enumerate(options):
        print(f"  [{_OPTION_LETTERS[i]}] {opt}")
    print("  ↑↓ navigate · Enter select · Ctrl+C abort\n")
    while True:
        try:
            key = input("Selection: ").strip().upper()
            if key in _OPTION_LETTERS[:len(options)]:
                idx = _OPTION_LETTERS.index(key)
                return [options[idx]], [idx]
        except (EOFError, KeyboardInterrupt):
            return None


def _plain_fallback_single(question: str, options: list[str]) -> ChoiceResult | None:
    result = _plain_fallback(question, options)
    if result is None:
        return None
    answers, indices = result
    return ChoiceResult(selected_index=indices[0], selected_letter=_OPTION_LETTERS[indices[0]])
