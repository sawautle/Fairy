"""
controller/terminal_input.py
============================
Multi-line terminal input helper, extracted from main.py so tests
can drive it without dragging in numpy / ollama / discord / etc.

The binding contract:

  Enter              → submit (the content of the buffer is sent as-is)
  Alt+Enter / Ctrl+J → insert newline (keep editing)
  Ctrl+L             → clear screen (input is NOT lost)
  Ctrl+C             → raises KeyboardInterrupt (caller handles it)

The continuation prompt is "  … " so the Master always knows whether
bare Enter will send or wrap. UTF-8 / Bangla passes through untouched.

Graceful fallback: if prompt_toolkit is unavailable (bare env), the
module exposes ``read_line()`` that calls plain input() with a one-time
warning on stderr. The REPL still works, just without multi-line.

Import-time behaviour: _initialize() is LAZY — it is NOT called at
import time. Tests can safely import this module without needing a TTY.
The first read_line() call triggers initialization. main.py calls
_initialize() explicitly before the loop so the REPL startup is not
surprised by a lazy first-read.
"""
from __future__ import annotations

import os
import sys

# State — initialized lazily on first read_line() call.
_PT_AVAILABLE = False
_PT_SESSION = None
_pt_history = None
_continuation_prompt = "  … "
_main_prompt = "you› "
_init_done = False


def _initialize():
    """Set up the prompt_toolkit session. Safe to call multiple times.
    Raises only when the environment doesn't support PT (no TTY, missing
    dependency) — callers should catch and fall back to plain input()."""
    global _PT_AVAILABLE, _PT_SESSION, _pt_history, _init_done
    if _PT_SESSION is not None or _init_done:
        return

    _init_done = True  # mark done even if it fails — don't retry

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory, InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style as PTStyle

        _FAIRY_BLUE = "#0088cc"

        _pt_history_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".fairy_history"
        )
        try:
            _pt_history = FileHistory(_pt_history_file)
        except Exception:
            _pt_history = InMemoryHistory()

        _pt_style = PTStyle.from_dict({
            "continuation": f"fg:{_FAIRY_BLUE}",
        })

        def _make_keybindings():
            kb = KeyBindings()

            # Alt+Enter (escape+enter) — the standard prompt_toolkit multi-line
            # binding. Shift+Enter is not available in prompt_toolkit 3.x on
            # Windows, so we use Alt+Enter (the "common alternative" from the task).
            @kb.add("escape", "enter")
            def _newline_alt(event):
                event.current_buffer.insert_text("\n")

            # Ctrl+J — fallback that works even where terminals eat Alt+Enter.
            @kb.add("c-j")
            def _newline_ctrlj(event):
                event.current_buffer.insert_text("\n")

            # Ctrl+L — clear screen without losing input. Non-disruptive.
            @kb.add("c-l")
            def _clear_screen(event):
                try:
                    from rich.console import Console
                    _c = Console()
                    _c.clear()
                except Exception:
                    pass
                try:
                    event.app.renderer.reset()
                except Exception:
                    pass

            return kb

        def _prompt_continuation(width, line_number, wrap_count):
            """The continuation prompt shown when the buffer spans
            multiple lines. This is the visual affordance — the
            Master always knows whether bare Enter will send or wrap."""
            return _continuation_prompt

        _PT_SESSION = PromptSession(
            history=_pt_history,
            key_bindings=_make_keybindings(),
            multiline=True,
            wrap_lines=True,
            style=_pt_style,
            prompt_continuation=_prompt_continuation,
        )
        _PT_AVAILABLE = True
    except Exception:
        # PromptSession raised — no TTY or missing dependency.
        # The read_line() fallback will handle it.
        _PT_AVAILABLE = False
        _PT_SESSION = None


def read_line(prompt: str | None = None) -> str:
    """Read one line (or multi-line block) of user input.

    On first call, triggers _initialize() to set up the prompt_toolkit
    session. If that fails (no TTY, missing dep), falls back to plain
    input().

    If prompt_toolkit is available: returns the EXACT submitted text,
    stripping only a single trailing newline (so internal newlines from
    Alt+Enter survive).

    Bangla and other UTF-8 are passed through unchanged.

    Returns the exact text the user submitted (with internal newlines
    intact). Raises KeyboardInterrupt / EOFError so the caller can
    handle Ctrl-C / Ctrl-D.
    """
    # Lazily initialize on first read_line() call.
    _initialize()

    if _PT_SESSION is not None:
        try:
            text = _PT_SESSION.prompt(prompt or _main_prompt)
        except (EOFError, KeyboardInterrupt):
            raise
        # Strip only a trailing newline — never strip internal newlines
        # the Master typed with Alt+Enter.
        text = text.rstrip("\n")
        return text

    # Fallback path.
    if not hasattr(read_line, "_warned"):
        read_line._warned = True  # type: ignore
        print(
            "[FAIRY] prompt_toolkit not available — multi-line input disabled.\n"
            "        Install it with: pip install prompt_toolkit",
            file=sys.stderr,
        )
    return input(prompt or "you> ")


def _reset():
    """Reset module state. FOR TESTS ONLY — never call in production."""
    global _PT_AVAILABLE, _PT_SESSION, _pt_history, _init_done
    _PT_AVAILABLE = False
    _PT_SESSION = None
    _pt_history = None
    _init_done = False
