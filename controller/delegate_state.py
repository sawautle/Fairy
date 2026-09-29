"""
delegate_state.py — Phrase classifiers used by the handoff permission gate
and by the fallback-brain honesty guard.

Fairy used to maintain a module-level pending-delegation state machine
(IDLE → PENDING_APPROVAL → EXECUTING). That pipeline was removed in the
v1→v2 redesign. This module now only owns the phrase classifier
utilities that other layers still depend on:

  - `is_approval(text)`     — used by the heuristic permission gate
                              ("Use Claude Code for this?") and by the
                              fallback brain to detect honest yes/no.
  - `is_denial(text)`       — same.
  - `is_clarification(text)`— same.

The fallback-brain honesty guard (refactored out of agent_controller)
still consults is_approval/is_denial/is_clarification to detect when a
fallback LLM turn is narrating a fake delegation while the real LLM
counter is empty. The functions must stay import-stable.
"""
from __future__ import annotations

import re
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────────
# Debug hook — controller wires its _debug() into us at import time.
# Defined as a module-level variable so the phrase classifiers can emit
# structured debug events without taking a hard dependency on
# controller.agent_controller (which would create an import cycle).
# ─────────────────────────────────────────────────────────────────
_debug_hook: Optional[Callable[[str, dict], None]] = None


def set_debug_hook(fn: Optional[Callable[[str, dict], None]]) -> None:
    """Register a debug logger. ``fn`` receives ``(label, payload_dict)``.

    Pass ``None`` to clear the hook. Used by controller/agent_controller.py
    so DELEGATE_* events land in fairy_debug.log next to all the other
    chat-turn events. Safe to call multiple times.
    """
    global _debug_hook
    _debug_hook = fn


def _emit_debug(label: str, payload: Optional[dict] = None) -> None:
    """Best-effort fire of the debug hook. Never raises — debug logging
    must NEVER break the approval path."""
    if _debug_hook is None:
        return
    try:
        _debug_hook(label, payload or {})
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────
# Approval / denial phrase sets
# ─────────────────────────────────────────────────────────────────
# Approval matching uses PREFIX matching (with whitespace boundaries) for
# common yes-style words so that natural responses like "yes go ahead",
# "yes please", "yeah do it" are recognized. Longer sentences that merely
# contain "yes" but are obviously not permissions (e.g. "yes I have a
# question") still fall through to clarification handling.
#
# Exact-match phrases like "proceed", "approved", "let her do it" still
# require an exact match — those are unambiguous on their own.
#
# Denial matching remains exact-match only — "no" is short enough that
# any sentence containing it is almost certainly answering a different
# question.

APPROVAL_PREFIXES = frozenset({
    "yes", "yeah", "yep", "yup", "ya", "y",
    "ok", "okay", "alright", "k",
    "go", "go ahead", "do it", "proceed", "sure", "approved",
    "fix it", "let her do it", "let him do it",
})

# Phrases that match exactly (no prefix) — these are unambiguous and
# adding them to APPROVAL_PREFIXES would over-match.
APPROVAL_EXACT = frozenset({
    "yes", "yeah", "yep", "yup", "ya", "y",
    "do it", "go ahead", "proceed", "sure", "approved",
    "fix it", "let her do it", "let him do it",
    "ok", "okay", "alright", "go", "k",
})

# Sentences that contain an approval word but are clearly NOT permissions
# (e.g. "yes, but first I have a question about the approach"). The user's
# intent is to continue talking — keep the delegation pending and answer
# their question normally.
APPROVAL_BLOCKLIST = (
    "but", "however", "first", "wait", "actually", "though",
    "explain", "question", "how", "what", "why", "when", "where",
    "tell me", "show me", "before",
)

# Trailing punctuation stripped from the first word before prefix matching.
# "yes," → "yes", "yeah." → "yeah", "y;" → "y".
_APPROVAL_FIRST_WORD_STRIP = ',"\'.!?;:'

DENIAL_PHRASES = frozenset({
    "no", "nope", "nah", "na",
    "don't", "dont",
    "cancel", "stop", "never mind", "nevermind",
    "forget it", "forget it then", "skip",
})

# Phrase starts (lowercase) that suggest a clarification question
# rather than a yes/no answer. When a pending delegation exists and
# the user's message starts with one of these, we keep the
# delegation pending and answer the question normally.
CLARIFICATION_STARTS = (
    "what", "which", "how", "will it", "can it",
    "should i", "do you", "is it", "when", "where",
    "why", "who", "are you",
)


# ─────────────────────────────────────────────────────────────────
# Phrase classification
# ─────────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    """Lowercase + strip + collapse internal whitespace."""
    return " ".join((text or "").strip().lower().split())


def is_approval(text: str, _debug_pending: bool = False) -> bool:
    """True if the user's reply is an unambiguous approval.

    Matching rules (in order):
      1. Exact match against APPROVAL_EXACT — "yes", "do it", "go ahead"
         all count.
      2. Prefix match: if the first whitespace-delimited word, stripped of
         trailing punctuation, is an approval prefix (e.g. "yes, go ahead",
         "yeah!" → "yeah", "y; do it" → "y") AND the message does not
         contain any APPROVAL_BLOCKLIST conjunction ("but", "however",
         "first", "question"...) that would indicate the user is
         continuing the conversation.

    A trailing "?" or any blocklist word suppresses the prefix match so
    that "yes but how long will it take?" still falls through to
    clarification handling instead of firing a handoff mid-question.

    Args:
        text: The raw user reply.
        _debug_pending: When True, emit structured debug events on prefix
            match failure so the controller can log exactly why the approval
            was not recognised. Internal use only — callers should not pass
            this unless they are the controller and are logging.
    """
    norm = _normalize(text)
    if not norm:
        return False

    # Exact match always wins (after stripping trailing punctuation from the
    # full normalized string so "yes," → "yes" still matches the exact set).
    norm_stripped = norm.rstrip(_APPROVAL_FIRST_WORD_STRIP)
    if norm_stripped in APPROVAL_EXACT:
        _emit_debug("DELEGATE_IS_APPROVAL_EXACT", {"text": text, "norm": norm, "norm_stripped": norm_stripped})
        return True

    # Prefix match: first word, stripped of trailing punctuation, is an
    # approval word.  This handles "yes, go ahead", "yeah.", "y; do it", etc.
    raw_first = norm.split(" ", 1)[0]
    first_word = raw_first.rstrip(_APPROVAL_FIRST_WORD_STRIP)

    if first_word not in APPROVAL_PREFIXES:
        if _debug_pending:
            _emit_debug("DELEGATE_IS_APPROVAL_NO_PREFIX", {
                "text": text,
                "norm": norm,
                "raw_first_word": raw_first,
                "first_word_after_strip": first_word,
                "in_approval_prefixes": first_word in APPROVAL_PREFIXES,
            })
        return False

    # A question mark means the user is asking something, not approving.
    if norm.endswith("?"):
        if _debug_pending:
            _emit_debug("DELEGATE_IS_APPROVAL_QUESTION", {"text": text, "norm": norm})
        return False

    # If the reply contains a blocklist word, treat it as a continuation
    # of the conversation, not a bare approval.
    blocked_by = None
    for blocked in APPROVAL_BLOCKLIST:
        # Word-boundary match so "button" doesn't trigger on "but".
        if re.search(r"\b" + re.escape(blocked) + r"\b", norm):
            blocked_by = blocked
            break

    if blocked_by:
        if _debug_pending:
            _emit_debug("DELEGATE_IS_APPROVAL_BLOCKLIST", {
                "text": text,
                "norm": norm,
                "first_word": first_word,
                "blocked_by": blocked_by,
            })
        return False

    _emit_debug("DELEGATE_IS_APPROVAL_MATCH", {
        "text": text,
        "norm": norm,
        "first_word": first_word,
    })
    return True


def is_denial(text: str) -> bool:
    """True iff text is an exact match of a denial phrase.

    The exact-match rule is deliberately strict: "no" alone is short enough
    that any sentence containing it is almost certainly answering a different
    question ("no, I want to ask something"). Trailing punctuation on the
    full normalized string IS allowed ("no.", "nope!") so a simple trailing
    punctuation from the terminal doesn't suppress an explicit denial.
    """
    norm = _normalize(text)
    if not norm:
        return False
    if norm in DENIAL_PHRASES:
        return True
    # Allow trailing punctuation at the end of the whole string: "no,", "nope."
    # (NOT in the middle — "no, I want to" is a clarification, not a denial.)
    norm_punct = norm.rstrip(_APPROVAL_FIRST_WORD_STRIP)
    return norm_punct in DENIAL_PHRASES


def is_clarification(text: str) -> bool:
    """True iff text looks like a clarification question. Used by the
    fallback-brain honesty guard to decide whether a fallback turn is
    asking a question (legitimate) vs. narrating a fake handoff (guarded).

    Trailing punctuation on the first word is stripped so "what, that one?"
    is still recognised as a clarification.
    """
    t = _normalize(text)
    if not t:
        return False
    if t.endswith("?"):
        return True
    first_word = t.split(" ", 1)[0].rstrip(_APPROVAL_FIRST_WORD_STRIP)
    return any(first_word == start.split(" ", 1)[0] for start in CLARIFICATION_STARTS)


# ─────────────────────────────────────────────────────────────────
# No-op shims (v2 — delegation pipeline removed)
#
# These are kept for import compatibility. The pending state machine was
# removed in the v1→v2 redesign; these functions are now no-ops.
# Tests that import them will pass but the pending machinery tests
# in test_delegate_state.py are now skipped.
# ─────────────────────────────────────────────────────────────────

def clear_pending() -> None:
    """No-op. The pending delegation state machine was removed in v2."""


def set_pending(delegation: object) -> None:
    """No-op. The pending delegation state machine was removed in v2."""


def get_pending() -> None:  # type: ignore[return-value]
    """No-op. The pending delegation state machine was removed in v2."""
    return None


class PendingDelegation:
    """No-op placeholder. The pending delegation dataclass was removed in v2.

    Kept so that tests that import ``PendingDelegation`` don't break.
    """

    def __init__(self, original_request: str = "", project_root: str = "", **kwargs) -> None:
        pass

    @property
    def is_expired(self) -> bool:
        return True  # Always expired — pending delegation never active in v2.

