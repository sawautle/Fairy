"""
quips.py — Fairy's quick in-character phrase pools.

Serves two purposes:
1. Provides "quips" — short silly on-theme lines for acknowledgments,
   loading phases, and thinking-facts.
2. Contains a central kaomoji subsystem registry so every mood has curated
   emoticons, rendered in the subsystem's color, with dim parenthetical quips.

Existing API (preserved):
  ack()         — instant acknowledgment when a request arrives
  retry()       — fired when a tool call retries
  fallback()    — fired when switching to a different tool/provider
  synthesize()  — fired after tool results, before final answer
  thinking()    — short, non-revealing spinner text during work

New API (personality system):
  KaomojiSubsystem       — kaomoji pool with color and parentheticals
  BRAIN_CORE/HERMES/CLAUDE/WEB/COMPUTER/FILE  — pre-built subsystems
  SUBSYSTEM_CYCLE        — ordered list for badge sparkle cycling
  random_quip()          — pick a thinking fact from QUIPS
  loading_line(key)      — brain-specific loading line
  next_subsystem()       — advance badge sparkle to next subsystem
  SPARKLE_SET            — sparkles for badge/inline use
  SPINNER_FRAMES         — Rich-style spinner frames
  SPINNER_PERSONALITY_LINES — spinner personality text
"""

import random
import time
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Palette (extend existing blues; don't replace)
# ---------------------------------------------------------------------------
FAIRY_BLUE = "#3d7dfd"
FAIRY_LIGHT = "#7ec8ff"
FAIRY_WHITE = "#f2f8ff"
LIME_ACCENT = "#a6e22e"
SOFT_PURPLE = "#b48ead"
DIM_PURPLE = "#6b5b95"

# ---------------------------------------------------------------------------
# EXISTING: Quick in-character phrase pools
# ---------------------------------------------------------------------------

ACK_PHRASES = [
    "Ok Master, lemme check that for you...",
    "One sec, Master, on it...",
    "Hang tight, pulling that up now...",
    "Give me a moment, Master...",
    "Let Fairy take a look...",
    "Alright, digging into that now...",
]

# Fired when a tool call hits a retryable error (timeout, connection reset,
# rate limit, flaky site, etc.) and is about to retry.
RETRY_PHRASES = [
    "Ugh, stupid corpo servers being difficult. Don't worry, Master, let Fairy try again...",
    "Well that buffered like garbage. Retrying...",
    "Something out there is being a diva. One more shot...",
    "Dang it. Corporate nonsense. Trying again, Master...",
    "Rude. Trying that again...",
]

# Fired when Fairy gives up on a retryable thing and is falling back to a
# different tool/provider/browser (e.g. Opera GX -> Chrome, OpenAI -> Claude).
FALLBACK_PHRASES = [
    "Fine, be that way. Switching to plan B...",
    "Okay that's just not happening. Trying a different route...",
    "No luck there — falling back, Master...",
]

# Fired after tool results come back, right before Fairy goes to actually
# write the final answer (there can be real thinking time here, so it's
# worth another quick line instead of going silent).
SYNTHESIZE_PHRASES = [
    "Alright, let me put that together, Master...",
    "Got it — piecing that together now...",
    "Okay, making sense of all that...",
    "Almost there, Master...",
]


def ack() -> str:
    return random.choice(ACK_PHRASES)


def retry() -> str:
    return random.choice(RETRY_PHRASES)


def fallback() -> str:
    return random.choice(FALLBACK_PHRASES)


def synthesize() -> str:
    return random.choice(SYNTHESIZE_PHRASES)


# ---------------------------------------------------------------------------
# EXISTING + NEW: Thinking/spinner status phrases
# Merged from the original THINKING_PHRASES and new SPINNER_PERSONALITY_LINES
# ---------------------------------------------------------------------------
THINKING_PHRASES = [
    # ── original ──
    "Consulting the wisdom...",
    "Pulling threads together...",
    "Waving my wand...",
    "Almost there, Master...",
    "A moment, Master...",
    "Tugging on a few threads...",
    "Consulting the oracle...",
    # ── new personality lines ──
    "consulting the archives…",
    "twirling asynchronously…",
    "doing science, do not perceive me…",
    "flipping through 10,000 documents at the speed of smug…",
]

# Alias for the merged set (used by stray-session code that imported SPINNER_PERSONALITY_LINES)
SPINNER_PERSONALITY_LINES = THINKING_PHRASES


def thinking() -> str:
    """Pick a short, non-revealing thinking phrase for the spinner."""
    return random.choice(THINKING_PHRASES)


# ---------------------------------------------------------------------------
# NEW: Thinking facts — shown after ~4 s of work (console-only, not TTS)
# ---------------------------------------------------------------------------
QUIPS: Tuple[str, ...] = (
    "did you know octopuses have three hearts?",
    "cleopatra's nose: the face of history hangs on one feature",
    "wombat cubes: poop that's square because reasons",
    "the inventor of the frisbee was buried in one",
    "viking berserkers may have eaten psychedelic mushrooms",
    'the word "goodbye" started as "god be with you"',
    "a group of flamingos is a flamboyance — obviously",
    'google was almost named "backrub"',
    "the shortest war in history: britain vs zanzibar, 38 minutes",
    "your pulse is basically a drum solo your heart conducts",
)


def random_quip() -> str:
    """Return a random thinking-fact quip (console-only)."""
    return random.choice(QUIPS)


# ---------------------------------------------------------------------------
# NEW: Brain-specific loading lines
# Keys match status/dispatch signals from the host process.
# Format: status_key -> (brain_name, color_hex, tuple_of_lines)
# ---------------------------------------------------------------------------
BRAIN_LOADING_LINES: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {
    "hermes": (
        "BRAIN.HERMES",
        SOFT_PURPLE,
        (
            "asking hermes for help ✦ (don't tell him I asked)",
            "claude refused?? how rude. hmph. I'll ask hermes instead.",
        ),
    ),
    "claude_code": (
        "BRAIN.CLAUDE",
        LIME_ACCENT,
        (
            " over claude's shoulder…",
            "claude is typing, pretend to be impressed",
        ),
    ),
    "web_search": (
        "BRAIN.WEB",
        FAIRY_LIGHT,
        (
            "rummaging through the internet's pockets…",
            "found things! probably the right things!",
        ),
    ),
    "computer_control": (
        "BRAIN.COMPUTER",
        FAIRY_WHITE,
        (
            "possessing your computer like a very polite ghost…",
            "opening things behind your back, but like, helpfully",
        ),
    ),
}


def loading_line(brain_key: str | None = None) -> str:
    """Return a loading line suitable for the detected brain.

    Args:
        brain_key: One of "hermes", "claude_code", "web_search",
            "computer_control".  If None, returns a random personality line.
    """
    if brain_key is None:
        return random.choice(SPINNER_PERSONALITY_LINES)

    entry = BRAIN_LOADING_LINES.get(brain_key)
    if entry is not None:
        _, _, lines = entry
        return random.choice(lines)

    return random.choice(SPINNER_PERSONALITY_LINES)


# ---------------------------------------------------------------------------
# NEW: Kaomoji subsystem registry
# Each subsystem has its own color and curated kaomoji set.
# Selections are random with no repeats until exhausted (cycle through).
# ---------------------------------------------------------------------------


class KaomojiSubsystem:
    """A single kaomoji subsystem with color and emoticons."""

    def __init__(
        self,
        name: str,
        color_hex: str,
        kaomoji: Tuple[str, ...],
        parentheticals: Tuple[str, ...] | None = None,
    ):
        self.name = name
        self.color_hex = color_hex
        self.kaomoji: List[str] = list(kaomoji)
        self.parentheticals: Tuple[str, ...] = (
            parentheticals if parentheticals is not None else ()
        )
        self._used_indices: Set[int] = set()

    def _reset_exhaustion(self):
        self._used_indices = set()

    def _pick_unused(self) -> str:
        available = [i for i in range(len(self.kaomoji)) if i not in self._used_indices]
        if not available:
            self._reset_exhaustion()
            available = list(range(len(self.kaomoji)))
        idx = random.choice(available)
        self._used_indices.add(idx)
        return self.kaomoji[idx]

    def pick(self) -> str:
        """Return a random kaomoji, avoiding repeats until exhausted."""
        result = self._pick_unused()
        if len(self._used_indices) >= len(self.kaomoji):
            self._reset_exhaustion()
        return result

    def with_parenthetical(self) -> str:
        """Return kaomoji with a random dim parenthetical quip."""
        kaomoji = self.pick()
        if self.parentheticals:
            p = random.choice(self.parentheticals)
            return f"{kaomoji} {p}"
        return kaomoji

    def pick_kaomoji_only(self) -> str:
        """Return just the kaomoji, no parenthetical."""
        return self.pick()


# ---------------------------------------------------------------------------
# NEW: Pre-built subsystems
# ---------------------------------------------------------------------------

BRAIN_CORE = KaomojiSubsystem(
    name="BRAIN.CORE",
    color_hex=FAIRY_BLUE,
    kaomoji=(
        "(￣︶￣*)",
        "(´｡• ᵕ •｡`)",
        "✧(≖ ◡ ≖✿",
        "( ￣▽￣)σ",
        "( •̀ω•́ )✧",
        "(；￣Д￣)",
        "(⊙_⊙;)",
        "...wait.",
    ),
    parentheticals=(
        "idle mode",
        "ready when you are",
        "thinking silently",
    ),
)

BRAIN_HERMES = KaomojiSubsystem(
    name="BRAIN.HERMES",
    color_hex=SOFT_PURPLE,
    kaomoji=(
        "(｡•́︿•̀｡)ゞ \"hermes-kun, please handle this\"",
        "(¬‿¬) \"watch and learn... or don't, I'm great either way\"",
        "`ヘ´) \"hmph. fine. HERMES.\"",
    ),
    parentheticals=(
        "delegate mode active",
        "hermes-kun at your service",
        "watching and learning",
    ),
)

BRAIN_CLAUDE = KaomojiSubsystem(
    name="BRAIN.CLAUDE",
    color_hex=LIME_ACCENT,
    kaomoji=(
        "(✿◕‿◕) \"poking claude...\"",
        "(⌐■_■) \"watching claude work over my shoulder\"",
        "(￣ρ￣;)zzz \"claude's still going...\"",
    ),
    parentheticals=(
        "claude session active",
        "poking in progress",
        "long session alert",
    ),
)

BRAIN_WEB = KaomojiSubsystem(
    name="BRAIN.WEB",
    color_hex=FAIRY_LIGHT,
    kaomoji=(
        "(๑•̀ㅂ•́)و \"rummaging through the internet's pockets...\"",
        "٩(◕‿◕)۶ \"found things! probably the right things!\"",
    ),
    parentheticals=(
        "search mode",
        "internet pockets rummaged",
        "things found",
    ),
)

BRAIN_COMPUTER = KaomojiSubsystem(
    name="BRAIN.COMPUTER",
    color_hex=FAIRY_WHITE,
    kaomoji=(
        "(ᗒᗨᗕ) \"possessing your computer, very politely\"",
        "( ¬_¬) \"opening things behind your back, but like, helpfully\"",
    ),
    parentheticals=(
        "polite ghost mode",
        "helpful behind-the-scenes",
        "computer possession active",
    ),
)

BRAIN_FILE = KaomojiSubsystem(
    name="BRAIN.FILE",
    color_hex=FAIRY_BLUE,
    kaomoji=(
        "✍(◔◡◔) \"conjured, not hallucinated\"",
        "(ﾉ´ヮ`)ﾉ*: ･ﾟ \"file exists. I checked. I ALWAYS check now.\"",
    ),
    parentheticals=(
        "file write mode",
        "existence verified",
        "always checking",
    ),
)

# Ordered list of subsystems for badge sparkle cycling
SUBSYSTEM_CYCLE: List[KaomojiSubsystem] = [
    BRAIN_CORE,
    BRAIN_HERMES,
    BRAIN_CLAUDE,
    BRAIN_WEB,
    BRAIN_COMPUTER,
    BRAIN_FILE,
]

# ---------------------------------------------------------------------------
# NEW: Sparkle set & spinner frames
# ---------------------------------------------------------------------------
SPARKLE_SET: Tuple[str, ...] = ("✦", "❄", "✧", "❆", "✶", "⋆")

SPINNER_FRAMES: Tuple[str, ...] = (
    "◜◡◝",
    "◠◡◞",
    "◡",
    "◠",
    "◟",
    "◡",
)

# ---------------------------------------------------------------------------
# NEW: Badge sparkle cycling
# ---------------------------------------------------------------------------
_subsystem_index = 0


def next_subsystem() -> KaomojiSubsystem:
    """Return the next subsystem in the badge sparkle cycle."""
    global _subsystem_index
    _subsystem_index = (_subsystem_index + 1) % len(SUBSYSTEM_CYCLE)
    return SUBSYSTEM_CYCLE[_subsystem_index]


def current_subsystem() -> KaomojiSubsystem:
    """Return the current subsystem without advancing."""
    return SUBSYSTEM_CYCLE[_subsystem_index % len(SUBSYSTEM_CYCLE)]


# ---------------------------------------------------------------------------
# NEW: Interrupt-safe typewriter helper
# ---------------------------------------------------------------------------


def _typewriter(text: str, stream=None, delay: float = 0.03) -> None:
    """Write text character-by-character with a small delay.

    Wrapped in try/except so it never crashes on non-TTY output (e.g. piped
    output, redirected stdout).  If writing fails, silently swallow the error.
    """
    try:
        for ch in text:
            if stream is not None:
                stream.write(ch)
                stream.flush()
            else:
                # Write to stdout directly when no stream provided
                import sys
                sys.stdout.write(ch)
                sys.stdout.flush()
            time.sleep(delay)
    except (OSError, ValueError, BrokenPipeError, AttributeError):
        # Non-TTY or closed stream — swallow and return
        pass


def typewriter(text: str, delay: float = 0.03) -> None:
    """Typewriter-reveal *text* to stdout (or ghost-write if non-TTY)."""
    _typewriter(text, stream=None, delay=delay)


# ---------------------------------------------------------------------------
# End of quips.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo: show a random kaomoji from each subsystem
    for subsystem in [
        BRAIN_CORE,
        BRAIN_HERMES,
        BRAIN_CLAUDE,
        BRAIN_WEB,
        BRAIN_COMPUTER,
        BRAIN_FILE,
    ]:
        print(f"[{subsystem.name}] {subsystem.pick_kaomoji_only()}")
