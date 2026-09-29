#!/usr/bin/env python3
"""
Intent resolution layer for Fairy.

Pre-processing reasoning stage that runs on EVERY user message BEFORE command
routing. It produces a structured interpretation:

    {raw_input, candidate_meanings, chosen_meaning, confidence, reasoning_trace}

Context-based disambiguation logic:
  - Generate multiple candidate interpretations for ambiguous terms (LLM-assisted).
  - Check each candidate against the stated task's requirements: what capability
    does the task need (open something? create files? play something?) and
    discard candidates that can't fulfil it (games can't make folders).
  - Use world knowledge: what is this term most known for? What's the typical way
    people interact with it (launcher vs website vs app)?
  - Autonomous research fallback: when the term is unknown (not in the local
    table AND the LLM isn't confident), use the existing web-research tool to
    search what the term refers to and how it's used (capped at a few seconds /
    one search) then decide. Log the research step.
  - Clarification gate (deterministic, not LLM-judged): if the top two
    interpretations are close in confidence AND meaningfully different (game vs
    dev-tool vs website), ask the user. If one interpretation dominates (high
    confidence gap), proceed and state the interpretation in the reply.
  - Wire into routing: the resolved interpretation feeds the existing pipelines
    (site-opener, tool executor, task delegation, plain chat) - don't duplicate
    routing logic, feed it.
  - Voice-awareness: Whisper transcripts produce near-miss words ("faury",
    "hoyola") so the resolver's fuzzy matching must be the norm, not the exception.
  - Log the reasoning trace (candidates -> scores -> decision) to the debug log
    for every resolved turn.

The resolver is designed to add ZERO noticeable latency to simple, unambiguous
messages: a fast-path skip returns an explicit "no ambiguity" signal without
any LLM call or web search.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the project root is on sys.path so sibling imports work regardless of
# how the module is loaded (pytest, main.py, etc.).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger("fairy.intent_resolver")

# Optional import of web research helpers (not used directly here; we use the
# raw web_search skill via the agent_controller dispatcher path). Kept as a
# module-level sentinel so the optional code path can short-circuit cleanly.
_maybe_research = None
try:
    from controller.web_research import maybe_research as _maybe_research
except Exception:
    _maybe_research = None

# ─── Debug logging (mirrors agent_controller._debug) ───────────────────────────

_DEBUG = os.environ.get("FAIRY_DEBUG", "0").strip() in ("1", "true", "yes", "on")
_DEBUG_FILE = os.environ.get(
    "FAIRY_DEBUG_FILE",
    os.path.join(os.path.dirname(__file__), "..", "fairy_debug.log"),
)


def _debug(label: str, data: Any = None) -> None:
    """Append a structured INTENT_RESOLVER line to the debug log."""
    if not _DEBUG:
        return
    try:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if data is None:
            line = f"[{ts}] {label}"
        else:
            try:
                payload = json.dumps(data, ensure_ascii=False, default=str)
            except Exception as enc_exc:
                payload = f"<unserializable: {enc_exc}>"
            line = f"[{ts}] INTENT_RESOLVER | {label} | {payload}"
        with open(_DEBUG_FILE, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(line + "\n")
            fh.flush()
    except Exception:
        pass


# ─── Data structures ────────────────────────────────────────────────────────────


@dataclass
class CandidateMeaning:
    """A single candidate interpretation for an ambiguous term."""

    term: str
    meaning: str
    category: str  # "dev_tool" | "game" | "website" | "app" | "unknown"
    capability: str  # what this candidate can do
    confidence: float  # 0.0–1.0
    evidence: List[str] = field(default_factory=list)
    source: str = "local_knowledge"  # local_knowledge | llm_reasoning | web_research
    can_fulfill_task: Optional[bool] = None  # set after task-capability check


@dataclass
class IntentResolution:
    """Structured output from the intent resolver."""

    raw_input: str
    candidate_meanings: List[CandidateMeaning]
    chosen_meaning: Optional[CandidateMeaning]
    confidence: float  # confidence in chosen_meaning
    reasoning_trace: List[Dict[str, Any]]
    clarification_question: Optional[str] = None
    research_performed: bool = False
    research_summary: str = ""
    # Routing hint feeds the existing pipelines (site-opener, tool executor,
    # task delegation, plain chat). We do NOT duplicate routing logic; we
    # produce a hint that handle_request / Hermes can consume.
    routing_hint: Optional[str] = None  # claude_code_delegate | browser | app_action | chat
    # When True, the resolver is confident this input is unambiguous and the
    # existing routing can handle it without the resolver's help.
    fast_path_eligible: bool = False
    # Visible reasoning the caller can surface in the reply ("Using Claude Code
    # for that - a game can't make folders").
    visible_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_input": self.raw_input,
            "candidate_meanings": [asdict(c) for c in self.candidate_meanings],
            "chosen_meaning": asdict(self.chosen_meaning) if self.chosen_meaning else None,
            "confidence": self.confidence,
            "reasoning_trace": self.reasoning_trace,
            "clarification_question": self.clarification_question,
            "research_performed": self.research_performed,
            "research_summary": self.research_summary[:200],
            "routing_hint": self.routing_hint,
            "fast_path_eligible": self.fast_path_eligible,
            "visible_reasoning": self.visible_reasoning,
        }


# ─── Local knowledge base ───────────────────────────────────────────────────────
# Known ambiguous terms and their interpretations. This is the first lookup
# before any LLM/web-search fallback. Each entry maps a canonical term to a list
# of candidate meanings ordered by default confidence.

# Category values: "dev_tool" | "game" | "website" | "app" | "unknown"
_KNOWN_TERMS: Dict[str, List[Dict[str, Any]]] = {
    "cod": [
        {
            "meaning": "Call of Duty (video game)",
            "category": "game",
            "capability": "entertainment, multiplayer FPS gaming",
            "confidence": 0.55,
            "evidence": ["'cod' is the common abbreviation for Call of Duty"],
        },
        {
            "meaning": "Claude Code (CLI coding tool)",
            "category": "dev_tool",
            "capability": "file operations, code editing, repository tasks",
            "confidence": 0.45,
            "evidence": ["sounds like 'cod'; Claude Code is a dev tool that can make folders"],
        },
    ],
    "claude code": [
        {
            "meaning": "Claude Code CLI",
            "category": "dev_tool",
            "capability": "file operations, code editing, repository tasks",
            "confidence": 0.98,
            "evidence": ["Anthropic's Claude Code CLI"],
        },
    ],
    "call of duty": [
        {
            "meaning": "Call of Duty (video game)",
            "category": "game",
            "capability": "entertainment, multiplayer FPS gaming",
            "confidence": 0.99,
            "evidence": ["popular first-person shooter franchise", "played via Steam/Epic"],
        },
    ],
    "fortnite": [
        {
            "meaning": "Fortnite (video game)",
            "category": "game",
            "capability": "entertainment, battle royale",
            "confidence": 0.99,
            "evidence": ["Epic Games battle royale"],
        },
    ],
    "minecraft": [
        {
            "meaning": "Minecraft (video game)",
            "category": "game",
            "capability": "entertainment, sandbox building",
            "confidence": 0.99,
            "evidence": ["Mojang sandbox game"],
        },
    ],
    "gpt": [
        {
            "meaning": "ChatGPT / OpenAI GPT",
            "category": "website",
            "capability": "chat, code, analysis",
            "confidence": 0.9,
            "evidence": ["OpenAI GPT models", "chat.openai.com"],
        },
    ],
    "youtube": [
        {
            "meaning": "YouTube video platform",
            "category": "website",
            "capability": "video streaming, search",
            "confidence": 0.99,
            "evidence": ["youtube.com"],
        },
    ],
    "discord": [
        {
            "meaning": "Discord communication app",
            "category": "app",
            "capability": "chat, voice, video",
            "confidence": 0.95,
            "evidence": ["discord.com", "desktop app"],
        },
    ],
    "github": [
        {
            "meaning": "GitHub code hosting",
            "category": "website",
            "capability": "code hosting, collaboration",
            "confidence": 0.99,
            "evidence": ["github.com"],
        },
    ],
}

# Whitelist of clearly websites (skip resolver entirely)
_KNOWN_WEBSITES = frozenset({
    "youtube", "youtube.com", "youtu.be", "reddit", "reddit.com",
    "twitter", "twitter.com", "x.com", "facebook", "facebook.com",
    "instagram", "instagram.com", "tiktok", "tiktok.com", "twitch",
    "twitch.tv", "discord", "discord.com", "github", "github.com",
    "gitlab", "gitlab.com", "stackoverflow", "stackoverflow.com",
    "gmail", "gmail.com", "google", "google.com", "wikipedia",
    "wikipedia.org", "netflix", "netflix.com", "spotify",
    "spotify.com", "linkedin", "linkedin.com",
})

# Whitelist of clearly desktop apps (skip resolver for "open X")
_KNOWN_DESKTOP_APPS = frozenset({
    "notepad", "notepad++", "word", "excel", "powerpoint", "calc",
    "calculator", "powershell", "cmd", "terminal", "vscode",
    "visual studio code", "steam", "epic games", "origin", "uplay",
    "vlc", "spotify", "discord", "teams", "zoom", "skype",
    "telegram", "whatsapp", "signal", "chrome", "firefox", "edge",
    "opera", "brave",
})

# Words that strongly indicate file-system / dev tasks (games can't do these).
_FILE_TASK_VERBS = (
    "make a folder", "create folder", "make folder", "create directory",
    "make a file", "create file", "write file", "save file", "delete file",
    "remove file", "edit file", "build project", "build a project",
    "code", "script", "python", "javascript", "bug fix", "debug",
    "refactor", "run tests", "write tests", "commit", "repository",
    "project", "zip", "compress", "extract",
)

# Words that strongly indicate the user wants to launch/open something.
_LAUNCH_VERBS = ("open", "launch", "start", "run", "close", "kill", "quit", "stop")

# Patterns that are unambiguous and can skip the resolver entirely.
_FAST_SKIP_PATTERNS = re.compile(
    r"^(what\s+time|what'?s\s+the\s+time|current\s+time|what\s+day|what\s+date|"
    r"cpu|gpu|memory|ram|system|weather|what'?s\s+up|sup|hey|hello|hi\s|yo\s|"
    r"thanks|thank\s+you|bye|goodbye)\b",
    re.IGNORECASE,
)

# Thresholds for the clarification gate.
_CLARIFY_CONFIDENCE_GAP = 0.18  # If top two are within this, consider asking.
_CLARIFY_MIN_CONFIDENCE = 0.45  # Don't ask if even the top candidate is weak.
_RESEARCH_CONFIDENCE_THRESHOLD = 0.40  # Below this, try web research.
_LLM_CONFIDENCE_THRESHOLD = 0.55  # Above this, trust the LLM/local table.

# Cap on web research time so the resolver never blocks for long.
_RESEARCH_TIMEOUT_SECONDS = 8.0


# ─── Fuzzy matching (Whisper near-miss) ─────────────────────────────────────────


def _fuzzy_match(term: str, candidates: List[str], threshold: float = 0.80) -> Optional[str]:
    """Return the closest candidate to `term` with a similarity ratio >= threshold.

    Uses difflib.SequenceMatcher (token_sort_ratio-like). This is the norm for
    voice transcripts where "faury" should match "fairy", "hoyola" -> "hiya", etc.
    """
    if not term or not candidates:
        return None
    term_lower = term.lower().strip()
    best: Optional[Tuple[float, str]] = None
    for cand in candidates:
        ratio = SequenceMatcher(None, term_lower, cand.lower().strip()).ratio()
        if best is None or ratio > best[0]:
            best = (ratio, cand)
    if best and best[0] >= threshold:
        return best[1]
    return None


def _normalize_term(text: str) -> str:
    """Lowercase, strip, collapse whitespace, drop trailing punctuation."""
    return re.sub(r"\s+", " ", text.lower().strip().strip(".,!?;:\"'"))


# ─── Term extraction ───────────────────────────────────────────────────────────


_OPEN_RE = re.compile(
    r"^\s*(?:open|launch|start|run|close|kill|quit|stop|activate)\s+(.+?)(?:\s+(?:for\s+me|please|now))?\s*$",
    re.IGNORECASE,
)
_USE_TOOL_RE = re.compile(
    r"^\s*use\s+(\S+)(?:\s+(?:for|to)\s+.+)?\s*$",
    re.IGNORECASE,
)


def _extract_subject(raw_input: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the ambiguous subject term and the action verb from the input.

    Returns (verb, subject) where verb is one of: open|use|make|launch|start|
    run|close|None. `subject` is the noun the user is referring to (e.g. "cod",
    "youtube", "zorblat"). Returns (None, None) when no subject is found.
    """
    text = raw_input.strip()
    # "use cod to make a folder" → verb=use, subject=cod
    m = _USE_TOOL_RE.match(text)
    if m:
        subject = _normalize_term(m.group(1))
        return "use", subject
    # "open cod" / "launch steam" → verb=open/launch, subject=cod/steam
    m = _OPEN_RE.match(text)
    if m:
        subject = _normalize_term(m.group(1))
        # Strip trailing "app"/"website" qualifiers
        for suffix in (" app", " website", " the game", " game"):
            if subject.endswith(suffix):
                subject = subject[: -len(suffix)].strip()
        if subject:
            return m.group(1).lower(), subject
    # "make a folder" style — no explicit tool, but a file task
    if any(v in text.lower() for v in _FILE_TASK_VERBS):
        # Look for "use X to ..." already handled; otherwise the subject is None
        # — the file task itself is the action, not a tool to resolve.
        return None, None
    return None, None


# ─── Task-capability check ──────────────────────────────────────────────────────


def _task_requires_file_ops(text: str) -> bool:
    """True if the user's request needs file-system / dev capabilities."""
    t = text.lower()
    return any(v in t for v in _FILE_TASK_VERBS)


def _task_is_launch_only(text: str) -> bool:
    """True if the request is purely 'open/launch X' with no file/dev task."""
    t = text.lower().strip()
    has_launch = any(t.startswith(v + " ") for v in _LAUNCH_VERBS) or any(
        (" " + v + " ") in (" " + t + " ") for v in _LAUNCH_VERBS
    )
    return has_launch and not _task_requires_file_ops(t)


# ─── Candidate generation ───────────────────────────────────────────────────────


def _local_candidates(subject: str) -> List[CandidateMeaning]:
    """Look up the subject in the local knowledge base (with fuzzy matching)."""
    out: List[CandidateMeaning] = []
    # Exact match first.
    if subject in _KNOWN_TERMS:
        for entry in _KNOWN_TERMS[subject]:
            out.append(CandidateMeaning(
                term=subject,
                meaning=entry["meaning"],
                category=entry["category"],
                capability=entry["capability"],
                confidence=entry["confidence"],
                evidence=entry.get("evidence", []),
                source="local_knowledge",
            ))
        return out
    # Fuzzy match against known terms (Whisper near-miss).
    all_known = list(_KNOWN_TERMS.keys())
    match = _fuzzy_match(subject, all_known, threshold=0.80)
    if match:
        for entry in _KNOWN_TERMS[match]:
            out.append(CandidateMeaning(
                term=subject,
                meaning=entry["meaning"],
                category=entry["category"],
                capability=entry["capability"],
                confidence=max(0.0, entry["confidence"] - 0.10),  # small penalty for fuzzy
                evidence=entry.get("evidence", []) + [f"fuzzy match: '{subject}' ~ '{match}'"],
                source="local_knowledge",
            ))
    return out


def _llm_candidates(subject: str, verb: Optional[str], raw_input: str,
                    brain_fn: Optional[Any]) -> Tuple[List[CandidateMeaning], str]:
    """Ask the LLM to generate candidate meanings for an unknown subject.

    Returns (candidates, research_hint) where research_hint is a short query
    string to feed web research if the LLM was not confident. brain_fn is the
    agent_controller._brain callable (Ollama -> OpenRouter fallback).
    """
    if brain_fn is None:
        return [], ""

    prompt = (
        "You are Fairy's intent resolver. A user said:\n"
        f"  \"{raw_input}\"\n\n"
        f"The ambiguous term is: \"{subject}\"\n"
        f"The action verb is: \"{verb or 'none'}\"\n\n"
        "List up to 3 possible meanings of this term. For each, give:\n"
        "  meaning: short name\n"
        "  category: one of dev_tool, game, website, app, unknown\n"
        "  capability: what it can do\n"
        "  confidence: 0.0-1.0 how likely this is the intended meaning given the action\n\n"
        "Reply ONLY with a JSON array. Example:\n"
        "[{\"meaning\":\"Claude Code CLI\",\"category\":\"dev_tool\","
        "\"capability\":\"file operations, code editing\",\"confidence\":0.7}]\n"
    )
    try:
        from controller.agent_controller import _brain as _ctrl_brain, MODEL_BRAIN
        resp = _ctrl_brain(
            MODEL_BRAIN,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 200, "num_ctx": 512, "temperature": 0.1},
        )
        content = (resp.get("message", {}) or {}).get("content", "") or ""
    except Exception as exc:
        _debug("LLM_CANDIDATES_FAIL", {"error": str(exc)})
        return [], ""

    # Detect error patterns from Gemma4 or other models that return errors as text
    content_lower = content.lower()
    error_patterns = [
        "context length exceeded",
        "cannot compress",
        "context too long",
        "too many tokens",
    ]
    if any(pattern in content_lower for pattern in error_patterns):
        _debug("LLM_CANDIDATES_CONTEXT_ERROR", {"content": content[:200]})
        # Fall back to web search for ambiguous terms
        return [], f'what is "{subject}"'

    candidates: List[CandidateMeaning] = []
    try:
        # Extract JSON array from the response.
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            arr = json.loads(content[start : end + 1])
            for item in arr:
                if not isinstance(item, dict):
                    continue
                conf = float(item.get("confidence", 0.5))
                candidates.append(CandidateMeaning(
                    term=subject,
                    meaning=str(item.get("meaning", "")),
                    category=str(item.get("category", "unknown")),
                    capability=str(item.get("capability", "")),
                    confidence=max(0.0, min(1.0, conf)),
                    evidence=["llm_reasoning"],
                    source="llm_reasoning",
                ))
    except Exception as exc:
        _debug("LLM_CANDIDATES_PARSE_FAIL", {"error": str(exc), "raw": content[:200]})

    # A research hint: if the LLM had no strong candidate, search the term.
    research_hint = ""
    if candidates and max((c.confidence for c in candidates), default=0.0) < _LLM_CONFIDENCE_THRESHOLD:
        research_hint = f'what is "{subject}" software or app'
    elif not candidates:
        research_hint = f'what is "{subject}"'

    return candidates, research_hint


# ─── Web research fallback ──────────────────────────────────────────────────────


def _web_research_term(query: str) -> Tuple[str, bool]:
    """Run a single web search to learn what an unknown term refers to.

    Returns (summary, performed). `performed` is False if the search was
    skipped or failed. Capped at _RESEARCH_TIMEOUT_SECONDS so the resolver
    never blocks for long.
    """
    if _maybe_research is None:
        # The web_research module is unavailable; fall through to using
        # the web_search skill directly via the dispatcher path below.
        pass
    try:
        import threading

        result_box: Dict[str, Any] = {"summary": "", "ok": False}

        def _do_search() -> None:
            try:
                # Use the existing web_search skill via the dispatcher path.
                from controller import agent_controller as _ac

                raw = _ac.web_search(query)
                if isinstance(raw, str):
                    result_box["summary"] = raw[:800]
                    result_box["ok"] = True
                elif isinstance(raw, dict):
                    # web_search returns a dict with "results" list
                    results = raw.get("results", [])
                    snippets = []
                    for r in results[:3]:
                        title = r.get("title", "")
                        body = r.get("body", r.get("snippet", ""))
                        if title or body:
                            snippets.append(f"{title}: {body[:200]}")
                    if snippets:
                        result_box["summary"] = " | ".join(snippets)[:800]
                        result_box["ok"] = True
            except Exception as exc:
                result_box["summary"] = ""
                result_box["ok"] = False
                _debug("WEB_RESEARCH_FAIL", {"query": query, "error": str(exc)})

        t = threading.Thread(target=_do_search, daemon=True)
        t.start()
        t.join(timeout=_RESEARCH_TIMEOUT_SECONDS)
        if result_box.get("ok"):
            return str(result_box.get("summary", "")), True
        return "", False
    except Exception as exc:
        _debug("WEB_RESEARCH_EXCEPTION", {"query": query, "error": str(exc)})
        return "", False


# ─── Capability filtering ───────────────────────────────────────────────────────


def _filter_by_capability(candidates: List[CandidateMeaning], raw_input: str) -> List[CandidateMeaning]:
    """Discard candidates that cannot fulfil the stated task.

    A game can't make a folder. A website can't make a folder. If the user
    asked for a file-system task, dev_tool candidates survive; games/websites
    are down-weighted (not dropped outright — the user may be mistaken about
    the task, but we surface that in reasoning).

    When filtering narrows the field to a single viable candidate, that
    candidate is boosted: the filter itself is evidence of the right choice.
    """
    needs_files = _task_requires_file_ops(raw_input)
    if not needs_files:
        for c in candidates:
            c.can_fulfill_task = None
        return candidates

    for c in candidates:
        if c.category in ("game", "website"):
            c.can_fulfill_task = False
        elif c.category in ("dev_tool", "app"):
            c.can_fulfill_task = True
        else:
            c.can_fulfill_task = None
        if c.can_fulfill_task is False:
            # Down-weight but keep (for traceability).
            c.confidence *= 0.3

    # If only one candidate can fulfil the task and it was the underdog
    # before filtering, boost it — the filter's verdict is meaningful evidence.
    viable = [c for c in candidates if c.can_fulfill_task is True]
    if len(viable) == 1 and viable[0].confidence < 0.55:
        viable[0].confidence = min(viable[0].confidence * 2.0, 0.95)

    return candidates


# ─── Public entry point ─────────────────────────────────────────────────────────


def resolve_intent(raw_input: str,
                    brain_fn: Optional[Any] = None,
                    allow_research: bool = True) -> IntentResolution:
    """Resolve the intent of a user message.

    Args:
        raw_input: The raw user message.
        brain_fn: Optional LLM callable (agent_controller._brain) for candidate
            generation. If None, the resolver imports it lazily.
        allow_research: If True (default), the resolver may do a single web
            search to learn about an unknown term. Tests set this to False
            with mocked search to keep them deterministic.

    Returns an IntentResolution. Never raises — on any internal error it
    returns a resolution with fast_path_eligible=True so the caller falls
    through to existing routing unchanged.
    """
    trace: List[Dict[str, Any]] = []
    raw_input = (raw_input or "").strip()
    if not raw_input:
        return IntentResolution(
            raw_input="",
            candidate_meanings=[],
            chosen_meaning=None,
            confidence=0.0,
            reasoning_trace=[{"step": "empty_input", "action": "skip"}],
            fast_path_eligible=True,
        )

    _debug("RESOLVE_START", {"input": raw_input[:200]})

    # ── Fast path: unambiguous messages skip the resolver entirely ────────
    if _FAST_SKIP_PATTERNS.match(raw_input):
        trace.append({"step": "fast_path_skip", "reason": "unambiguous pattern matched"})
        _debug("FAST_PATH_SKIP", {"input": raw_input[:120]})
        return IntentResolution(
            raw_input=raw_input,
            candidate_meanings=[],
            chosen_meaning=None,
            confidence=1.0,
            reasoning_trace=trace,
            fast_path_eligible=True,
        )

    verb, subject = _extract_subject(raw_input)
    _debug("SUBJECT_EXTRACTED", {"verb": verb, "subject": subject})

    # If no subject was extracted, there's nothing to disambiguate. Let the
    # existing routing handle it.
    if not subject:
        trace.append({"step": "no_subject", "action": "skip_to_existing_routing"})
        return IntentResolution(
            raw_input=raw_input,
            candidate_meanings=[],
            chosen_meaning=None,
            confidence=0.0,
            reasoning_trace=trace,
            fast_path_eligible=True,
        )

    # ── Known website / desktop app: fast path ────────────────────────────
    if subject in _KNOWN_WEBSITES:
        trace.append({"step": "known_website", "subject": subject})
        chosen = CandidateMeaning(
            term=subject,
            meaning=subject,
            category="website",
            capability="web browsing",
            confidence=0.99,
            evidence=["known website"],
        )
        return IntentResolution(
            raw_input=raw_input,
            candidate_meanings=[chosen],
            chosen_meaning=chosen,
            confidence=0.99,
            reasoning_trace=trace,
            routing_hint="browser",
            fast_path_eligible=True,
            visible_reasoning="",
        )

    if subject in _KNOWN_DESKTOP_APPS:
        trace.append({"step": "known_desktop_app", "subject": subject})
        chosen = CandidateMeaning(
            term=subject,
            meaning=subject,
            category="app",
            capability="desktop application",
            confidence=0.97,
            evidence=["known desktop app"],
        )
        hint = "app_action"
        return IntentResolution(
            raw_input=raw_input,
            candidate_meanings=[chosen],
            chosen_meaning=chosen,
            confidence=0.97,
            reasoning_trace=trace,
            routing_hint=hint,
            fast_path_eligible=True,
            visible_reasoning="",
        )

    # ── Local knowledge base lookup (with fuzzy match) ──────────────────
    candidates = _local_candidates(subject)
    trace.append({"step": "local_lookup", "subject": subject, "count": len(candidates)})

    # ── LLM candidate generation (if local table didn't dominate) ───────
    top_local_conf = max((c.confidence for c in candidates), default=0.0)
    if top_local_conf < _LLM_CONFIDENCE_THRESHOLD and brain_fn is not None:
        llm_cands, research_hint = _llm_candidates(subject, verb, raw_input, brain_fn)
        if llm_cands:
            # Merge: prefer the source with higher confidence but keep both.
            candidates.extend(llm_cands)
            trace.append({"step": "llm_candidates", "count": len(llm_cands), "research_hint": research_hint})
    else:
        research_hint = ""

    # ── Web research fallback for unknown terms ──────────────────────────
    research_summary = ""
    research_performed = False
    if allow_research and not candidates:
        trace.append({"step": "web_research", "reason": "no local or LLM candidates"})
        research_summary, research_performed = _web_research_term(
            research_hint or f'what is "{subject}" software or app'
        )
        if research_performed and research_summary:
            # Feed the research back into the LLM for a second pass.
            llm_cands2, _ = _llm_candidates_with_context(
                subject, verb, raw_input, research_summary, brain_fn
            )
            if llm_cands2:
                candidates.extend(llm_cands2)
                trace.append({"step": "post_research_llm", "count": len(llm_cands2)})
            _debug("WEB_RESEARCH_PERFORMED", {"summary": research_summary[:200]})
        elif not research_performed:
            trace.append({"step": "web_research_skipped", "reason": "search failed or disabled"})

    if not candidates:
        # Genuinely unknown — ask the user or let existing routing handle it.
        trace.append({"step": "no_candidates", "action": "honest_unknown"})
        _debug("NO_CANDIDATES", {"subject": subject})
        return IntentResolution(
            raw_input=raw_input,
            candidate_meanings=[],
            chosen_meaning=None,
            confidence=0.0,
            reasoning_trace=trace,
            research_performed=research_performed,
            research_summary=research_summary,
            routing_hint=None,
            fast_path_eligible=False,
            visible_reasoning=f"I couldn't figure out what '{subject}' refers to, Master.",
        )

    # ── Capability filtering (games can't make folders) ──────────────────
    candidates = _filter_by_capability(candidates, raw_input)
    trace.append({
        "step": "capability_filter",
        "needs_files": _task_requires_file_ops(raw_input),
        "candidates": [
            {"meaning": c.meaning, "category": c.category, "confidence": c.confidence,
             "can_fulfill": c.can_fulfill_task}
            for c in candidates
        ],
    })

    # Sort by confidence descending.
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    if not candidates:
        return IntentResolution(
            raw_input=raw_input,
            candidate_meanings=[],
            chosen_meaning=None,
            confidence=0.0,
            reasoning_trace=trace,
            research_performed=research_performed,
            fast_path_eligible=False,
        )

    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    # ── Clarification gate (deterministic) ──────────────────────────────
    # If the top two are close in confidence AND meaningfully different
    # (different category), ask the user rather than silently guessing.
    ask_clarification = False
    clarification_question: Optional[str] = None
    if (
        second is not None
        and top.confidence >= _CLARIFY_MIN_CONFIDENCE
        and (top.confidence - second.confidence) < _CLARIFY_CONFIDENCE_GAP
        and top.category != second.category
        # Only ask when the categories are meaningfully different for the task.
        and {top.category, second.category} != {"website", "app"}
    ):
        ask_clarification = True
        # Build a short, natural clarifying question — derive a human-readable
        # short label from the category so the user gets e.g. "the game, or
        # Claude Code" not "(game) or (dev_tool)".
        _CATEGORY_SHORT: dict[str, str] = {
            "game": "the game",
            "dev_tool": "Claude Code",
            "website": "the website",
            "app": "the app",
        }
        def _short_label(c: CandidateMeaning) -> str:
            return _CATEGORY_SHORT.get(c.category, c.meaning)

        names = " or ".join(_short_label(c) for c in [top, second])
        clarification_question = f"{subject.capitalize()} — {names}, Master?"
        trace.append({"step": "clarification_gate", "top": top.meaning, "second": second.meaning})

    # ── Choose the meaning ───────────────────────────────────────────────
    chosen = top if not ask_clarification else None
    routing_hint: Optional[str] = None
    visible_reasoning = ""

    if chosen is not None:
        # Map category to a routing hint for the existing pipelines.
        if chosen.category == "dev_tool":
            routing_hint = "claude_code_delegate"
            if _task_requires_file_ops(raw_input):
                visible_reasoning = (
                    f"Using {chosen.meaning} for that — a game can't make folders."
                )
        elif chosen.category == "website":
            routing_hint = "browser"
        elif chosen.category == "app":
            routing_hint = "app_action"
        elif chosen.category == "game":
            routing_hint = "app_action"
            if _task_requires_file_ops(raw_input):
                # A game can't fulfil a file task — we'd have asked for
                # clarification, but if we're here, surface the mismatch.
                visible_reasoning = (
                    f"{chosen.meaning} is a game, Master — it can't make folders."
                )
        trace.append({
            "step": "chosen",
            "meaning": chosen.meaning,
            "category": chosen.category,
            "confidence": chosen.confidence,
            "routing_hint": routing_hint,
        })

    _debug("RESOLVE_RESULT", {
        "subject": subject,
        "chosen": chosen.meaning if chosen else None,
        "confidence": top.confidence if not ask_clarification else 0.0,
        "ask_clarification": ask_clarification,
        "routing_hint": routing_hint,
        "candidates": [c.meaning for c in candidates],
    })

    return IntentResolution(
        raw_input=raw_input,
        candidate_meanings=candidates,
        chosen_meaning=chosen,
        confidence=top.confidence if not ask_clarification else 0.0,
        reasoning_trace=trace,
        clarification_question=clarification_question if ask_clarification else None,
        research_performed=research_performed,
        research_summary=research_summary,
        routing_hint=routing_hint,
        fast_path_eligible=False,
        visible_reasoning=visible_reasoning,
    )


def _llm_candidates_with_context(subject: str, verb: Optional[str], raw_input: str,
                                  research: str, brain_fn: Optional[Any]) -> Tuple[List[CandidateMeaning], str]:
    """Second-pass LLM candidate generation with web research context.

    Returns (candidates, research_hint) — research_hint is always "" here
    (we already researched).
    """
    if brain_fn is None:
        return [], ""
    prompt = (
        "You are Fairy's intent resolver. A user said:\n"
        f"  \"{raw_input}\"\n\n"
        f"The ambiguous term is: \"{subject}\"\n\n"
        f"Web search results about this term:\n{research[:1500]}\n\n"
        "Based on the search results and the user's action, list up to 3 "
        "possible meanings. For each:\n"
        "  meaning: short name\n"
        "  category: one of dev_tool, game, website, app, unknown\n"
        "  capability: what it can do\n"
        "  confidence: 0.0-1.0\n\n"
        "Reply ONLY with a JSON array.\n"
    )
    try:
        from controller.agent_controller import _brain as _ctrl_brain, MODEL_BRAIN
        resp = _ctrl_brain(
            MODEL_BRAIN,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 200, "num_ctx": 512, "temperature": 0.1},
        )
        content = (resp.get("message", {}) or {}).get("content", "") or ""
    except Exception as exc:
        _debug("LLM_CANDIDATES_CTX_FAIL", {"error": str(exc)})
        return [], ""

    # Detect error patterns from Gemma4 or other models
    content_lower = content.lower()
    error_patterns = [
        "context length exceeded",
        "cannot compress",
        "context too long",
        "too many tokens",
    ]
    if any(pattern in content_lower for pattern in error_patterns):
        _debug("LLM_CANDIDATES_CTX_ERROR", {"content": content[:200]})
        return [], ""

    candidates: List[CandidateMeaning] = []
    try:
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1 and end > start:
            arr = json.loads(content[start : end + 1])
            for item in arr:
                if not isinstance(item, dict):
                    continue
                conf = float(item.get("confidence", 0.5))
                candidates.append(CandidateMeaning(
                    term=subject,
                    meaning=str(item.get("meaning", "")),
                    category=str(item.get("category", "unknown")),
                    capability=str(item.get("capability", "")),
                    confidence=max(0.0, min(1.0, conf)),
                    evidence=["web_research"],
                    source="web_research",
                ))
    except Exception as exc:
        _debug("LLM_CANDIDATES_CTX_PARSE_FAIL", {"error": str(exc)})
    return candidates, ""
