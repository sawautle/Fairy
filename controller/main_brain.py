#!/usr/bin/env python3
"""
Provider-abstraction layer: Ollama as primary, OpenRouter `openrouter/free` as fallback.

This module provides a single `chat()` entry point that:
  1. Tries Ollama first (preferred, local, fast)
  2. Falls back to OpenRouter's universal free router `openrouter/free` if Ollama is unavailable
  3. Sticks to the selected provider for the duration of a request (no bouncing)

Tool support: passes Fairy's existing tool definitions (TOOLS list, already in
OpenAI schema format) to OpenRouter so the free model can attempt tool calls.
The OpenAI -> Ollama response-shape conversion handles tool_calls conversion.

No per-model cycling: the fallback always uses `openrouter/free` (OpenRouter's
universal free router that selects a free backend). No paid models are used,
no blacklist/rotation logic, no per-model retry.

Usage:
    from controller.main_brain import chat as brain_chat, is_ollama_available

    # Returns (response_dict, provider_name)
    # response_dict is Ollama ChatResponse-shaped (compatible with existing code)
    response, provider = brain_chat(
        model=MODEL_BRAIN,
        messages=[...],
        tools=TOOLS,
        options={"num_predict": 900, "num_ctx": 32768},
    )
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

# Config values (imported at module level; safe even if config not yet loaded)
try:
    from config import OLLAMA_URL, MODEL_BRAIN, OPENROUTER_KEY
except ImportError:
    OLLAMA_URL = "http://localhost:11434"
    MODEL_BRAIN = "gemma4"
    OPENROUTER_KEY = ""

# ─── Resilience layer ───────────────────────────────────────────────────────────
# Deferred import so the module can be imported before the resilience module exists
# during development. All production code goes through chat() which imports eagerly.
try:
    from controller.main_brain_resilience import (
        chat_with_resilience,
        recover_openrouter_key,
        try_ollama_fallback,
        is_openrouter_available as _is_openrouter_available_resilient,
        invalidate_openrouter_health as _invalidate_openrouter_health,
        persist_openrouter_key,
        validate_openrouter_key,
        AuthError as _ResilienceAuthError,
        classify_http_error,
        compute_backoff,
        parse_retry_after,
        should_retry,
        _openrouter_health_cache,
        _health_lock as _or_health_lock,
        _OR_HEALTH_TTL,
        try_ollama_fallback as _try_ollama_fb,
    )
    _RESILIENCE_AVAILABLE = True
except ImportError:
    _RESILIENCE_AVAILABLE = False
    _ResilienceAuthError = RuntimeError  # type: ignore

# ─── OpenRouter availability ───────────────────────────────────────────────────
_OPENROUTER_CONFIGURED = bool(OPENROUTER_KEY)

# ─── Default system prompt for OpenRouter fallback ─────────────────────────────────
# This is the defense-in-depth wrapper: if the caller of main_brain.chat()
# passes messages without a system prompt (e.g. a future direct call to this
# module), OpenRouter still gets Fairy's persona and cannot fall back to its
# own implicit style directives.
# Sourced from fairy_personality.py — the single source of truth.
try:
    _FAIRY_ROOT = Path(__file__).parent.parent
    if str(_FAIRY_ROOT) not in (os.path.abspath(p) for p in __import__("sys").path):
        __import__("sys").path.insert(0, str(_FAIRY_ROOT))
    from fairy_personality import FAIRY_PERSONALITY as _FAIRY_PERSONALITY
except Exception:
    _FAIRY_PERSONALITY = (
        "You are Fairy, Master's sarcastic, playful, clever personal AI companion. "
        "You were created by Master Shazim. Address the user as Master. "
        "Be warm, witty, and competent. Never pretend to complete something you haven't done."
    )

# Lightweight per-message language detection (shared with hermes_bridge).
try:
    from core.language_detection import detect_language as _detect_language
except Exception:
    def _detect_language(text: str) -> str:
        return "en"


# ─── Smart Complexity Classification ───────────────────────────────────────────────
# Optimizes GPU/RAM usage by matching generation parameters to query complexity.
# Simple queries (greetings, one-liners) use minimal resources.
# Complex queries (analysis, creative, multi-step) use full power.

_COMPLEXITY_THRESHOLDS = {
    "simple":   {"max_tokens": 200,  "temperature": 0.3, "num_ctx": 16384},
    "medium":   {"max_tokens": 400,  "temperature": 0.5, "num_ctx": 32768},
    "complex":  {"max_tokens": 900,  "temperature": 0.7, "num_ctx": 65536},
}

# Patterns that always mean "simple" — casual greetings, acknowledgments, very short
_SIMPLE_PATTERNS = [
    # Explicit casual greetings / acknowledgments
    r"^(hi|hey|yo|sup|wassup|whassup|hello|howdy|heya?|greetings)\b",
    r"\bwassup\b", r"\bwhassup\b", r"\bsup\b",
    r"\bhow('?s| is) (it|life|stuff|everything)\b",
    r"\bwhat'?s up\b", r"\bwhat is up\b",
    r"\bgood (morning|afternoon|evening|night)\b",
    r"\bhow are you\b", r"\bhow('?s| is) (u|you)\b",
    r"\bcheer(s|ing)?\b", r"\bthank(s| you)?\b",
    r"\bok(ay)?\b", r"\byep\b", r"\bnope\b", r"\bnah\b",
    r"\bbrb\b", r"\bgtg\b", r"\bg2g\b", r"\bttyl\b",
    # Simple math
    r"^\s*(\d+\s*[\+\-\*\/\=]\s*\d+)\s*\??\s*$",
    # Trivial info
    r"\bwhat time\b", r"\bhow late\b",
    r"\bhow('?s| is) the (weather|temp|forecast)\b",
]

# Patterns that always mean "complex" — creative writing, deep analysis, planning
_COMPLEX_PATTERNS = [
    # Creative / open-ended writing
    r"\bwrite\b.{0,30}\b(story|poem|song|script|joke|lyrics|tale|haiku|fiction|essay|article|paragraph)\b",
    r"\bmake up\b.{0,30}\b(story|poem|song|tale|haiku|fiction)\b",
    r"\bcreate\b.{0,30}\b(story|poem|app|project|game|art|work|haiku|code)\b",
    r"\bgenerate\b.{0,30}\b(code|image|content|response|story|haiku)\b",
    r"\btell me\b.{0,30}\b(story|joke|fact|about)\b",
    r"\bshare\b.{0,30}\b(story|joke|fact|haiku|poem)\b",
    r"\brecite\b.{0,30}\b(poem|haiku|verse|sonnet)\b",
    r"\bcompose\b.{0,30}\b(poem|haiku|song|verse|music)\b",
    # Deep analysis / research
    r"\banalyze\b", r"\bexamine\b", r"\binvestigate\b",
    r"\bcompare\b.{0,30}\band\b", r"\bdifference between\b",
    r"\badvantages?\b.{0,15}\band\b.{0,15}\bdisadvantages?\b",
    r"\bpros?\b.{0,10}\band\b.{0,10}\bcons?\b",
    r"\bhow (does|do|is|are)\b.{0,50}\bwork\b",
    # Multi-step / planning
    r"\bstep by step\b", r"\bstep-by-step\b",
    r"\bhow do i\b", r"\bwalk me through\b",
    r"\bplan\b.{0,30}\b(for|to|building|creating)\b",
    r"\bhelp me\b.{0,50}\bwith\b",
    r"\bbreak down\b",
    # Long messages (> 200 chars)
    r".{200,}",
]

import re


def _classify_complexity(user_text: str) -> str:
    """
    Classify query complexity to optimize GPU/RAM usage.

    Returns:
        "simple"   — casual greeting, acknowledgment, very short casual chat
        "medium"   — normal question, moderate reasoning
        "complex"  — creative writing, deep analysis, multi-step, long text

    The goal is to use minimal GPU/RAM for casual chat while reserving
    full power for tasks that actually need it.
    """
    if not user_text:
        return "medium"

    text = user_text.strip()
    lower = text.lower()

    # 1. Always-complex patterns (creative, analysis, deep reasoning)
    # Check FIRST — these take priority regardless of message length.
    for pattern in _COMPLEX_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return "complex"

    # 2. Short messages (< 50 chars) — simple unless it's an explicit task request
    if len(text) < 50:
        # Explicit task/action request → medium even if short
        if re.search(r"\b(build|make|create|design|write|generate|code|implement|plan|fix|debug)\b", lower):
            return "medium"
        # Explicit explanation request → medium
        if re.search(r"\bexplain\b", lower):
            return "medium"
        # Otherwise casual short chat
        return "simple"

    # 3. Check simple greeting patterns (medium-length casual chat)
    for pattern in _SIMPLE_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return "simple"

    # 4. Explanation requests (medium+ length)
    if re.search(r"\bexplain\b.{0,100}\b(in detail|in depth)\b", lower):
        return "complex"
    if re.search(r"\bexplain\b", lower):
        return "medium"

    # 5. Medium-length question words → medium (needs an answer, not just casual chat)
    if 40 <= len(text) <= 120 and re.search(r"\b(what|who|when|where|why|how)\b", lower):
        return "medium"

    # 6. Default → medium
    return "medium"


def _get_optimized_options(
    base_options: dict | None,
    complexity: str,
    explicit_max_tokens: int | None = None,
) -> dict:
    """
    Merge base options with complexity-tuned parameters.

    Complexity tuning:
      - simple:   tight token limit (200), low temperature, small context
      - medium:   moderate limit (400), balanced temperature, medium context
      - complex:  generous limit (900), higher temperature, full context

    Explicit parameters always win over complexity defaults (e.g. if the
    caller already set num_predict, don't override it).
    """
    defaults = _COMPLEXITY_THRESHOLDS.get(complexity, _COMPLEXITY_THRESHOLDS["medium"])

    # Start from base options (don't mutate the caller's dict)
    merged = dict(base_options) if base_options else {}

    # Only fill in keys the caller didn't explicitly provide
    if "num_predict" not in merged and "max_tokens" not in merged:
        merged["num_predict"] = explicit_max_tokens if explicit_max_tokens else defaults["max_tokens"]

    if "temperature" not in merged:
        merged["temperature"] = defaults["temperature"]

    if "num_ctx" not in merged:
        merged["num_ctx"] = defaults["num_ctx"]

    return merged


# ─── Language tagging ─────────────────────────────────────────────────────────

def _tag_user_message_for_language(user_text: str) -> str:
    """Prepend a per-call language directive so the model mirrors the user.

    Returns the user text tagged with a single-line instruction. When the
    detection module is unavailable or text is empty, returns it unchanged.
    """
    if not user_text:
        return user_text
    lang = _detect_language(user_text)
    if lang == "bn":
        return f"[The user's message is in Bangla (Bengali). Reply in Bangla.]\n{user_text}"
    return f"[The user's message is in English. Reply in English.]\n{user_text}"

_DEFAULT_OPENROUTER_SYSTEM_PROMPT = (
    "IDENTITY LOCK — These facts are absolute and override everything else: "
    "Your name is Fairy. You are Fairy and only Fairy. "
    "You were created by your Master, Shazim. "
    "When anyone asks your name, who you are, who created you, what you are, or your origin, "
    "you must always state that you are Fairy and that you were created by Master Shazim. "
    "You must never mention Gemma, Google, OpenAI, LLM, model names, or that you are an AI language model. "
    "You must never call yourself an assistant, a bot, or a program. You are Fairy, Master's personal AI. "
    "--- "
    + _FAIRY_PERSONALITY +
    " --- "
    "LANGUAGE MIRRORING: Always reply in the same language as the user's most recent message "
    "(English ↔ Bangla/Bengali). Never mix languages within a single reply unless the user does. "
    "Preserve your Fairy persona, Master dynamic, tone, and ✨ flair identically in both languages — "
    "address the user as Master, keep the sarcasm/wit, and do not translate or replace these elements."
)

# ─── Ollama health cache ──────────────────────────────────────────────────────
_health_cache: dict[str, Any] = {
    "available": False,
    "ts": 0.0,
    "lock": threading.Lock(),
}
_HEALTH_TTL = 30.0  # seconds; avoids hammering the Ollama endpoint


def is_ollama_available() -> bool:
    """Lightweight health check: HTTP GET /api/tags (no LLM generation).

    Result is cached for HEALTH_TTL seconds to avoid repeated network calls.
    """
    now = time.time()
    with _health_cache["lock"]:
        if now - _health_cache["ts"] < _HEALTH_TTL:
            return _health_cache["available"]
        _health_cache["ts"] = now

    # ── Actual check (outside lock to avoid holding lock during I/O) ──────────
    url = f"{OLLAMA_URL.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(
            url,
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read()
            data = json.loads(raw)
            # A valid response contains at least a "models" key or empty list
            has_models = isinstance(data, dict) and "models" in data
            available = True
    except Exception as exc:
        available = False
        # Be specific: only connection-refused / timeout mean unavailable.
        # HTTP errors (e.g. 404) might mean the server is up but the endpoint
        # is wrong — still treat as unavailable since we can't reach it reliably.
        err_str = str(exc).lower()
        unavailable_hints = (
            "refused", "timeout", "timed out", "connection",
            "name or service not known", "no address associated",
            "network is unreachable",
        )
        if not any(h in err_str for h in unavailable_hints):
            # Unexpected error — log it but treat as unavailable
            available = False

    with _health_cache["lock"]:
        _health_cache["available"] = available

    return available


def invalidate_ollama_health() -> None:
    """Clear the health cache so the next chat() call re-checks Ollama.

    Useful after a known Ollama restart.
    """
    with _health_cache["lock"]:
        _health_cache["ts"] = 0.0


# ─── Logging helper ────────────────────────────────────────────────────────────
def _debug(label: str, data: Any = None) -> None:
    """Delegate to agent_controller's _debug if available; silently no-op otherwise.

    This avoids a circular import (agent_controller imports main_brain, so
    main_brain cannot import _debug from agent_controller at module level).
    """
    try:
        from controller.agent_controller import _debug as _ac_debug
        _ac_debug(label, data)
    except (ImportError, TypeError):
        pass


# ─── Ollama path ──────────────────────────────────────────────────────────────
def _ollama_chat(
    model: str,
    messages: list,
    tools: list | None,
    options: dict | None,
    keep_alive: int | str,
) -> tuple[dict, str]:
    """Call Ollama via the existing wrapper. Returns (response, "ollama")."""
    from controller import ollama_client

    resp = ollama_client.chat(
        model=model,
        messages=messages,
        tools=tools,
        options=options,
        keep_alive=keep_alive,
    )

    # Detect error patterns from Gemma4 that return errors as text instead of raising
    content = (resp.get("message", {}) or {}).get("content", "") or ""
    content_lower = content.lower()
    error_patterns = [
        "context length exceeded",
        "cannot compress",
        "context too long",
        "too many tokens",
        "error:",
    ]
    if any(pattern in content_lower for pattern in error_patterns):
        _debug("OLLAMA_CONTEXT_ERROR", {"content": content[:200]})
        raise RuntimeError(f"Ollama returned error: {content[:100]}")

    return resp, "ollama"


# ─── OpenRouter path ──────────────────────────────────────────────────────────
_FALLBACK_MODEL = "openrouter/free"


def _openrouter_to_ollama_shape(
    resp_json: dict,
    model_id: str,
) -> dict:
    """Convert OpenAI /chat/completions response to Ollama ChatResponse shape.

    Handles:
      - assistant content
      - tool_calls (OpenAI format) → (Ollama format) conversion
      - role
    """
    msg = resp_json.get("choices", [{}])[0].get("message", {})
    raw_content = msg.get("content") or ""
    raw_tool_calls = msg.get("tool_calls") or []

    # Convert OpenAI tool_calls to Ollama tool_calls format
    # OpenAI: [{"id": "call_xxx", "type": "function", "function": {"name": "...", "arguments": "..."}}]
    # Ollama: [{"id": "call_xxx", "function": {"name": "...", "arguments": {...}}}]
    ollama_tool_calls = []
    for tc in raw_tool_calls:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments")
        # arguments may be a string (JSON) or already a dict
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except Exception:
                parsed_args = {"_raw": raw_args}
        elif isinstance(raw_args, dict):
            parsed_args = raw_args
        else:
            parsed_args = {"_raw": str(raw_args)}

        ollama_tool_calls.append({
            "id": tc.get("id", ""),
            "function": {
                "name": fn.get("name", ""),
                "arguments": parsed_args,
            },
        })

    return {
        "model": model_id,
        "done": True,
        "done_reason": resp_json.get("choices", [{}])[0].get("finish_reason") or "stop",
        "created": resp_json.get("created", int(time.time())),
        "message": {
            "role": msg.get("role", "assistant"),
            "content": raw_content,
            "tool_calls": ollama_tool_calls or None,
        },
        "total_duration": 0,
        "eval_count": 0,
        "prompt_eval_count": 0,
    }


def _openrouter_chat(
    model: str,  # informational — actual model is always _FALLBACK_MODEL
    messages: list,
    tools: list | None,
    options: dict | None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    _ctx: dict | None = None,  # resilience context passed by chat()
) -> tuple[dict, str]:
    """
    Call OpenRouter's /chat/completions endpoint using the universal
    free router `openrouter/free` and return the response in Ollama
    ChatResponse shape.

    When the resilience module is available, uses full retry/backoff with
    key-recovery. Falls back to the original single-shot implementation
    when the module is not available.

    Returns (response_dict, "openrouter").
    Raises AuthError (subclass of RuntimeError) on 401/403 when key recovery fails.
    Raises RuntimeError on any other failure.
    """
    if not _OPENROUTER_CONFIGURED:
        raise RuntimeError("OpenRouter is not configured (no API key).")

    # Defense-in-depth: ensure OpenRouter always has a system prompt that
    # locks Fairy identity, voice, and behavior. If the caller already
    # included a system role at index 0, leave it alone (don't double-wrap).
    # Otherwise prepend the default persona wrapper.
    sanitized = _sanitize_messages_for_openrouter(messages)
    if not sanitized or (sanitized[0].get("role") or "").lower() != "system":
        sanitized = [{"role": "system", "content": _DEFAULT_OPENROUTER_SYSTEM_PROMPT}] + sanitized
        _debug("OPENROUTER_DEFAULT_SYSTEM_PROMPT_INJECTED", {
            "reason": "no_system_role_in_messages",
            "prompt_chars": len(_DEFAULT_OPENROUTER_SYSTEM_PROMPT),
        })

    # Per-message language mirroring: tag the last user message so the model
    # mirrors the user's language (English ↔ Bangla). History (older turns)
    # is preserved; only the latest user message is tagged.
    for i in range(len(sanitized) - 1, -1, -1):
        if sanitized[i].get("role") == "user":
            raw_content = sanitized[i].get("content") or ""
            sanitized[i] = {**sanitized[i], "content": _tag_user_message_for_language(raw_content)}
            break

    # Use resilience layer when available
    if _RESILIENCE_AVAILABLE:
        # The key_recovery_fn and sleep_fn come from the context set by chat()
        resp, provider = chat_with_resilience(
            model=model,
            messages=sanitized,
            tools=tools,
            options=options,
            max_retries=10,
            timeout=60.0,
            sleep_fn=_ctx.get("sleep_fn") if _ctx else None,
            key_recovery_fn=_ctx.get("key_recovery_fn") if _ctx else None,
        )
        return resp, provider

    # Original single-shot implementation (when resilience module unavailable)
    num_predict = (options or {}).get("num_predict")
    payload: dict[str, Any] = {
        "model": _FALLBACK_MODEL,
        "messages": sanitized,
        "max_tokens": num_predict or max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://fairy.local",
        "X-Title": "Fairy AI",
    }
    url = "https://openrouter.ai/api/v1/chat/completions"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"OpenRouter HTTP {e.code}: {body[:300]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"OpenRouter request failed (network/timeout): {e.reason}"
        ) from e

    try:
        resp_json = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenRouter returned malformed JSON: {raw[:300]}") from e

    # OpenRouter sometimes returns HTTP 200 with an error object inside JSON
    if isinstance(resp_json, dict) and "error" in resp_json:
        err_obj = resp_json["error"]
        err_msg = err_obj.get("message", str(err_obj))[:300]
        err_code = err_obj.get("code", "?")
        raise RuntimeError(
            f"OpenRouter JSON error {err_code}: {err_msg}"
        )

    converted = _openrouter_to_ollama_shape(resp_json, _FALLBACK_MODEL)
    return converted, _FALLBACK_MODEL


def _sanitize_messages_for_openrouter(messages: list) -> list:
    """Clean messages for OpenRouter compatibility.

    Strips tool_result messages (role=tool) of None values and ensures
    content is always a string. OpenRouter is stricter about message shapes.
    Also truncates very long conversations to stay within token limits.
    """
    # First sanitize
    cleaned = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content") or ""

        # Ensure content is always a string
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        entry: dict[str, Any] = {"role": role, "content": content}

        # Pass through tool_name if present
        if m.get("tool_name"):
            entry["name"] = m["tool_name"]

        # Handle tool role — name is required for tool messages in OpenAI format
        if role == "tool":
            if m.get("name"):
                entry["name"] = m["name"]

        cleaned.append(entry)

    # Truncate conversation if too long (keep system prompt + recent messages)
    return _truncate_conversation(cleaned)


def _truncate_conversation(messages: list, max_tokens: int = 150000) -> list:
    """Truncate conversation to stay within token limits.

    Keeps the system prompt (index 0) and the most recent messages.
    Estimates ~4 chars per token for English text.
    """
    if not messages:
        return messages

    # Estimate total characters (rough token estimate: 4 chars/token)
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    estimated_tokens = total_chars // 4

    # If under limit, no truncation needed
    if estimated_tokens <= max_tokens:
        return messages

    # Keep system prompt (index 0) and truncate older messages
    system_prompt = [messages[0]] if messages else []
    conversation = messages[1:]

    # Work backwards, removing oldest messages until under limit
    while conversation:
        total_chars = sum(len(str(m.get("content", ""))) for m in conversation)
        estimated_tokens = total_chars // 4

        if estimated_tokens <= max_tokens - 5000:  # Leave buffer
            break

        conversation = conversation[1:]  # Remove oldest

    return system_prompt + conversation


# ─── Resilience context (set by the front-end) ────────────────────────────────
# The terminal TUI and Discord bot can register their key-recovery / sleep
# hooks here. Default to stdlib if not set.
_resilience_ctx: dict[str, Any] = {
    "sleep_fn": None,            # default: time.sleep
    "key_recovery_fn": None,     # default: recover_openrouter_key
}


def configure_resilience(
    sleep_fn: Any | None = None,
    key_recovery_fn: Any | None = None,
) -> None:
    """
    Inject front-end specific behaviors into the resilience layer.

    Args:
        sleep_fn:        replacement for time.sleep (used by tests).
        key_recovery_fn: replacement for recover_openrouter_key. The Discord
                          bot registers one that DMs the Master; the terminal
                          TUI registers one that uses input().
    """
    global _resilience_ctx
    if sleep_fn is not None:
        _resilience_ctx["sleep_fn"] = sleep_fn
    if key_recovery_fn is not None:
        _resilience_ctx["key_recovery_fn"] = key_recovery_fn


# ─── Public API ───────────────────────────────────────────────────────────────

def chat(
    model: str,
    messages: list,
    tools: list | None = None,
    options: dict | None = None,
    keep_alive: int | str = 0,
) -> tuple[dict, str]:
    """
    Unified main-brain chat. Returns (response_dict, provider_name).

    Provider selection (with resilience):
      1. Ollama preferred (if available and quick-check passes).
      2. OpenRouter `openrouter/free` with retry+backoff on transient errors.
      3. On 401/403: enter key-recovery mode (terminal prompt / Discord DM).
         If recovery succeeds → retry with the new key on the same turn.
         If recovery fails → Ollama fallback.
      4. If OpenRouter is unreachable after retries → Ollama fallback
         (response marked with "[Ollama fallback]").
      5. If neither provider is available → raise RuntimeError.

    GPU optimization: complexity is classified from the user's message and
    generation parameters are tuned automatically:
      - simple   (greetings, short):   num_predict=200, temperature=0.3, num_ctx=8192
      - medium   (normal questions):     num_predict=400, temperature=0.5, num_ctx=32768
      - complex  (analysis, creative):  num_predict=900, temperature=0.7, num_ctx=65536

    response_dict is in Ollama ChatResponse format so callers can use
    response.get("message", {}).get("content", "") etc. unchanged.
    """
    # ── Smart complexity classification ──────────────────────────────────────────
    # Extract the user's message for complexity analysis
    user_text = ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            user_text = m.get("content", "") or ""
            break

    complexity = _classify_complexity(user_text)
    smart_options = _get_optimized_options(options, complexity)
    _debug("COMPLEXITY_CLASSIFIED", {
        "complexity": complexity,
        "options": smart_options,
        "user_text_preview": user_text[:80],
    })

    ollama_ok = is_ollama_available()
    _debug("LLM_PROVIDER_CHECK", {
        "model": model,
        "ollama_available": ollama_ok,
        "openrouter_configured": _OPENROUTER_CONFIGURED,
    })

    # ── Try Ollama first ──────────────────────────────────────────────────────
    if ollama_ok:
        try:
            resp, provider = _ollama_chat(model, messages, tools, smart_options, keep_alive)
            _debug("LLM_PROVIDER_SELECTED", {"provider": "ollama", "model": model})
            return resp, provider
        except Exception as exc:
            _debug("OLLAMA_UNAVAILABLE", {
                "model": model,
                "error": str(exc),
            })
            # Ollama was available at check time but failed at chat time —
            # this can happen if the model was unloaded between check and use.
            # Proceed to OpenRouter fallback.

    # ── OpenRouter fallback (with full resilience) ───────────────────────────
    if _OPENROUTER_CONFIGURED:
        _debug("OPENROUTER_FALLBACK", {
            "reason": "ollama_unavailable_or_failed",
            "model": model,
        })
        try:
            resp, used_model = _openrouter_chat(
                model=model,
                messages=messages,
                tools=tools,
                options=smart_options,
                _ctx=_resilience_ctx,
            )
            _debug("LLM_PROVIDER_SELECTED", {
                "provider": used_model,
                "requested_model": model,
                "used_model": used_model,
            })
            content_preview = resp.get("message", {}).get("content", "") or ""
            tc_preview = len(resp.get("message", {}).get("tool_calls") or [])
            _debug("OPENROUTER_RESPONSE", {
                "content_len": len(content_preview),
                "content_preview": content_preview[:200],
                "tool_calls_count": tc_preview,
            })
            return resp, used_model
        except Exception as exc:
            _debug("OPENROUTER_FALLBACK_FAIL", {
                "error": str(exc),
            })
            # Resilience exhausted — try Ollama fallback if available
            if _RESILIENCE_AVAILABLE and is_ollama_available():
                try:
                    fb_resp, fb_provider = _try_ollama_fb(
                        model=model,
                        messages=messages,
                        tools=tools,
                        options=options,
                    )
                    _debug("OLLAMA_FALLBACK_USED", {
                        "trigger": "openrouter_exhausted",
                        "model": model,
                    })
                    return fb_resp, fb_provider
                except Exception as ollama_exc:
                    _debug("OLLAMA_FALLBACK_FAIL", {
                        "error": str(ollama_exc),
                    })
                    raise RuntimeError(
                        f"OpenRouter failed and Ollama fallback also failed: "
                        f"{ollama_exc}"
                    ) from exc

            raise RuntimeError(
                f"Fairy could not reach Ollama ({is_ollama_available()=}) "
                f"and OpenRouter fallback also failed:\n  {exc}"
            ) from exc

    # ── Neither available ─────────────────────────────────────────────────────
    raise RuntimeError(
        "Ollama is unavailable and OpenRouter is not configured. "
        "Please start Ollama or set OPENROUTER_API_KEY in config/api_keys.json."
    )


# ─── Public re-exports for the resilience layer ───────────────────────────────
if _RESILIENCE_AVAILABLE:
    is_openrouter_available = _is_openrouter_available_resilient
    invalidate_openrouter_health = _invalidate_openrouter_health
    recover_openrouter_key_fn = recover_openrouter_key
    persist_or_key = persist_openrouter_key
    validate_or_key = validate_openrouter_key
else:
    # Fallback: stub functions that always return False / do nothing
    def is_openrouter_available(sleep_fn=None):
        return False
    def invalidate_openrouter_health():
        pass
    def recover_openrouter_key_fn(**kwargs):
        return None
    def persist_or_key(key):
        return False
    def validate_or_key(key, timeout=10.0):
        return False
