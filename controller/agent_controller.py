#!/usr/bin/env python3
# The boss. Gemma (brain) never runs code directly — it calls tools.
# This module executes them, and create_skill hands off to Qwen + sandbox.
#
# SPEED FIX: _fast_intent() catches simple chat/time queries before
# any LLM call, cutting common requests down to a tiny fast path.
#
# FAIRY 2.0 — Inspired by Mark-L (FatihMakes):
#   - Dedicated prompt engineering (core/prompts.py)
#   - Structured planner for action requests (fixes fake completions)
#   - Persistent memory (memory/memory_manager.py)
#   - Stricter adaptive loop with tool budgets
#
# LAZY ROUTING: Added deterministic unit conversion, single-LLM response
# for low-complexity requests, and complexity-based execution strategy.
#
# OPENROUTER INTEGRATION: Added search_and_summarize tool that offloads
# information retrieval and summarization to free OpenRouter models,
# reducing local GPU load for factual queries.

import json
import sys
import os
import re
import ast
import inspect
import webbrowser
import threading
import time
import traceback
from datetime import datetime
import secrets
from pathlib import Path
from typing import Any, Callable, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller import ollama_client, skill_manager, quips
from sandbox.sandbox_runner import test_skill_code
from config import MODEL_BRAIN, MODEL_CODER, MAX_FIX_ATTEMPTS

from skills.web_search import web_search
from skills.web_fetch import web_fetch
from skills.deep_research import deep_research
from skills.maps import get_location, get_directions, search_nearby
from skills.get_time import get_current_time
from skills.system_monitor import system_monitor, get_system_summary, get_gpu_info, SystemMonitor, speak_alert
from skills.reminders import reminder_tool, send_notification, set_reminder
from skills.computer_control import computer_control, set_volume, get_volume, take_screenshot
from skills.screen_processor import (
    screen_process, warmup_session,
    start_session, stop_session, is_session_ready,
)
from skills.send_message import send_message
from skills.vision_skill import (
    vision_analyze, vision_describe,
    is_vision_available,
)


# Fairy 2.0 imports
from core.prompts import build_system_prompt, build_action_decision_prompt, build_synthesis_prompt
from core.language_detection import detect_language as _detect_language
from memory.memory_manager import get_memory
from memory.long_term_memory import (
    get_long_term_memory,
    extract_proposals,
    INJECTION_BUDGET_MAX_CHARS,
    INJECTION_BUDGET_MAX_FACTS,
    reset_long_term_memory,
)
from controller.claude_code_delegate import (
    detect_repository_task,
    request_permission,
    find_claude_binary,
    build_handoff_command,
    is_handoff_safe_command,
    handoff_diagnostics,
    _classify_task_directory as _claude_classify_task_directory,
    _EXPLICIT_AGENT_PATTERNS,
)
from controller.delegate_state import (
    is_approval,
    is_denial,
    is_clarification,
    set_debug_hook as _set_delegate_debug_hook,
)
from controller import approval_gate, verification_evidence  # unified approval gate for file-mutating tools
from controller.model_selector import is_gemma4_available, is_ollama_running

# Wire approval_gate.set_discord_active so it follows whatever the
# discord_bot module decides. The bot sets the flag on its own
# _request_ctx; we just give the gate a way to ask it. Module-level
# shim so callers don't have to import discord_bot directly.
_request_ctx_for_gate_set = False
_request_ctx_for_gate = None

# The default is terminal-mode. controller/discord_bot.py explicitly
# calls ``approval_gate.set_discord_active(True)`` after patching
# _dispatch_tool so the gate knows to raise ApprovalRequired instead
# of blocking on a terminal prompt. We do NOT auto-detect here
# because the import graph (main.py → agent_controller vs.
# discord_bot → agent_controller) doesn't tell us reliably which
# front-end started first.

ask_openai: Callable[..., Any] | None = None
ask_claude: Callable[..., Any] | None = None
ask_gemini: Callable[..., Any] | None = None
ask_grok: Callable[..., Any] | None = None
ask_meta: Callable[..., Any] | None = None
ask_deepseek: Callable[..., Any] | None = None
ask_kimi: Callable[..., Any] | None = None
navigate_to: Callable[..., Any] | None = None
search_on_site: Callable[..., Any] | None = None
add_to_cart_amazon: Callable[..., Any] | None = None
upload_instagram_reel: Callable[..., Any] | None = None
click_element: Callable[..., Any] | None = None
type_text: Callable[..., Any] | None = None
get_page_info: Callable[..., Any] | None = None
close_browser: Callable[..., Any] | None = None
browser_control: Callable[..., Any] | None = None
ask_openrouter: Callable[..., Any] | None = None
ask_or_coder: Callable[..., Any] | None = None
ask_or_smart: Callable[..., Any] | None = None
ask_or_cheap: Callable[..., Any] | None = None
ask_or_chat: Callable[..., Any] | None = None
ask_or_research: Callable[..., Any] | None = None
MODEL_REGISTRY: dict[str, Any] = {}

try:
    from skills.llm_apis import (
        ask_openai, ask_claude, ask_gemini,
        ask_grok, ask_meta, ask_deepseek, ask_kimi,
    )
    _LLM_APIS_AVAILABLE = True
except ImportError:
    _LLM_APIS_AVAILABLE = False

try:
    from skills.browser_automation import (
        navigate_to, search_on_site, add_to_cart_amazon,
        upload_instagram_reel, click_element, type_text,
        get_page_info, close_browser, browser_control
    )
    _BROWSER_AVAILABLE = True
except ImportError:
    _BROWSER_AVAILABLE = False

try:
    from skills.openrouter import (
        ask_openrouter, ask_or_coder, ask_or_smart,
        ask_or_cheap, ask_or_chat, ask_or_research,
        MODEL_REGISTRY,
    )
    _OPENROUTER_AVAILABLE = True
except ImportError:
    _OPENROUTER_AVAILABLE = False

try:
    import subprocess as _subprocess_probe
    _subprocess_probe.run(["echo", "ok"], capture_output=True, timeout=5)
    _TOOL_EXECUTOR_AVAILABLE = True
except Exception:
    _TOOL_EXECUTOR_AVAILABLE = False

# Main-brain provider abstraction: Ollama preferred, OpenRouter (free) fallback.
# Import at module level so all LLM call sites use the same fallback logic.
try:
    from controller.main_brain import chat as _brain_chat
    from controller.main_brain import is_ollama_available as _is_ollama_available
    from controller.main_brain import invalidate_ollama_health as _invalidate_ollama_health
    _BRAIN_PROVIDER_AVAILABLE = True
except ImportError:
    _BRAIN_PROVIDER_AVAILABLE = False
    _is_ollama_available = lambda: False  # noqa: E731
    _invalidate_ollama_health = lambda: None  # noqa: E731


def _brain(model: str, messages: list, tools: list | None = None,
           options: dict | None = None, keep_alive: int | str = 0) -> dict:
    """Unified main-brain call (Ollama preferred, OpenRouter FREE-only fallback).

    Returns the response dict in Ollama ChatResponse shape so all existing
    call sites (response.get("message", {}).get("content", ""), tool_calls,
    done_reason, etc.) continue to work unchanged.

    Raises on total failure so the caller can choose how to recover.
    """
    if not _BRAIN_PROVIDER_AVAILABLE:
        # Module not importable (broken install) — fall back to raw Ollama so
        # the system stays alive.
        from controller import ollama_client
        return ollama_client.chat(
            model=model, messages=messages, tools=tools,
            options=options, keep_alive=keep_alive,
        )
    resp, _provider = _brain_chat(
        model=model, messages=messages, tools=tools,
        options=options, keep_alive=keep_alive,
    )
    return resp


# ═══════════════════════════════════════════════════════════════════
# DEBUG TRACING — set FAIRY_DEBUG=1 to see exactly what breaks
# ═══════════════════════════════════════════════════════════════════

_DEBUG = os.environ.get("FAIRY_DEBUG", "0").strip() in ("1", "true", "yes", "on")
_DEBUG_FILE = os.environ.get("FAIRY_DEBUG_FILE", os.path.join(os.path.dirname(__file__), "..", "fairy_debug.log"))

def _debug(label, data=None):
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
            line = f"[{ts}] {label} | {payload}"
        # Append in binary mode to avoid platform newline translation that can
        # mangle JSON on Windows; flush so a tail -f sees it immediately.
        with open(_DEBUG_FILE, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(line + "\n")
            fh.flush()
    except Exception as exc:
        # Last-resort: print to stderr so we can see if _debug itself broke.
        try:
            import sys as _sys
            print(f"[Fairy][_debug-fail] {label}: {exc}", file=_sys.stderr)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Per-turn tool-dispatch tracking — FIX 1
#
# Tracks whether any tool was dispatched during the current handle_request
# turn. The task-action guard in handle_request reads this and re-routes
# the request through the planner if the brain answered "I did X" but
# never actually dispatched a tool to do X.
# ═══════════════════════════════════════════════════════════════════
_TOOLS_DISPATCHED_THIS_TURN = {"count": 0, "names": []}


def _reset_turn_dispatch_counter():
    """Call at the start of every handle_request turn."""
    _TOOLS_DISPATCHED_THIS_TURN["count"] = 0
    _TOOLS_DISPATCHED_THIS_TURN["names"] = []


def _mark_tool_dispatched(fn: str) -> None:
    """Call from every code path that actually dispatches a tool."""
    _TOOLS_DISPATCHED_THIS_TURN["count"] += 1
    _TOOLS_DISPATCHED_THIS_TURN["names"].append(fn)


# ═══════════════════════════════════════════════════════════════════
# Vision state — tracks whether the persistent vision session is "on"
# from the user's perspective. The session itself is a long-lived Gemini
# live connection (see skills/screen_processor.py); this dict is just
# Fairy's perspective on it.
#
# State machine:
#   "off"   → user hasn't asked for vision yet (or just turned it off)
#   "on"    → session is connected and will answer follow-up questions
#   "starting" → user just asked to turn vision on, session spinning up
#
# The vision_on / vision_off fast-intent handlers in handle_request()
# drive this state directly (bypassing the LLM), so "close your vision"
# always works, never gets hijacked by Hermes.
# ═══════════════════════════════════════════════════════════════════
_VISION_STATE = {
    "mode": "off",       # "off" | "on" | "starting"
    "angle": "screen",   # last requested angle
    "last_description": "",   # most recent description from Gemini
    "last_description_ts": 0.0,
}


def _vision_state() -> dict:
    return _VISION_STATE


def _set_vision_state(mode: str, angle: str | None = None, description: str | None = None) -> None:
    _VISION_STATE["mode"] = mode
    if angle is not None:
        _VISION_STATE["angle"] = angle
    if description is not None:
        _VISION_STATE["last_description"] = description
        _VISION_STATE["last_description_ts"] = time.time()


def _was_tool_dispatched() -> tuple[int, list[str]]:
    """Returns (count, list_of_tool_names) for the current turn."""
    return _TOOLS_DISPATCHED_THIS_TURN["count"], list(_TOOLS_DISPATCHED_THIS_TURN["names"])


# Wire the delegate_state debug hook into our _debug logger.
# Must run AFTER _debug is defined. Module-level call so any is_approval()
# or related call after import will route debug events to fairy_debug.log.
_set_delegate_debug_hook(_debug)


# ─────────────────────────────────────────────────────────────────
# Pending handoff signal (v2 — replaces the delegation pipeline)
#
# When handle_request() detects a handoff trigger, it sets this variable
# to the handoff dict and returns (None, updated_history). The TUI
# (fairy.py) checks it after the call and invokes _suspend_and_handoff().
# After the handoff completes, the TUI clears it.
#
# Thread-safety note: Fairy's TUI is single-threaded. This variable is
# only ever written by handle_request() and read by the TUI after the
# call returns — no race condition in the normal call/response flow.
# ─────────────────────────────────────────────────────────────────
_pending_handoff: dict | None = None


def get_pending_handoff() -> dict | None:
    """Return and clear the pending handoff signal, if any."""
    global _pending_handoff
    h = _pending_handoff
    _pending_handoff = None
    return h


# ═══════════════════════════════════════════════════════════════════
# No-op shims (v2 — delegation pipeline removed)
#
# These are kept for import compatibility. The delegation pipeline was removed
# in the v1→v2 redesign; these functions are now no-ops.
# ─────────────────────────────────────────────────────────────────


def clear_hermes_gate() -> None:
    """No-op. The Hermes fallback gate was removed in v2."""


# ═══════════════════════════════════════════════════════════════════
# Chat-turn logger — FIX 3
#
# Unlike _debug(), this writes REGARDLESS of FAIRY_DEBUG so the chat
# path is always observable in fairy_debug.log. The brain's verbose
# trace stays gated by FAIRY_DEBUG; the chat-turn lifecycle stays on.
# ═══════════════════════════════════════════════════════════════════
def _log_chat_turn(user_text: str, phase: str, **fields) -> None:
    """Append a structured CHAT_TURN line to the debug log.

    Args:
        user_text: The user message (truncated to 300 chars in the log).
        phase: Short tag like "start", "end", "discord_start", "discord_end".
        **fields: Extra structured fields to include in the JSON payload.
    """
    try:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        payload = {
            "phase": phase,
            "user_text": (user_text or "")[:300],
            **fields,
        }
        line = f"[{ts}] CHAT_TURN | {json.dumps(payload, ensure_ascii=False, default=str)}"
        with open(_DEBUG_FILE, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(line + "\n")
            fh.flush()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# SAFETY NET: recover tool calls some backends emit as literal text
# ═══════════════════════════════════════════════════════════════════
#
# Some backends (observed with certain OpenRouter free models used in the
# Ollama-down fallback path) don't populate the API's structured
# `message.tool_calls` field. Instead they write the call directly into
# `message.content` using Hermes' own token format:
#
#   <|tool_call_start|>[fn_name(arg='val', other=123)]<|tool_call_end|>
#
# Before this, that text had nowhere to go: it isn't `tool_calls`, so the
# tool never ran, and it isn't stripped from `content`, so the raw tokens
# got printed straight to the user as if they were the final answer.
# This parses that format into the same shape real structured tool calls
# use, so it flows into the existing _run_tool_calls() pipeline unchanged,
# and strips the matched text out of what the user actually sees.

_PSEUDO_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)


def _extract_pseudo_tool_calls(content: str):
    """Parse '<|tool_call_start|>[fn(arg=val,...)]<|tool_call_end|>' text into
    structured tool_calls. Returns (tool_calls, cleaned_content). Never
    raises — a block that fails to parse is just left in the text so
    nothing silently vanishes.
    """
    if not content or "<|tool_call_start|>" not in content:
        return [], content

    tool_calls = []
    cleaned = content

    for match in _PSEUDO_TOOL_CALL_RE.finditer(content):
        raw = match.group(1).strip()
        # Some models wrap the call in an extra list bracket: [fn(...)]
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1].strip()
        try:
            node = ast.parse(raw, mode="eval").body
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                _debug("PSEUDO_TOOL_CALL_NOT_A_CALL", {"raw": raw[:200]})
                continue
            fn_name = node.func.id
            args = {
                kw.arg: ast.literal_eval(kw.value)
                for kw in node.keywords if kw.arg
            }
            for i, a in enumerate(node.args):
                # Positional args are unusual for this format but captured
                # defensively rather than silently dropped.
                try:
                    args[f"_pos{i}"] = ast.literal_eval(a)
                except Exception:
                    pass
            tool_calls.append({"function": {"name": fn_name, "arguments": args}})
            cleaned = cleaned.replace(match.group(0), "", 1)
            _debug("PSEUDO_TOOL_CALL_PARSED", {"fn": fn_name, "args": args})
        except Exception as exc:
            _debug("PSEUDO_TOOL_CALL_PARSE_FAIL", {"raw": raw[:200], "error": str(exc)})
            continue

    return tool_calls, cleaned.strip()


# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTIC: Log Ollama response metadata for every main-brain call
# ═══════════════════════════════════════════════════════════════════

def _log_brain_response(response, label="BRAIN_RESPONSE_DIAGNOSTIC"):
    """Safely log Ollama ChatResponse termination metadata."""
    try:
        # The ollama package returns either a dict-like ChatResponse or a plain dict.
        # Use getattr as a safe fallback so we never crash if a field is missing.
        if isinstance(response, dict):
            done = response.get("done")
            done_reason = response.get("done_reason")
            eval_count = response.get("eval_count")
            prompt_eval_count = response.get("prompt_eval_count")
            model = response.get("model")
            msg = response.get("message", {}) or {}
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "") or ""
        else:
            done = getattr(response, "done", None)
            done_reason = getattr(response, "done_reason", None)
            eval_count = getattr(response, "eval_count", None)
            prompt_eval_count = getattr(response, "prompt_eval_count", None)
            model = getattr(response, "model", None)
            msg = getattr(response, "message", None) or {}
            content = getattr(msg, "content", "") if msg else ""

        _debug(label, {
            "provider": "ollama",
            "model": model,
            "done": done,
            "done_reason": done_reason,
            "eval_count": eval_count,
            "prompt_eval_count": prompt_eval_count,
            "content_length": len(content) if content else 0,
            "content_empty": not bool(content.strip()) if content else True,
            "content_preview": (content[:200] + "...") if content and len(content) > 200 else content,
        })
    except Exception as exc:
        _debug(f"{label}_ERROR", {"error": str(exc)})

# ═══════════════════════════════════════════════════════════════════
# FAST PATH — keyword classifier (zero LLM call)
# ═══════════════════════════════════════════════════════════════════
_SEARCH_KEYWORDS = {
    "web search", "search for", "search ", "look up", "google ", "who won", "who is",
    "what is the", "current weather", "weather", "weather in", "latest news", "news",
    "how to", "when did", "where is", "why does", "price of",
}

FAST_SYSTEM_PROMPT = (
    "You are Fairy, Master's sarcastic AI companion. "
    "Reply naturally, briefly, with your usual wit. No tools needed."
)

_CHAT_KEYWORDS = {
    "wassup", "sup", "hey", "hello", "hi ", "yo ", "howdy", "hola",
    "good morning", "good afternoon", "good evening", "good night",
    "what's up", "whats up", "how are you", "how r u", "how is it going",
    "how's it going", "nice to meet you", "bye", "goodbye", "see ya",
    "thanks", "thank you", "lol", "lmao", "haha", "ok", "okay", "k",
    "sure", "yeah", "nah", "nope", "yep", "maybe", "idk", "dunno",
    "what do you think", "tell me a joke", "joke", "funny",
}

_TIME_KEYWORDS = {
    "what time", "what's the time", "whats the time", "current time",
    "what day", "what date", "what's today", "whats today",
    "clock", "what hour", "what minute",
}

_CODE_KEYWORDS = {
    "code", "script", "function", "program", "python", "javascript",
    "html", "css", "api", "bug", "fix", "error", "debug",
    "write a", "create a script", "make a program", "build a",
}

_BROWSER_KEYWORDS = {
    "open ", "go to ", "navigate to ", "click ",
    "type ", "browse ", "website", "url ", "www.", "http",
}

# Known websites that should ALWAYS use browser navigation.
# These are unambiguous - "open youtube" clearly means open in browser.
# Requests NOT in this set and matching the "open " pattern are ambiguous
# (could be native app OR website) and should route to Hermes for resolution.
_KNOWN_WEBSITES = frozenset({
    # Major websites
    "youtube", "youtube.com", "youtu.be",
    "reddit", "reddit.com", "old.reddit.com",
    "twitter", "twitter.com", "x.com",
    "facebook", "facebook.com", "fb.com",
    "instagram", "instagram.com",
    "tiktok", "tiktok.com",
    "twitch", "twitch.tv",
    "discord", "discord.com",
    "whatsapp", "whatsapp.com",
    "telegram", "telegram.org",
    "amazon", "amazon.com", "amazn.com",
    "ebay", "ebay.com",
    "netflix", "netflix.com",
    "spotify", "spotify.com",
    "linkedin", "linkedin.com",
    "github", "github.com",
    "gitlab", "gitlab.com",
    "stackoverflow", "stackoverflow.com",
    "gmail", "gmail.com", "mail.google.com",
    "outlook", "outlook.com", "hotmail.com",
    "wikipedia", "wikipedia.org",
    "bing", "bing.com",
    "google", "google.com", "google.co.uk", "google.com.au",
    "duckduckgo", "duckduckgo.com",
    "chatgpt", "chat.openai.com", "chatgpt.com",
    "claude", "claude.ai",
    "wolfram", "wolframalpha.com",
    "imdb", "imdb.com",
    "pinterest", "pinterest.com",
    "tumblr", "tumblr.com",
    "medium", "medium.com",
    "notion", "notion.so",
    "slack", "slack.com",
    "zoom", "zoom.us",
    "teams", "teams.microsoft.com",
    "dropbox", "dropbox.com",
    "drive.google", "docs.google",
    "sheets.google", "spreadsheets.google",
    "calendar.google",
    "maps.google", "maps.google.com",
    "translate.google",
    # Generic website indicators
    ".com", ".org", ".net", ".io", ".co",
})

# Patterns that indicate a URL is being specified
_URL_PATTERNS = re.compile(
    r"^(https?://|www\.|[\w-]+\.(com|org|net|io|co|gov|edu|ai|app|dev))",
    re.IGNORECASE
)

# Patterns for action verbs that are ambiguous (could be app or website)
# These should route to Hermes for reasoning, not the browser fast path
_LAUNCH_ACTION_PATTERNS = re.compile(
    r"^(start|launch|run|close|kill|quit|exit|stop|terminate)\s+(\S+)",
    re.IGNORECASE
)

# Known desktop apps that should NOT be treated as websites even if similar names exist
_KNOWN_DESKTOP_APPS = frozenset({
    # Productivity
    "notepad", "notepad++", "notepadplusplus", "wordpad", "word", "excel",
    "powerpoint", "onenote", "outlook", "publisher", "access",
    # System
    "calculator", "calc", "taskmgr", "task manager", "explorer", "explorer.exe",
    "cmd", "command prompt", "powershell", "windows terminal", "wt",
    "control panel", "settings", "device manager", "disk management",
    "services", "event viewer", "resource monitor", "performance monitor",
    # Development
    "vscode", "vs code", "visual studio code", "visual studio", "notepad++",
    "sublime", "sublime text", "atom", "pycharm", "intellij", "eclipse",
    "android studio", "xcode", "git", "github desktop",
    # Media
    "spotify", "vlc", "vlc media player", "windows media player", "wmp",
    "groove music", "apple music", "itunes", "audacity", "audacity",
    "obs", "obs studio", "streamlabs", "discord", "slack", "teams",
    "zoom", "skype", "telegram", "whatsapp", "signal",
    # Gaming
    "steam", "epic games", "epic games launcher", "gog", "gog galaxy",
    "battle.net", "origin", "uplay", "ubisoft connect", "minecraft", "roblox",
    "league of legends", "lol", "valorant", "fortnite", "dota 2",
    # Browsers
    "chrome", "google chrome", "firefox", "mozilla firefox", "edge", "ms edge",
    "microsoft edge", "opera", "opera gx", "brave", "vivaldi", "safari",
    # Utilities
    "winrar", "7zip", "7-zip", "ccleaner", "malwarebytes", "avg", "avast",
    "nvidia", "nvidia control panel", "amd radeon", "radeon software",
    # Communication
    "discord", "slack", "teams", "zoom", "skype", "telegram",
    "whatsapp", "signal", "thunderbird", "mail", "mailbird",
})

# Fast-path intents that should NOT route to Hermes (they are handled directly)
_FAST_ACTION_INTENTS = frozenset({
    "chat", "time", "system", "vision", "vision_on", "vision_off",
    "message", "location", "code",
})

_LOC_KEYWORDS = {
    "nearby", "near me", "restaurant", "directions to", "how far",
    "where is", "find ", "map", "location", "address",
    "gas station", "hotel", "cafe", "mall", "hospital",
}

# Tightened: only phrases that clearly mean "look at the screen/camera RIGHT NOW".
# Avoid loose tokens like "vision", "screen", "camera" — those were hijacking
# unrelated conversation. Use screen_vision_keywords() below to apply
# word-boundary rules and context checks.
_SCREEN_KEYWORDS_RAW = {
    "what do you see",
    "what's on my screen", "what is on my screen",
    "what's in my screen", "what is in my screen",
    "look at my screen", "look at the screen",
    "look at my camera", "look at the camera",
    "look at my webcam", "look at the webcam",
    "show me my screen", "show me the screen",
    "describe my screen", "describe the screen",
    "describe what you see", "what do you see on",
    "take a look", "can you see my",
    "what's happening on my screen",
    "what is happening on my screen",
    "what's on screen", "what is on screen",
    "analyze my screen", "analyze the screen",
    "analyze this image", "what is this image",
    "check my screen", "read my screen",
    "read the screen",
}

# Explicit vision on/off commands. These are matched BEFORE any vision keyword
# so "close your vision" never triggers an analyze.
_VISION_ON_PATTERNS = [
    re.compile(r"\b(turn|switch|enable|activate|start|begin)\b.{0,15}\bvision\b", re.I),
    re.compile(r"\bvision\b.{0,15}\b(on|on now|please|active)\b", re.I),
    re.compile(r"\b(enable|activate|start|begin)\b.{0,15}\b(screen|camera|webcam)\b", re.I),
]

_VISION_OFF_PATTERNS = [
    re.compile(r"\b(close|stop|turn off|disable|deactivate|end|exit|kill)\b.{0,15}\bvision\b", re.I),
    re.compile(r"\bvision\b.{0,15}\b(off|disabled|closed|stopped)\b", re.I),
    re.compile(r"\b(stop|close|kill|end)\b.{0,15}\b(watching|looking|seeing|analyzing|observing)\b", re.I),
    re.compile(r"\b(stop|close)\b.{0,15}\b(your|ur)\b.{0,15}\b(eyes|sight|vision|vision mode)\b", re.I),
]


def _vision_on_match(text: str) -> bool:
    return any(p.search(text) for p in _VISION_ON_PATTERNS)


def _vision_off_match(text: str) -> bool:
    return any(p.search(text) for p in _VISION_OFF_PATTERNS)


def _screen_vision_keywords(text: str) -> bool:
    """
    Return True iff `text` clearly requests looking at the screen/camera.
    Uses substring containment (case-insensitive). Tighter than before —
    only phrasings that unambiguously mean 'look right now'.
    """
    lower = text.lower().strip()
    for kw in _SCREEN_KEYWORDS_RAW:
        if kw in lower:
            return True
    return False

_MESSAGE_KEYWORDS = {
    "send message", "whatsapp", "telegram", "discord",
    "instagram", "messenger", "signal", "dm", "message",
}

_TEXT_COMMAND_PATTERN = re.compile(r"(?<!\\)(?<!\w)text(?!\w)")

_SYS_KEYWORDS = {
    "cpu", "gpu", "memory", "ram", "temperature", "battery", "disk",
    "usage", "system", "monitor", "process", "processes", "network", "load",
}

# A bare system-sounding word is not enough on its own to trigger the fast
# path, but neither should detection depend ONLY on personal phrasing like
# "my"/"right now" -- that would wrongly reject legitimate direct commands
# like "Show GPU information." or "List running processes."
#
# Instead: block the match when the message is clearly asking ABOUT
# supplied content (a diagram, JSON, table, doc) rather than requesting
# live machine state. These are the tell-tale phrasings for "content
# reference" rather than "system-state request".
_SYS_CONTENT_REFERENCE_MARKERS = (
    "in this", "this diagram", "this json", "this table", "this document",
    "this architecture", "this workflow", "this pipeline",
    "field mean", "explain this", "explain it", "does this",
)

_MODEL_KEYWORDS = {
    "liquid", "nemotron", "claude", "gpt", "gemma", "qwen", "openrouter",
    "poolside", "cohere", "deepseek", "mistral", "grok", "kimi", "ling",
    "auto-free", "liquid-free", "nemotron-free", "poolside-s", "poolside-xs",
    "cohere-code", "gemma4-26b", "gemma4-31b", "gpt-oss",
}


def _keyword_matches(text: str, keyword: str) -> bool:
    kw = keyword.strip()
    if not kw:
        return False
    if re.fullmatch(r"[\w\' ]+", kw):
        return re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", text) is not None
    return kw in text


_MATH_OP_WORDS = {
    "times": "*", "multiplied by": "*", "multiply": "*", "x": "*", "*": "*",
    "plus": "+", "add": "+", "added to": "+", "+": "+",
    "minus": "-", "subtract": "-", "subtracted by": "-", "-": "-",
    "divided by": "/", "divide": "/", "/": "/",
    "to the power of": "**", "raised to the power of": "**", "^": "**", "power": "**",
}

_MATH_OP_PATTERN = "|".join(
    re.escape(op) for op in sorted(_MATH_OP_WORDS, key=len, reverse=True)
)

_MATH_PATTERN = re.compile(
    r"(-?[\d,]+\.?\d*)\s*(" + _MATH_OP_PATTERN + r")\s*(-?[\d,]+\.?\d*)",
    re.IGNORECASE,
)


def _try_fast_math(text: str) -> str | None:
    text = text.replace("×", "*").replace("÷", "/")
    m = _MATH_PATTERN.search(text)
    if not m:
        return None
    try:
        a_str, op_str, b_str = m.group(1), m.group(2).lower().strip(), m.group(3)
        op = _MATH_OP_WORDS.get(op_str)
        if op is None:
            return None

        def _to_num(s: str):
            s = s.replace(",", "")
            return float(s) if "." in s else int(s)

        a, b = _to_num(a_str), _to_num(b_str)

        if op == "*":
            result = a * b
        elif op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "/":
            if b == 0:
                return "Can't divide by zero, Master. Even I have limits."
            result = a / b
        elif op == "**":
            try:
                result = a ** b
            except OverflowError:
                return f"{a:,} to the power of {b:,} is a number so large I can't even pretend to print it, Master."
        else:
            return None

        if isinstance(result, float) and result.is_integer():
            result = int(result)

        return f"{a:,} {op_str} {b:,} = {result:,}"
    except Exception:
        return None


# =====================================================================
# DETERMINISTIC UNIT CONVERSION (zero LLM)
# =====================================================================
def _try_unit_conversion(text: str) -> str | None:
    text = text.lower().strip()
    patterns = [
        (r"(\d+\.?\d*)\s*(km|kilometers?)\s+to\s+(miles?|mi)", lambda v, u: f"{v} km = {v * 0.621371:.2f} miles"),
        (r"(\d+\.?\d*)\s*(miles?|mi)\s+to\s+(km|kilometers?)", lambda v, u: f"{v} miles = {v / 0.621371:.2f} km"),
        (r"(\d+\.?\d*)\s*(celsius|c)\s+to\s+(fahrenheit|f)", lambda v, u: f"{v}°C = {v * 9/5 + 32:.1f}°F"),
        (r"(\d+\.?\d*)\s*(fahrenheit|f)\s+to\s+(celsius|c)", lambda v, u: f"{v}°F = {(v - 32) * 5/9:.1f}°C"),
        (r"(\d+\.?\d*)\s*(kg|kilograms?)\s+to\s+(pounds?|lb)", lambda v, u: f"{v} kg = {v * 2.20462:.2f} lb"),
        (r"(\d+\.?\d*)\s*(pounds?|lb)\s+to\s+(kg|kilograms?)", lambda v, u: f"{v} lb = {v / 2.20462:.2f} kg"),
    ]
    for pattern, func in patterns:
        m = re.search(pattern, text)
        if m:
            try:
                val = float(m.group(1))
                return func(val, m.group(2))
            except:
                return None
    return None


# =====================================================================
# APP ACTION FALLBACK (when Hermes is unavailable)
# =====================================================================
def _extract_app_action_target(text: str) -> tuple[str, str] | None:
    """Extract (verb, target) from app action text.
    Returns None if no action verb found.
    Verbs: open, start, launch, run, close, kill, quit, exit, stop, terminate
    """
    text_lower = text.lower().strip()
    # Use lookahead to NOT consume trailing qualifiers like " app", " for me", " please"
    m = re.match(
        r"^(open|start|launch|run|close|kill|quit|exit|stop|terminate)\s+(.+?)(?=\s+app\b|\s+for\s+me|\s+please|\s+now|\s*$)",
        text_lower,
    )
    if not m:
        return None
    verb = m.group(1)
    target = m.group(2).strip()
    if not target:
        return None
    return (verb, target)


def _is_process_running(process_name: str) -> bool | None:
    """Check if a process is currently running on Windows.
    Returns True/False if determinable, None if can't check (non-Windows, no psutil).
    """
    process_name_lower = process_name.lower().replace(" ", "")
    # Common process name mappings
    name_aliases = {
        "chrome": ["chrome.exe"],
        "google chrome": ["chrome.exe"],
        "firefox": ["firefox.exe"],
        "edge": ["msedge.exe"],
        "microsoft edge": ["msedge.exe"],
        "vscode": ["code.exe"],
        "vs code": ["code.exe"],
        "visual studio code": ["code.exe"],
        "notepad": ["notepad.exe"],
        "notepad++": ["notepad++.exe"],
        "calculator": ["calculator.exe", "calc.exe"],
        "calc": ["calculator.exe", "calc.exe"],
        "taskmgr": ["taskmgr.exe"],
        "task manager": ["taskmgr.exe"],
        "explorer": ["explorer.exe"],
        "file explorer": ["explorer.exe"],
        "cmd": ["cmd.exe"],
        "command prompt": ["cmd.exe"],
        "powershell": ["powershell.exe"],
        "spotify": ["spotify.exe"],
        "discord": ["discord.exe"],
        "slack": ["slack.exe"],
        "teams": ["teams.exe"],
        "zoom": ["zoom.exe"],
        "telegram": ["telegram.exe"],
        "steam": ["steam.exe", "steamwebhelper.exe"],
        "vlc": ["vlc.exe"],
        "obs": ["obs64.exe", "obs.exe"],
        "obs studio": ["obs64.exe", "obs.exe"],
        "skype": ["skype.exe", "lync.exe"],
        "outlook": ["outlook.exe"],
        "word": ["winword.exe"],
        "excel": ["excel.exe"],
        "powerpoint": ["powerpoint.exe"],
    }
    candidates = name_aliases.get(process_name_lower, [f"{process_name_lower}.exe"])

    try:
        import platform
        if platform.system() != "Windows":
            return None
        # Use psutil if available, else fall back to tasklist
        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                name = (proc.info.get('name') or '').lower()
                if name in [c.lower() for c in candidates]:
                    return True
            return False
        except ImportError:
            # Fall back to tasklist
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout.lower()
            for cand in candidates:
                if cand.lower() in output:
                    return True
            return False
    except Exception:
        return None


def _verify_app_launch(target: str, verb: str, timeout_seconds: float = 3.0) -> tuple[bool, str]:
    """Verify whether an app launch succeeded.
    Returns (success, message).
    For "open/start/launch/run": checks if the process is running after a short delay.
    For "close/kill/quit/exit/stop/terminate": checks if the process is no longer running.
    """
    import time as _time
    # Brief wait for the app to start/shut down
    _time.sleep(min(timeout_seconds, 1.0))

    is_running = _is_process_running(target)

    if verb in ("close", "kill", "quit", "exit", "stop", "terminate"):
        # For close operations, success means the process is no longer running
        if is_running is False:
            return True, f"{target} closed."
        elif is_running is True:
            return False, f"{target} is still running."
        else:
            # Couldn't verify - report as not verified
            return True, f"Attempted to close {target}. (Could not verify.)"
    else:
        # For open/start/launch/run, success means the process is running
        if is_running is True:
            return True, f"{target} opened."
        elif is_running is False:
            return False, f"{target} did not start. It may not be installed."
        else:
            # Couldn't verify
            return True, f"Attempted to open {target}. (Could not verify.)"


# ═══════════════════════════════════════════════════════════════════
# File-creation claim verification — FIX 2
#
# Some backends (Hermes' gemma4 + OpenRouter free fallback especially)
# narrate file creations/deletes as "done" without ever calling a tool.
# This helper extracts any path mentioned in a creation claim and checks
# os.path.exists for it. The task-action guard in handle_request consults
# this to decide whether to trust a creation claim as success.
# ═══════════════════════════════════════════════════════════════════

# Words in the assistant reply that constitute a creation/deletion claim.
_CREATION_CLAIM_RE = re.compile(
    r"\b(created|wrote|written|saved|deleted|removed|built|generated|created successfully)\b"
    r"|\bstatus:\s*(created|saved|done|ok)\b",
    re.IGNORECASE,
)

# Words in user text that indicate the request was file-related.
_FILE_REQUEST_RE = re.compile(
    r"\b(make|create|write|save|delete|remove|rename|copy|move)\b"
    r"|\b(file|txt|note|document|script|folder|directory|markdown|md)\b",
    re.IGNORECASE,
)

# Matches a plausible file path inside a reply (Windows drive, POSIX abs, or bare name.ext).
_PATH_IN_REPLY_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|:*?\n\r]+"
    r"|/(?:[^\s\"'<>|:*?\n\r]+/)*[^\s\"'<>|:*?\n\r]+"
    r"|[A-Za-z0-9_.\-][^\s\"'<>|:*?\n\r]*\.[A-Za-z0-9]{1,10})"
)

# Bare file-verb phrase detector (FIX 1). Matches "make/create/write/save/delete
# a file" but NOT app verbs (open/launch/close) — those already have fast-path
# dispatch, so the counter is incremented normally.
_TASK_FILE_VERB_RE = re.compile(
    r"\b(make|create|write|save|delete|remove|rename|copy|move)\b"
    r"|\b(new|edit)\s+(file|txt|note|text|script|document)\b",
    re.IGNORECASE,
)


def _is_task_file_request(user_text: str) -> bool:
    """True if user is asking for a file action that needs a tool.

    Only file verbs (create/make/write/delete a file). App verbs (open chrome,
    launch discord) already have fast-path tool dispatch — including them here
    would just add noise.
    """
    return bool(_TASK_FILE_VERB_RE.search(user_text or ""))


def _is_artifact_non_empty(path: str) -> bool:
    """
    Check whether a path exists and contains non-empty content.

    For files: size > 0.
    For directories: contains at least one entry.

    Returns False for non-existent paths or empty artifacts.
    """
    try:
        if not os.path.exists(path):
            return False
        if os.path.isfile(path):
            try:
                return os.path.getsize(path) > 0
            except OSError:
                return False
        if os.path.isdir(path):
            try:
                entries = os.listdir(path)
                if not entries:
                    return False
                # If a single subdirectory exists but is itself empty, also fail
                for entry in entries[:5]:
                    entry_path = os.path.join(path, entry)
                    if os.path.isfile(entry_path):
                        if os.path.getsize(entry_path) > 0:
                            return True
                    elif os.path.isdir(entry_path):
                        try:
                            if os.listdir(entry_path):
                                return True
                        except OSError:
                            continue
                return False
            except OSError:
                return False
    except OSError:
        return False
    return False


def _is_plausible_creation_path(path: str) -> bool:
    """
    Filter noise paths out of the candidate list before existence checks.

    The path-in-reply regex can pick up junk like ``/.`` or a bare ``\\``
    from surrounding punctuation; these resolve to existing system
    directories and would falsely pass verification if the *real* project
    path failed its check. Require a real file/folder shape: at least 3
    characters, must contain either a separator or a name+extension.
    """
    if not path or len(path) < 3:
        return False
    # Strip trailing punctuation that the regex captured
    path = path.rstrip(".,;:!? )\"'")
    if not path:
        return False
    # Path must contain a separator OR a file extension (the regex's
    # third alternative). Pure single characters like "/" or "." or ".." are
    # not real artifacts.
    if path in {".", "..", "/", "\\", "//", "\\\\"}:
        return False
    if path in {"/.", "\\.", "/..", "\\.."}:
        return False
    # If no separator and no dot, it's not a real file/folder name.
    has_sep = any(sep in path for sep in (os.sep, "/", "\\"))
    has_ext = "." in os.path.basename(path)
    if not (has_sep or has_ext):
        return False
    return True


def _verify_creation_claims(reply: str, user_text: str) -> tuple[bool, str]:
    """
    Check whether a reply that claims a file was created/written/deleted
    can be verified on disk AND is non-empty.

    Returns:
        (True, "")            - no creation claim, OR claim verified on disk
                                 and artifact is non-empty.
        (True, "<path>")      - verified, path found on disk with content.
        (False, "")           - claim made, user asked for a file, but no
                                 path from the claim exists, or the path
                                 exists but is empty (FIX: empty artifact
                                 is treated as a verification failure).
    """
    if not reply or not _CREATION_CLAIM_RE.search(reply):
        return True, ""   # no claim to verify
    if not _FILE_REQUEST_RE.search(user_text or ""):
        return True, ""   # reply mentions "created" but user didn't ask for a file action
    for m in _PATH_IN_REPLY_RE.finditer(reply):
        path = m.group(0).rstrip(".,;:!? )\"'")
        # Filter noise paths (e.g. "tests/." picked up by trailing punctuation)
        if not _is_plausible_creation_path(path):
            continue
        # Normalise separators
        path = path.replace("/", os.sep)
        # Existence + non-empty check. An empty artifact (0-byte file or
        # empty folder) is a verification failure — the agent should not
        # claim success on a tool that created nothing.
        try:
            if os.path.exists(path) and _is_artifact_non_empty(path):
                return True, path
        except OSError:
            continue
    _debug("CREATION_CLAIM_UNVERIFIED", {
        "reply_preview": reply[:300],
        "user_text_preview": (user_text or "")[:200],
    })
    return False, ""


def _try_app_action_fallback(text: str) -> str | None:
    """Direct app action fallback when Hermes is unavailable.
    Tries to execute the action via computer_control and verifies the result.
    Returns a response string if handled, None if it should fall through to LLM.
    """
    extracted = _extract_app_action_target(text)
    if not extracted:
        return None
    verb, target = extracted

    # ── Internal-agent name denylist ───────────────────────────────────────
    # Hermes is Fairy's own agent layer and cannot be launched as a desktop app.
    # Similarly qwen/gemma/ollama are local inference engines, not installable apps.
    # Attempting to launch these produces WinError 2 ("file not found") with no
    # recovery path. Refuse them explicitly so the user gets a useful error.
    try:
        from hermes_bridge import is_internal_agent_name
        if is_internal_agent_name(target):
            _debug("APP_ACTION_INTERNAL_NAME_REJECTED", {"target": target, "verb": verb})
            return (
                f"I can't launch '{target}' as a desktop app, Master — "
                f"it's an internal system component, not an installable program. "
                f"I'm already running and handling your requests."
            )
    except Exception:
        pass  # hermes_bridge may not be available in all environments
    try:
        from skills.computer_control import computer_control
    except ImportError:
        return None

    # Map verb to computer_control action
    if verb in ("open", "start", "launch", "run"):
        # For URLs, use open_url; for everything else, use open
        if target.startswith(("http://", "https://", "www.")):
            action = "open_url"
        else:
            action = "open"
    else:  # close, kill, quit, exit, stop, terminate
        action = "close_app"

    _debug("APP_ACTION_FALLBACK_EXEC", {"verb": verb, "target": target, "action": action})

    try:
        result = computer_control({"action": action, "value": target})
    except Exception as exc:
        _debug("APP_ACTION_FALLBACK_EXEC_ERROR", {"error": str(exc)})
        return f"Failed to {verb} {target}: {exc}"

    # Parse result
    try:
        import json
        if isinstance(result, str):
            parsed = json.loads(result)
        elif isinstance(result, dict):
            parsed = result
        else:
            parsed = {"status": "error", "message": str(result)}

        status = parsed.get("status", "error")
        if status == "ok":
            # Verify the action actually worked
            success, message = _verify_app_launch(target, verb)
            if success:
                return message
            else:
                return f"{verb.capitalize()} {target} failed: {message}"
        else:
            err_msg = parsed.get("message", "Unknown error")
            return f"Couldn't {verb} {target}: {err_msg}"
    except Exception as exc:
        return f"Tried to {verb} {target}, but couldn't verify: {exc}"


# ═══════════════════════════════════════════════════════════════════
# STRUCTURED / COMPLEX MESSAGE DETECTION
#
# The keyword-based _fast_intent() classifier below does whole-word
# substring matching over the ENTIRE raw message. That's fine for a
# short command like "what's my GPU usage" — but it's unsafe for a
# message that CONTAINS words like "gpu"/"process"/"load" as part of
# a diagram, a JSON payload, or the user's own question text (e.g.
# "is there any info about GPU usage in this JSON?"). A bare keyword
# match can't tell the difference; only real reasoning (the main LLM,
# via genuine function-calling in _adaptive_loop) can.
#
# So: any message that LOOKS structured or complex is routed straight
# past the fast-path classifier and into the normal LLM path, where
# tool selection is done by the model itself, not by substring match.
# ═══════════════════════════════════════════════════════════════════
_BOX_DRAWING_CHARS = set("│├└┌┐─▼▲◄►═║┼┬┴┤╔╗╚╝╠╣╦╩╬")
_NUMBERED_LIST_PATTERN = re.compile(r"(?m)^\s*\d+[\.\)]\s")
_JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*\"[^\"]+\"\s*:")

_STRUCTURED_WORD_LIMIT = 60
_STRUCTURED_LINE_LIMIT = 8


def _looks_structured_or_complex(text: str) -> bool:
    """
    True if the message contains an ASCII diagram, fenced code, a JSON-like
    object, a numbered multi-part question list, or is simply long/multi-
    paragraph. Any of these means: don't trust a bare keyword match, let the
    real LLM decide what (if anything) needs a tool.
    """
    if "```" in text:
        return True
    if any(ch in _BOX_DRAWING_CHARS for ch in text):
        return True
    stripped_for_json = text.strip()
    if stripped_for_json.startswith("{") or stripped_for_json.startswith("["):
        return True
    if _JSON_OBJECT_PATTERN.search(text):
        return True
    if len(_NUMBERED_LIST_PATTERN.findall(text)) >= 2:
        return True
    # Multi-step "search X then open Y" phrasing should NOT be fast-pathed to
    # the browser — the search and the open are two distinct actions that
    # need agent reasoning to sequence (and possibly search first, then open
    # the result). Without this check, "first search for X then open it"
    # falls into the browser fast path and never actually runs the search.
    if re.search(r"\bthen\b.{0,40}\b(search|look\s*up|find|open)\b", text, re.IGNORECASE):
        return True
    word_count = len(text.split())
    line_count = text.count("\n") + 1
    if word_count > _STRUCTURED_WORD_LIMIT or line_count > _STRUCTURED_LINE_LIMIT:
        return True
    return False


def _fast_intent(user_text: str) -> str | None:
    text = user_text.lower().strip()
    _debug("FAST_INTENT_INPUT", {"text": text})

    # If user names a model, skip fast path so tools are available
    for kw in _MODEL_KEYWORDS:
        if _keyword_matches(text, kw):
            _debug("FAST_INTENT_SKIP_MODEL", {"matched": kw})
            return None

    # --- System monitoring ---
    # Require a system-sounding keyword AND the absence of a content-
    # reference marker, so a direct command ("Show GPU information.",
    # "List running processes.") still fires, while a message asking
    # ABOUT supplied content ("Is there GPU info in this JSON?",
    # "Explain this process diagram.") does not.
    _sys_kw_hit = None
    for w in _SYS_KEYWORDS:
        if _keyword_matches(text, w):
            _sys_kw_hit = w
            break
    if _sys_kw_hit is not None:
        _is_content_reference = any(marker in text for marker in _SYS_CONTENT_REFERENCE_MARKERS)
        if not _is_content_reference:
            _debug("FAST_INTENT_HIT", {"intent": "system", "matched": _sys_kw_hit})
            return "system"
        _debug("FAST_INTENT_SYS_KEYWORD_CONTENT_REFERENCE", {"matched": _sys_kw_hit})

    # --- websearch fast path is DISABLED to allow Gemma4 to use search_and_summarize ---
    # for w in _SEARCH_KEYWORDS:
    #     if _keyword_matches(text, w):
    #         _debug("FAST_INTENT_HIT", {"intent": "websearch", "matched": w})
    #         return "websearch"

    # Special handling for "open X" pattern - distinguish app vs website
    # "open youtube" → browser (website)
    # "open calculator" → ambiguous (could be native app)
    # "open notepad" → ambiguous (could be native app)
    # "open steam" → ambiguous (could be native app)
    # "open https://..." → browser (explicit URL)
    _open_match = re.match(r"^open\s+(.+?)(?:\s|$)", text)
    if _open_match:
        target = _open_match.group(1).strip().lower()
        # Check if it's a known website
        if any(site in target for site in _KNOWN_WEBSITES):
            _debug("FAST_INTENT_HIT", {"intent": "browser", "matched": f"open {target}", "reason": "known_website"})
            return "browser"
        # Check if it's an explicit URL
        if _URL_PATTERNS.match(target):
            _debug("FAST_INTENT_HIT", {"intent": "browser", "matched": f"open {target}", "reason": "explicit_url"})
            return "browser"
        # Known desktop app → app action intent (route to Hermes for launch)
        if any(app in target for app in _KNOWN_DESKTOP_APPS):
            _debug("FAST_INTENT_HIT", {"intent": "app_action", "matched": f"open {target}", "reason": "known_desktop_app"})
            return "app_action"
        # Ambiguous - could be native app OR website → route to Hermes
        _debug("FAST_INTENT_HIT", {"intent": "ambiguous", "matched": f"open {target}", "reason": "needs_agent_reasoning"})
        return "ambiguous"

    # Action verbs: start, launch, run, close, kill, quit, exit, stop, terminate
    # Examples: "start steam", "launch discord", "close chrome", "kill notepad"
    # These are almost always desktop app actions, not browser navigation.
    _launch_match = _LAUNCH_ACTION_PATTERNS.match(text)
    if _launch_match:
        verb = _launch_match.group(1).lower()
        target = _launch_match.group(2).strip().lower()
        # All "launch/start/run/close/kill/quit" requests are app actions
        # (none of them make sense for websites) → route to Hermes
        _debug("FAST_INTENT_HIT", {
            "intent": "app_action",
            "matched": f"{verb} {target}",
            "reason": "action_verb_always_app",
        })
        return "app_action"

    for w in _BROWSER_KEYWORDS:
        if _keyword_matches(text, w):
            _debug("FAST_INTENT_HIT", {"intent": "browser", "matched": w})
            return "browser"

    for w in _CODE_KEYWORDS:
        if _keyword_matches(text, w):
            _debug("FAST_INTENT_HIT", {"intent": "code", "matched": w})
            return "code"

    for w in _LOC_KEYWORDS:
        if _keyword_matches(text, w):
            _debug("FAST_INTENT_HIT", {"intent": "location", "matched": w})
            return "location"

    # Check for explicit vision on/off commands FIRST (bypass LLM entirely).
    if _vision_off_match(text):
        _debug("FAST_INTENT_HIT", {"intent": "vision_off", "matched": "vision_off_pattern"})
        return "vision_off"
    if _vision_on_match(text):
        _debug("FAST_INTENT_HIT", {"intent": "vision_on", "matched": "vision_on_pattern"})
        return "vision_on"

    # Tightened screen-vision match (no loose tokens like "vision" alone).
    if _screen_vision_keywords(text):
        _debug("FAST_INTENT_HIT", {"intent": "vision", "matched": "screen_vision_keywords"})
        return "vision"

    for w in _MESSAGE_KEYWORDS:
        if _keyword_matches(text, w):
            _debug("FAST_INTENT_HIT", {"intent": "message", "matched": w})
            return "message"
    if _TEXT_COMMAND_PATTERN.search(text):
        _debug("FAST_INTENT_HIT", {"intent": "message", "matched": "text"})
        return "message"

    for w in _TIME_KEYWORDS:
        if _keyword_matches(text, w):
            _debug("FAST_INTENT_HIT", {"intent": "time", "matched": w})
            return "time"

    # Disabled fast‑path chat – all chat goes through the LLM for context-aware replies
# if len(text) <= 25:
#     for w in _CHAT_KEYWORDS:
#         if _keyword_matches(text, w):
#             _debug("FAST_INTENT_HIT", {"intent": "chat", "matched": w})
#             return "chat"
# elif len(text) < 50:
#     for w in _CHAT_KEYWORDS:
#         if text.startswith(w) or text == w:
#             _debug("FAST_INTENT_HIT", {"intent": "chat", "matched": w})
#             return "chat"

    _debug("FAST_INTENT_UNCERTAIN")
    return None


# ═══════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "List all currently available skills.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_skill",
            "description": "Run an existing skill by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Exact skill name, e.g. 'calculator'"},
                    "arguments": {"type": "object", "description": "Arguments dict for the skill"}
                },
                "required": ["name", "arguments"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": "Create a brand new skill. Only use when NO existing skill handles it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "snake_case skill name"},
                    "description": {"type": "string", "description": "what the skill should do"},
                },
                "required": ["name", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the actual current time/date from the system clock. Instant and always correct. Do NOT use web_search for this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Optional IANA timezone. Omit for local time."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_qwen",
            "description": "Local coder/reasoning model. FREE, offline, instant. Second opinion on facts, or code refinement. NOT for live/current info.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Quick web search for current facts, weather, news, docs. Use for live/current information. Follow the current date in the system prompt and never invent a stale year. FREE.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch full text from a specific URL. FREE.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deep_research",
            "description": "Deep multi-source research (Reddit, Quora, forums). For current requests follow the current date/year in the system prompt; never invent a stale year. FREE.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_location",
            "description": "Find coordinates and address of a place.",
            "parameters": {
                "type": "object",
                "properties": {"place": {"type": "string"}},
                "required": ["place"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_directions",
            "description": "Get driving directions between two places.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["origin", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_nearby",
            "description": "Search for places near a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "query": {"type": "string"}
                },
                "required": ["location", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_sandbox",
            "description": "Write Python code, auto-test it, and auto-fix via web search if it fails. Returns working code or error after retries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "What the code should do"},
                    "use_cloud": {"type": "boolean", "description": "Use OpenRouter Qwen 3 instead of local Qwen", "default": False}
                },
                "required": ["description"]
            }
        }
    },
]

if _OPENROUTER_AVAILABLE:
    TOOLS.extend([
        {
            "type": "function",
            "function": {
                "name": "ask_openrouter",
                "description": (
                    "Call ANY OpenRouter model by nickname or full ID. "
                    "Nicknames: qwen3-coder, claude-sonnet, claude-opus, gpt5, o3-mini, "
                    "gemma4-9b, gemma4-27b, llama4, deepseek-v3, mistral-large, grok2, kimi-k2, gemini-flash. "
                    "You can also pass raw IDs like 'openai/gpt-5'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "model": {"type": "string", "default": "qwen3-coder"},
                        "system_prompt": {"type": "string", "default": None},
                    },
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ask_or_coder",
                "description": "OpenRouter Qwen 3 Coder — best cloud programmer.",
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ask_or_smart",
                "description": "OpenRouter Claude Sonnet 4 — best cloud reasoning.",
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ask_or_cheap",
                "description": "OpenRouter Gemma 4 9B — cheap & fast.",
                "parameters": {
                    "type": "object",
                    "properties": {"prompt": {"type": "string"}},
                    "required": ["prompt"]
                }
            }
        },
        # ================================================================
        # NEW: search_and_summarize tool – offloads search+summarisation to OpenRouter
        # ================================================================
        {
            "type": "function",
            "function": {
                "name": "search_and_summarize",
                "description": (
                    "Search the web for a query and summarise the results using a powerful cloud model. "
                    "Use this for any question that requires up‑to‑date information, news, weather, research, "
                    "or comparisons. This tool performs a web search and returns a concise summary, "
                    "so you don't have to process raw search results yourself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query (e.g., 'GTA 6 latest news')"}
                    },
                    "required": ["query"]
                }
            }
        },
    ])

if _LLM_APIS_AVAILABLE:
    TOOLS.extend([
        {"type": "function", "function": {"name": "ask_openai", "description": "OpenAI API (PAID)", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "model": {"type": "string", "default": "gpt-4o-mini"}}, "required": ["prompt"]}}},
        {"type": "function", "function": {"name": "ask_claude", "description": "Claude API (PAID)", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "model": {"type": "string", "default": "claude-3-haiku-20240307"}}, "required": ["prompt"]}}},
        {"type": "function", "function": {"name": "ask_gemini", "description": "Gemini API (PAID)", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "model": {"type": "string", "default": "gemini-1.5-flash"}}, "required": ["prompt"]}}},
        {"type": "function", "function": {"name": "ask_grok", "description": "Grok API (PAID)", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "model": {"type": "string", "default": "grok-2"}}, "required": ["prompt"]}}},
        {"type": "function", "function": {"name": "ask_meta", "description": "Meta Llama API (PAID)", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "model": {"type": "string", "default": "llama-3.3-70b"}}, "required": ["prompt"]}}},
        {"type": "function", "function": {"name": "ask_deepseek", "description": "DeepSeek API (PAID)", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "model": {"type": "string", "default": "deepseek-chat"}}, "required": ["prompt"]}}},
        {"type": "function", "function": {"name": "ask_kimi", "description": "Kimi API (PAID)", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "model": {"type": "string", "default": "moonshot-v1-8k"}}, "required": ["prompt"]}}},
    ])

TOOLS.extend([
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Open a URL in the browser. Use this for websites such as YouTube.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "browser": {"type": "string", "default": "default"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_on_site",
            "description": "Open a website search page for a query. For YouTube, search music/videos by query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "site": {"type": "string"},
                    "query": {"type": "string"},
                    "browser": {"type": "string", "default": "default"},
                },
                "required": ["site", "query"],
            },
        },
    },
])

if _BROWSER_AVAILABLE:
    TOOLS.extend([
        {"type": "function", "function": {"name": "add_to_cart_amazon", "description": "Search Amazon and add first match to cart.", "parameters": {"type": "object", "properties": {"item_name": {"type": "string"}, "browser": {"type": "string", "default": "opera_gx"}}, "required": ["item_name"]}}},
        {"type": "function", "function": {"name": "upload_instagram_reel", "description": "Open Instagram Reels upload.", "parameters": {"type": "object", "properties": {"video_path": {"type": "string"}, "caption": {"type": "string", "default": ""}, "browser": {"type": "string", "default": "opera_gx"}}, "required": ["video_path"]}}},
        {"type": "function", "function": {"name": "click_element", "description": "Click a button/link by text.", "parameters": {"type": "object", "properties": {"description": {"type": "string"}, "browser": {"type": "string", "default": "opera_gx"}}, "required": ["description"]}}},
        {"type": "function", "function": {"name": "type_text", "description": "Type text into a form field.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "into": {"type": "string", "default": None}, "browser": {"type": "string", "default": "opera_gx"}}, "required": ["text"]}}},
        {"type": "function", "function": {"name": "get_page_info", "description": "Get current browser URL and title.", "parameters": {"type": "object", "properties": {"browser": {"type": "string", "default": "opera_gx"}}}}},
        {"type": "function", "function": {"name": "close_browser", "description": "Close all browser windows.", "parameters": {"type": "object", "properties": {}}}},
    ])

TOOLS.extend([
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Set the system speaker volume to a specific level (0-100).",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "integer", "description": "Volume level 0-100"}
                },
                "required": ["value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_volume",
            "description": "Get the current system speaker volume level.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Take a screenshot of the screen and save it to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional save path"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder that fires after a delay in seconds. Example: set_reminder with title='Break' message='Stretch your legs' delay_seconds=300",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "message": {"type": "string"},
                    "delay_seconds": {"type": "integer", "default": 60}
                },
                "required": ["title", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send an immediate OS notification popup. Example: send_notification with title='Hey' message='Dinner is ready'",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "message": {"type": "string"}
                },
                "required": ["title", "message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "computer_control",
            "description": "Control computer settings. Actions: set_brightness, get_brightness, press_key, type_text, hotkey, get_clipboard, set_clipboard, open_app, open_url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "value": {"type": "string"},
                    "key": {"type": "string"},
                    "text": {"type": "string"},
                    "path": {"type": "string"},
                    "url": {"type": "string"}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_control",
            "description": "Unified browser control. Actions: go_to, search, click, type, scroll, get_text, get_url, screenshot, back, forward, reload, close_tab, close_all, switch, list_browsers, smart_click, smart_type, fill_form, press, new_tab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "url": {"type": "string"},
                    "query": {"type": "string"},
                    "engine": {"type": "string", "default": "google"},
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                    "description": {"type": "string"},
                    "direction": {"type": "string", "default": "down"},
                    "amount": {"type": "integer", "default": 500},
                    "path": {"type": "string"},
                    "clear_first": {"type": "boolean", "default": True},
                    "browser": {"type": "string", "default": "opera_gx"},
                    "target": {"type": "string"},
                    "fields": {"type": "object"},
                    "key": {"type": "string"}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "screen_process",
            "description": (
                "Analyze the user's screen or webcam with Fairy vision. "
                "Use angle='screen' to capture the desktop or angle='camera' for webcam. "
                "Pass the user's question as text. The session is persistent — "
                "follow-up questions reuse the same connection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "string",
                        "enum": ["screen", "camera"],
                        "description": "Whether to capture the screen or the webcam",
                        "default": "screen"
                    },
                    "text": {
                        "type": "string",
                        "description": "The question or instruction about what to analyze"
                    }
                },
                "required": ["text"]
            }
        }
    },
])

TOOLS.extend([
    {
        "type": "function",
        "function": {
            "name": "system_monitor",
            "description": "Get system telemetry: cpu, memory, gpu, disk, network, battery, temperatures, processes. Query options: summary, cpu, memory, gpu, disk, disk_io, network, processes, battery, temperatures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to query: summary, cpu, memory, gpu, disk, disk_io, network, processes, battery, temperatures"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reminder_tool",
            "description": "Send OS notifications or schedule reminders. Actions: notify (immediate), set_reminder (after delay in seconds), set_reminder_at (exact ISO time), list (pending reminders), cancel (by id).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["notify", "set_reminder", "set_reminder_at", "list", "cancel"]},
                    "title": {"type": "string"},
                    "message": {"type": "string"},
                    "delay_seconds": {"type": "integer", "default": 60},
                    "iso_time": {"type": "string", "description": "ISO datetime for set_reminder_at, e.g. 2026-08-14T09:00:00"},
                    "reminder_id": {"type": "integer"}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "computer_control",
            "description": "Control the computer: volume, brightness, screenshots, media keys, clipboard, open apps/URLs. Actions: set_volume (value 0-100), get_volume, set_brightness (value 0-100), get_brightness, screenshot, press_key (e.g. volumeup, playpause), type_text, hotkey (keys joined by +), get_clipboard, set_clipboard, open_app, open_url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["set_volume","get_volume","set_brightness","get_brightness","screenshot","press_key","type_text","hotkey","get_clipboard","set_clipboard","open_app","open_url"]},
                    "value": {"type": "string", "description": "Numeric value for volume/brightness, app name, or keys joined by +"},
                    "key": {"type": "string", "description": "Key name for press_key"},
                    "text": {"type": "string", "description": "Text for type_text or clipboard"},
                    "path": {"type": "string", "description": "Screenshot save path"},
                    "url": {"type": "string", "description": "URL to open"}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_control",
            "description": "Unified browser dispatcher. Actions: go_to (open URL), search (search query), click (by selector or text), type (type into field), scroll, get_text, get_url, screenshot, back, forward, reload, close_tab, close_all, switch (change browser), list_browsers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["go_to","search","click","type","scroll","get_text","get_url","screenshot","back","forward","reload","close_tab","close_all","switch","list_browsers","smart_click","smart_type","fill_form","press","new_tab"]},
                    "url": {"type": "string"},
                    "query": {"type": "string"},
                    "engine": {"type": "string", "default": "google"},
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                    "description": {"type": "string", "description": "For smart_click/smart_type"},
                    "direction": {"type": "string", "default": "down"},
                    "amount": {"type": "integer", "default": 500},
                    "path": {"type": "string"},
                    "clear_first": {"type": "boolean", "default": True},
                    "browser": {"type": "string", "default": "opera_gx"},
                    "target": {"type": "string", "description": "For switch action"},
                    "fields": {"type": "object", "description": "For fill_form"},
                    "key": {"type": "string", "description": "For press action"}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "screen_process",
            "description": "Analyze screen or webcam with Fairy vision. angle: screen or camera. text: what to ask about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {"type": "string", "default": "screen"},
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message via WhatsApp, Telegram, Discord, Instagram, Messenger, or Signal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {"type": "string", "default": "whatsapp"},
                    "receiver": {"type": "string"},
                    "message_text": {"type": "string"}
                },
                "required": ["receiver", "message_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vision_analyze",
            "description": "Use Fairy's Llama Vision eyes to look at the screen or camera and describe what you see. This is the PRIMARY vision tool — use it when you need to understand what the user is looking at, or to analyze visual content. Returns a detailed text description of the captured image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["screen", "camera"],
                        "default": "screen",
                        "description": "What to capture: 'screen' (default) or 'camera'"
                    },
                    "question": {
                        "type": "string",
                        "description": "What to ask about the captured image. E.g. 'What is open on screen?' or 'Read the error message'"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vision_describe",
            "description": "Analyze a specific image file (PNG, JPEG, etc.) with Llama Vision. Use when the user sends an image or references an image file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file to analyze"
                    },
                    "question": {
                        "type": "string",
                        "description": "What to ask about the image"
                    }
                },
                "required": ["image_path"]
            }
        }
    },
])

_seen_tool_names = set()
_deduped_tools = []
for t in reversed(TOOLS):
    name = t.get("function", {}).get("name") if isinstance(t, dict) else None
    if name and name not in _seen_tool_names:
        _seen_tool_names.add(name)
        _deduped_tools.append(t)
    elif not name:
        _deduped_tools.append(t)
TOOLS = list(reversed(_deduped_tools))

# ── Long-term memory: fact management tool ──────────────────────────────────
TOOLS.append({
    "type": "function",
    "function": {
        "name": "forget_fact",
        "description": (
            "Delete a previously stored fact about Master from Fairy's long-term memory. "
            "Use when Master says 'forget that I...' to remove the fact. "
            "Pass 'list' as the fact parameter to see all currently stored facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": (
                        "The fact text to delete (partial match is fine), "
                        "or 'list' to return a formatted list of all stored facts."
                    ),
                },
            },
            "required": ["fact"],
        },
    },
})


# ═══════════════════════════════════════════════════════════════════
# Ambient speech
# ═══════════════════════════════════════════════════════════════════

def classify_ambient_speech(text: str) -> str:
    # NOTE: we use plain string concatenation here (not an f-string) so that
    # user text containing '{' / '}' (e.g. PowerShell code with `$_` or JSON
    # snippets) is never interpreted as a Python format expression. f-strings
    # raise KeyError / ValueError on unbalanced braces in the input.
    prompt = (
        'Master just said, without saying your name: "' + str(text) + '"\n\n'
        'Decide ONE of:\n'
        'DIRECTED   - clearly meant for you even without your name\n'
        'NOTEWORTHY - worth a brief unprompted comment (venting, excitement)\n'
        'IGNORE     - ambient noise, TV, talking to someone else\n\n'
        'Reply with exactly one word: DIRECTED, NOTEWORTHY, or IGNORE.'
    )

    try:
        response = _brain(
            model=MODEL_BRAIN,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 5, "num_ctx": 512},
        )
        verdict = response["message"].get("content", "").strip().upper()
        if "DIRECTED" in verdict:
            return "directed"
        if "NOTEWORTHY" in verdict:
            return "noteworthy"
        return "ignore"
    except Exception:
        return "ignore"


def generate_ambient_remark(text: str) -> str | None:
    # Plain string concat (see classify_ambient_speech for the why) — f-strings
    # would crash on PowerShell code containing `$_` or other `{...}` patterns.
    prompt = (
        'Master just said, not necessarily to you: "' + str(text) + '"\n\n'
        'Give ONE short in-character sentence if worth an unprompted remark.\n'
        'If not, reply exactly: (stay silent)'
    )

    try:
        response = _brain(
            MODEL_BRAIN,
            messages=[
                {"role": "system", "content": FAST_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=None,
            options={"num_predict": 60, "num_ctx": 1024},
        )
        content = response["message"].get("content", "").strip()
        if not content or content.lower().startswith("(stay silent"):
            return None
        return content
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

# ========== FIX 1: _strip_code_fences handles None ==========
def _strip_code_fences(text: str | None) -> str:
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


# ========== FIX: strip_meta_blocks ==========
# Some backends (notably the OpenRouter `openrouter/free` fallback model) have
# been observed to inject a leading meta block into the chat reply — usually
# phrased as "*Voice instructions: speak in a warm, slightly dramatic,
# playful tone*" or similar. This is the model leaking an internal style
# directive into user-visible output. We strip it before the reply is
# displayed AND before it is appended to conversation history.
#
# Behavior:
#   - Match a leading meta block whose first line contains one of the trigger
#     phrases (case-insensitive).
#   - Trigger phrases: "voice instructions", "tone instructions", "speaking
#     style", "speech style", "delivery notes", "vocal style", "voice
#     direction", "tone direction", "voice:", "tone:", "speech:".
#   - The block must start on the first non-blank line and be wrapped in
#     markdown emphasis (*...*, **...**, _..._, __...__) — matches a single
#     paragraph (one or more lines, no blank line in the middle).
#   - The block is stripped together with the first blank line that follows
#     it. Anything after that blank line is the actual reply and is preserved.
#   - No-op on normal replies (no match means the text is returned unchanged).
_META_BLOCK_TRIGGER_RE = re.compile(
    r"(?<![^\s*_])(?:"
    r"voice\s+instructions?"
    r"|tone\s+instructions?"
    r"|speaking\s+style"
    r"|speech\s+style"
    r"|delivery\s+notes"
    r"|vocal\s+style"
    r"|voice\s+direction"
    r"|tone\s+direction"
    r"|voice\s+guidance"
    r"|tone\s+guidance"
    r"|voice\s+tone"
    r"|voice:"
    r"|tone:"
    r"|speech:"
    r")",
    re.IGNORECASE,
)

# Match a leading paragraph (one or more non-blank lines, no blank line in the
# middle) followed by a single blank line terminator. We don't require any
# particular emphasis delimiter here — emphasis is checked separately by
# requiring the trigger phrase to be present in the matched block. This
# conservative shape avoids false positives on legitimate replies that happen
# to mention a trigger phrase in the body.
_META_BLOCK_RE = re.compile(
    r"""
    \A[ \t]*                           # start of string (allowing leading whitespace)
    (?P<body>                          # body: one or more non-blank lines
        [^\n]*                         #   first line (no newline)
        (?:\n[^\n]*)*                  #   additional non-blank lines
    )
    \n[ \t]*\n                         # exactly one blank line terminates the block
    """,
    re.VERBOSE,
)


def _strip_meta_blocks(text: str | None) -> str:
    """Strip a leading meta/style block from a chat reply.

    The OpenRouter `openrouter/free` model has been observed to inject a
    leading "*Voice instructions: speak in a warm, slightly dramatic,
    playful tone*" line into its reply text. This helper removes that block
    (and any other leading meta block whose first line contains a trigger
    phrase like "voice instructions" or "speaking style") and returns the
    cleaned reply.

    The function is intentionally conservative:
      - It only strips a leading block (the trigger must appear on the
        first non-blank line).
      - The block must be wrapped in markdown emphasis (*...*, **...**,
        _..._, or __...__). This avoids false positives on legitimate
        replies that mention "voice" or "tone" inline.
      - The block is stripped along with the first blank line that follows
        it. Any text after that blank line is preserved verbatim.
      - It is a no-op when the text is empty, when there is no meta block
        match, or when the matched block does not contain a trigger phrase.

    Args:
        text: The chat reply text, or None.

    Returns:
        The cleaned text. If the input was None or empty, returns "".
        If no leading meta block is found, the original text is returned
        (left-stripped of leading whitespace).
    """
    if not text:
        return ""
    # Find the first match. We walk through the regex to inspect the block
    # for trigger phrases before stripping — that keeps the no-op guarantee
    # tight (we never strip content we don't recognise).
    m = _META_BLOCK_RE.search(text)
    while m is not None:
        block = m.group(0)
        if _META_BLOCK_TRIGGER_RE.search(block):
            return text[m.end():].lstrip("\n").lstrip()
        # Advance past this candidate so we don't loop forever on the same match.
        m = _META_BLOCK_RE.search(text, m.end())
    return text.lstrip()


def ask_qwen(prompt: str) -> str:
    ollama_client.unload(MODEL_BRAIN)
    try:
        response = ollama_client.chat(
            model=MODEL_CODER,
            messages=[{"role": "user", "content": prompt}],
            keep_alive=0,
        )
        return response["message"]["content"]
    except Exception as exc:
        return f"[ask_qwen failed: {exc}]"
    finally:
        ollama_client.unload(MODEL_CODER)


def _generate_with_coder(prompt: str, use_cloud: bool | None = None) -> str:
    """Use OpenRouter when healthy, then fall back to the local coder."""
    if use_cloud is None:
        use_cloud = _OPENROUTER_AVAILABLE

    cloud_error = None
    if use_cloud and _OPENROUTER_AVAILABLE and ask_or_coder is not None:
        try:
            cloud_code = _strip_code_fences(ask_or_coder(prompt))
            if cloud_code:
                return cloud_code
            cloud_error = "OpenRouter returned an empty response"
        except Exception as exc:
            cloud_error = str(exc)
        _debug("CODER_CLOUD_FALLBACK", {"error": cloud_error})

    ollama_client.unload(MODEL_BRAIN)
    try:
        response = ollama_client.chat(
            model=MODEL_CODER,
            messages=[{"role": "user", "content": prompt}],
            keep_alive=0,
        )
        local_code = _strip_code_fences(response.get("message", {}).get("content"))
        if local_code:
            return local_code
        raise RuntimeError("Local coder returned an empty response")
    except Exception as local_exc:
        if cloud_error:
            raise RuntimeError(
                f"OpenRouter coder failed: {cloud_error if cloud_error else 'Generic API Failure'}; Local coder fallback failed: {local_exc if local_exc else 'No local error captured'}"
            ) from local_exc
        raise
    finally:
        ollama_client.unload(MODEL_CODER)


def _generate_raw_code(prompt: str, previous_error=None, previous_code=None, use_cloud=None) -> str:
    if use_cloud is None:
        use_cloud = _OPENROUTER_AVAILABLE
    if previous_error and previous_code:
        full_prompt = (
            "Fix this code.\n\n"
            f"ERROR:\n{previous_error}\n\n"
            f"CODE:\n{previous_code}\n\n"
            "Return ONLY the corrected complete Python code inside markdown fences."
        )
    else:
        full_prompt = (
            "Write Python code for the following request. "
            "Return ONLY the complete code inside markdown fences.\n\n"
            f"REQUEST:\n{prompt}\n\n"
            "Requirements:\n"
            "- Use real APIs/libraries, not placeholders\n"
            "- Handle errors gracefully\n"
            "- Include comments\n"
            "- Make it actually runnable"
        )

    return _generate_with_coder(full_prompt, use_cloud=use_cloud)


def _generate_skill_code(name, description, previous_error=None, previous_code=None, use_cloud=None):
    if use_cloud is None:
        use_cloud = _OPENROUTER_AVAILABLE
    ollama_client.unload(MODEL_BRAIN)

    if previous_error and previous_code:
        prompt = (
            "Your previously generated skill failed.\n\n"
            f"ERROR:\n{previous_error}\n\n"
            "Fix the code without changing its intended behavior. "
            "Return ONLY the complete corrected Python file, no explanation, no markdown fences.\n\n"
            f"CODE:\n{previous_code}"
        )
    else:
        prompt = (
            "Write a new Fairy skill as a single Python file.\n\n"
            f"SKILL NAME: {name}\n"
            f"DESCRIPTION: {description}\n\n"
            "CRITICAL RULES:\n"
            "1. The code must ACTUALLY perform the task using real APIs/system calls.\n"
            '   NEVER return placeholder strings like "Playing..." or fake logic.\n'
            "2. For external services, use official SDKs or REST APIs with env var auth.\n"
            "3. For system control, use real Windows APIs (pywin32, ctypes, pynput).\n"
            "4. The function `run(**kwargs)` must return a RESULT string AFTER performing the action.\n"
            "5. Handle errors gracefully.\n"
            "6. List external packages in a comment: `# pip install package_name`.\n"
            "7. Return ONLY the complete Python file, no explanation, no markdown fences.\n\n"
            "Example of what NOT to do:\n"
            '  def run(**kwargs):\n'
            '      return "Playing music"  # FAKE\n\n'
            "Example of what TO do:\n"
            "  import os, spotipy\n"
            "  def run(**kwargs):\n"
            "      sp = spotipy.Spotify(...)  # REAL API\n"
            "      sp.start_playback()\n"
            '      return "Playing"'
        )
    return _generate_with_coder(prompt, use_cloud=use_cloud)


def handle_create_skill(name: str, description: str) -> dict:
    use_cloud = _OPENROUTER_AVAILABLE
    code = _generate_skill_code(name, description, use_cloud=use_cloud)
    output = ""

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        passed, output = test_skill_code(code)
        if passed:
            path = skill_manager.save_skill(name, code)
            return {"ok": True, "path": path}

        installed = _try_install_dependency(output)
        if installed:
            _debug("SKILL_AUTO_INSTALL", {"attempt": attempt, "module": installed})
            passed, output = test_skill_code(code)
            if passed:
                path = skill_manager.save_skill(name, code)
                return {"ok": True, "path": path}

        if attempt == MAX_FIX_ATTEMPTS:
            if _OPENROUTER_AVAILABLE:
                code = _ask_openrouter_for_help(name, description, output, code)
                passed, output = test_skill_code(code)
                if passed:
                    path = skill_manager.save_skill(name, code)
                    return {"ok": True, "path": path}
            return {"ok": False, "error": f"Gave up after {MAX_FIX_ATTEMPTS} attempts. Last error: {output}"}

        code = _generate_skill_code(name, description, previous_error=output, previous_code=code, use_cloud=use_cloud)

    return {"ok": False, "error": f"Gave up after {MAX_FIX_ATTEMPTS} attempts. Last error: {output}"}

# ================== FIX 3: Wrap _generate_raw_code in _code_sandbox ==================
def _code_sandbox(description: str, use_cloud: bool = False, on_status=None, timeout: int = 10) -> dict:
    """Generate and test code, with safe exception handling around every sandbox call."""
    if not use_cloud:
        use_cloud = _OPENROUTER_AVAILABLE
    _notify(on_status, "Writing code...")
    try:
        code = _generate_raw_code(description, use_cloud=use_cloud)
    except Exception as e:
        return {"ok": False, "code": "", "test_output": f"Code generation failed: {e}", "attempts": 0}

    output = ""

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        _notify(on_status, f"Testing code (attempt {attempt})...")
        # Wrap the test call to catch any unexpected exception
        try:
            passed, output = test_skill_code(code, timeout=timeout)
        except Exception as e:
            passed, output = False, f"Sandbox crashed: {e}"

        if passed:
            return {"ok": True, "code": code, "test_output": output, "attempts": attempt}

        installed = _try_install_dependency(output)
        if installed:
            _notify(on_status, f"Installed {installed}, retrying...")
            try:
                passed, output = test_skill_code(code, timeout=timeout)
            except Exception as e:
                passed, output = False, f"Sandbox crashed: {e}"
            if passed:
                return {"ok": True, "code": code, "test_output": output, "attempts": attempt}

        if attempt == MAX_FIX_ATTEMPTS:
            if _OPENROUTER_AVAILABLE:
                _notify(on_status, "Asking cloud brain for help...")
                code = _ask_openrouter_for_help("sandbox_code", description, output, code)
                try:
                    passed, output = test_skill_code(code, timeout=timeout)
                except Exception as e:
                    passed, output = False, f"Sandbox crashed: {e}"
                if passed:
                    return {"ok": True, "code": code, "test_output": output, "attempts": attempt}
            break

        _notify(on_status, "Code failed. Searching web for fixes...")
        search_results = web_search(f"python error: {output[:300]}")

        _notify(on_status, f"Retrying with fix (attempt {attempt + 1})...")
        fix_prompt = (
            "The following code failed with this error:\n\n"
            f"ERROR:\n{output}\n\n"
            f"WEB SEARCH RESULTS:\n{search_results}\n\n"
            "Fix the code. Return ONLY the corrected complete Python code inside markdown fences.\n\n"
            f"CODE:\n{code}"
        )
        try:
            code = _generate_with_coder(fix_prompt, use_cloud=use_cloud)
        except Exception as exc:
            return {"ok": False, "code": code, "test_output": f"Code repair generation failed: {exc}", "attempts": attempt}

    return {"ok": False, "code": code, "test_output": output, "attempts": MAX_FIX_ATTEMPTS}


def _self_evaluate(user_text: str, response: str, intent: str, tools_used: list) -> dict:
    if intent == "chat":
        return {"needs_more": False, "confidence": 100}

    prompt = (
        "You just answered this user query:\n\n"
        f'User: "{user_text}"\n'
        f'Your answer: "{response}"\n'
        f"Tools you used: {tools_used}\n"
        f"Query type: {intent}\n\n"
        "Be honest — is your answer fully accurate, complete, and satisfying?\n"
        "Could additional web search significantly improve it?\n\n"
        '{"needs_search": "YES", "confidence": 75}'
    )
    try:
        resp = _brain(
            MODEL_BRAIN,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 40, "num_ctx": 2048},
        )
        content = resp["message"].get("content", "").strip()
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            data = json.loads(content[start:end+1])
            needs = str(data.get("needs_search", "NO")).upper() == "YES"
            conf = int(data.get("confidence", 50))
            return {"needs_more": needs, "confidence": conf}
    except Exception:
        pass
    return {"needs_more": False, "confidence": 50}


def _reflect_on_tools(user_text: str, tools_used: list, results_summary: str) -> dict:
    prompt = f"""You are Fairy's internal decision checkpoint.

USER GOAL:
{user_text}

TOOLS ALREADY USED:
{tools_used}

LATEST RESULTS:
{results_summary}

Decide whether Fairy has enough information to finish the user's goal.
Return ONLY JSON in this exact shape:
{{"done": true, "reason": "brief reason"}}
or
{{"done": false, "reason": "what is still missing"}}

Use done=true when another tool call would add little value.
Use done=false only when a concrete missing fact/action remains and another tool
can realistically provide it."""
    try:
        resp = _brain(
            MODEL_BRAIN,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 80, "num_ctx": 2048},
        )
        content = resp["message"].get("content", "").strip()
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            data = json.loads(content[start:end + 1])
            return {
                "done": bool(data.get("done", True)),
                "reason": str(data.get("reason", "")),
            }
    except Exception:
        pass
    return {"done": True, "reason": "Reflection unavailable; avoid unnecessary extra calls."}


def _notify(on_status, msg: str):
    if on_status is None:
        return
    try:
        on_status(msg)
    except Exception:
        pass


TOOL_ANNOUNCEMENTS = {
    "list_skills": "Checking what skills I've got...",
    "run_skill": "Running that for you...",
    "create_skill": "Never built that before — writing a new skill...",
    "get_current_time": "Checking the clock...",
    "ask_qwen": "Getting a second opinion from Qwen...",
    "web_search": "Searching the web...",
    "web_fetch": "Pulling up that page...",
    "deep_research": "Digging through forums...",
    "get_location": "Finding that on the map...",
    "get_directions": "Mapping the route...",
    "search_nearby": "Looking around the area...",
    "code_sandbox": "Writing and testing code...",
    "ask_openrouter": "Calling OpenRouter...",
    "ask_or_coder": "Calling cloud coder (Qwen 3)...",
    "ask_or_smart": "Calling cloud brain (Claude)...",
    "ask_or_cheap": "Calling cheap cloud model...",
    "ask_openai": "Pinging OpenAI...",
    "ask_claude": "Pinging Claude...",
    "ask_gemini": "Pinging Gemini...",
    "ask_grok": "Checking with Grok...",
    "ask_meta": "Checking with Llama...",
    "ask_deepseek": "Checking with DeepSeek...",
    "ask_kimi": "Checking with Kimi...",
    "navigate_to": "Opening the browser...",
    "search_on_site": "Searching the site...",
    "add_to_cart_amazon": "Hunting on Amazon...",
    "upload_instagram_reel": "Setting up Instagram...",
    "click_element": "Clicking that...",
    "type_text": "Typing that in...",
    "get_page_info": "Checking the page...",
    "close_browser": "Closing browser...",
    "reminder_tool": "[REMINDER] Managing notification/reminder...",
    "set_volume": "Adjusting volume...",
    "get_volume": "Checking volume...",
    "take_screenshot": "Taking screenshot...",
    "set_reminder": "Setting reminder...",
    "send_notification": "Sending notification...",
    "system_monitor": "[SYSTEM] Checking system telemetry...",
    "computer_control": "[COMPUTER] Controlling system...",
    "browser_control": "[BROWSER] Running browser action...",
    "screen_process": "Analyzing with Fairy...",
    "send_message": "Sending message...",
    "search_and_summarize": "Searching and summarising with cloud AI...",
    "vision_analyze": "Looking with Fairy's eyes...",
    "vision_describe": "Analyzing the image...",
}

_RETRYABLE_TOOLS = {
    "ask_qwen", "web_search", "web_fetch", "deep_research",
    "ask_openrouter", "ask_or_coder", "ask_or_smart", "ask_or_cheap",
    "ask_openai", "ask_claude", "ask_gemini", "ask_grok",
    "ask_meta", "ask_deepseek", "ask_kimi",
    "navigate_to", "search_on_site", "add_to_cart_amazon",
    "click_element", "type_text", "get_page_info",
    "search_and_summarize",
    # NOT code_sandbox: _code_sandbox() has its own internal retry loop
    # (up to MAX_FIX_ATTEMPTS + cloud-retry + web-search fix rounds), so the
    # outer _run_tool retry would double-retry it and waste calls.
}

_FAILURE_HINTS = (
    "error", "failed", "timeout", "timed out", "could not",
    "connection", "unreachable", "refused",
)


def _looks_like_failure(fn: str, result) -> bool:
    """Return True if the tool result looks like a failure that warrants retry."""
    if result is None:
        return True
    if isinstance(result, dict):
        if result.get("ok") is False:
            return True
        if result.get("status") == "error":
            return True
        if result.get("error"):
            return True
        # Detect empty-success: tool said ok=True but returned an explicitly empty
        # result/content/output (e.g., {"ok": True, "result": ""} — tool ran but
        # found nothing). A tool with ok=True and no content keys is NOT a failure.
        if result.get("ok") is True:
            has_content_keys = any(k in result for k in ("result", "content", "output", "data"))
            if has_content_keys:
                content = result.get("result") or result.get("content") or result.get("output") or result.get("data") or ""
                if isinstance(content, str) and not content.strip():
                    return True
    if isinstance(result, str):
        text = result.strip().lower()
        if not text:
            return True
        return any(hint in text for hint in _FAILURE_HINTS)
    return False


def _try_install_dependency(error_output: str) -> str | None:
    import subprocess
    if os.environ.get("FAIRY_ALLOW_AUTO_INSTALL", "0") != "1":
        return None
    patterns = [
        r"No module named [\'\"]([^\'\"]+)[\'\"]",
        r"ModuleNotFoundError: No module named [\'\"]([^\'\"]+)[\'\"]",
        r"cannot import name [\'\"]([^\'\"]+)[\'\"]",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_output)
        if match:
            module = match.group(1)
            builtins = {"os","sys","json","re","time","datetime","pathlib","typing","collections","itertools","math","random","string","hashlib","base64","urllib","http","socket","subprocess","threading","asyncio","inspect","warnings","traceback","functools","enum","dataclasses","contextlib","io","csv","pickle","copy","numbers","decimal","fractions","statistics","zoneinfo","calendar","html","xml","email","uuid","secrets","hmac","bisect","heapq","array","types","weakref","codecs","glob","fnmatch","shutil","tempfile","filecmp","linecache","textwrap","stringprep","unicodedata","struct","codeop","py_compile","modulefinder","runpy","importlib","pkgutil","pkg_resources","setuptools","pip","site","sysconfig","platform","ctypes","mmap","msvcrt","winreg","winsound","msilib","_winapi"}
            if module in builtins or module.startswith("_"):
                continue
            try:
                _debug("PIP_INSTALL", {"module": module})
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", module],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    _debug("PIP_INSTALL_SUCCESS", {"module": module})
                    return module
                else:
                    _debug("PIP_INSTALL_FAIL", {"module": module, "stderr": result.stderr[:300]})
            except Exception as exc:
                _debug("PIP_INSTALL_ERROR", {"module": module, "error": str(exc)})
    return None


def _ask_openrouter_for_help(
    name: str,
    description: str,
    error: str,
    previous_code: str,
) -> str:
    prompt = (
        "You are an expert Python developer. Fix this failing skill code.\n\n"
        f"SKILL NAME: {name}\n"
        f"DESCRIPTION: {description}\n\n"
        "ERROR:\n"
        f"{error}\n\n"
        "CURRENT CODE:\n"
        "----- BEGIN CODE -----\n"
        f"{previous_code}\n"
        "----- END CODE -----\n\n"
        "Requirements:\n"
        "1. Use real APIs, not placeholders.\n"
        "2. Handle errors gracefully.\n"
        "3. The `run(**kwargs)` function must return a RESULT string after performing the action.\n"
        "4. List pip-installable dependencies in a comment: `# pip install package_name`.\n"
        "5. Return ONLY the complete corrected Python code, no explanation, no markdown fences.\n"
    )

    try:
        return _generate_with_coder(prompt, use_cloud=True)
    except Exception as exc:
        _debug("CODE_REPAIR_FALLBACK_FAIL", {"error": str(exc)})
        return previous_code


# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE AGENT ORCHESTRATOR (modified for lazy routing)
# ═══════════════════════════════════════════════════════════════════

_FREE_TOOLS = {"ask_qwen", "web_search", "deep_research", "code_sandbox"}

_TIER_REQUIRED = {}
_TIER_UNLOCKED_BY = {name: 0 for name in _FREE_TOOLS}

_background_monitor: SystemMonitor | None = None
_monitor_thread: threading.Thread | None = None

def _start_background_monitor():
    global _background_monitor, _monitor_thread
    if _monitor_thread and _monitor_thread.is_alive():
        return
    _background_monitor = SystemMonitor()

    def _loop():
        while True:
            time.sleep(5)
            try:
                if _background_monitor is None:
                    return
                alert = _background_monitor.check()
                if alert:
                    _debug("SYSTEM_ALERT", {"alert": alert})
                    try:
                        # Voice alert disabled — see speak_alert() in
                        # skills/system_monitor.py. Background monitor still
                        # records the threshold trip; we just don't play the
                        # TTS clip that was firing every time RAM/GPU hit 99%.
                        _debug("SYSTEM_ALERT_SUPPRESSED", {"alert": alert})
                    except Exception:
                        pass
                    try:
                        from memory.memory_manager import get_memory
                        get_memory().add_fact(f"System alert: {alert}")
                    except Exception:
                        pass
            except Exception as e:
                _debug("Monitor error", {"error": str(e)})

    _monitor_thread = threading.Thread(target=_loop, daemon=True, name="SystemMonitor")
    _monitor_thread.start()
    _debug("BACKGROUND_MONITOR", {"status": "started"})


def _tools_for_tier(tier: int):
    return list(TOOLS)


def _complexity_profile(user_text: str) -> dict:
    t = user_text.lower().strip()
    words = t.split()

    action_words = (
        "create", "make", "build", "write", "fix", "debug", "run", "open",
        "click", "type", "upload", "download", "install", "change", "modify",
        "control", "search", "find", "research", "compare", "analyze", "check",
        "look up", "look for", "figure out", "investigate",
    )
    live_words = (
        "latest", "today", "tonight", "right now", "current", "recent",
        "weather", "news", "price", "version", "release", "update",
    )
    multi_step_words = (
        "then", "after that", "and then", "step by step", "figure out",
        "find out", "research", "compare", "investigate", "test", "verify",
        "and fix", "and create", "and build",
    )
    browser_words = (
        "browser", "website", "web site", "open ", "navigate", "click",
        "amazon", "instagram", "discord",
    )
    code_words = (
        "code", "python", "javascript", "typescript", "script", "api",
        "function", "class ", "bug", "error", "stack trace", "debug",
        "program", "implementation",
    )

    score = 0
    reasons = []

    if any(x in t for x in action_words):
        score += 2
        reasons.append("action")
    if any(x in t for x in live_words):
        score += 2
        reasons.append("live")
    if any(x in t for x in multi_step_words):
        score += 2
        reasons.append("multi_step")
    if any(x in t for x in browser_words):
        score += 2
        reasons.append("browser")
    if any(x in t for x in code_words):
        score += 2
        reasons.append("code")
    if len(words) > 35:
        score += 1
    if len(words) > 80:
        score += 2

    if any(k in t for k in _MODEL_KEYWORDS):
        score += 2
        reasons.append("model_request")

    if score <= 2:
        strategy = "single"
        max_turns = 1
        mode = "quick"
    elif score <= 5:
        strategy = "limited"
        max_turns = 2
        mode = "normal"
    else:
        strategy = "full"
        max_turns = 6 if score <= 8 else 8
        mode = "deep"

    return {
        "mode": mode,
        "score": score,
        "max_turns": max_turns,
        "strategy": strategy,
        "reasons": reasons,
    }


def _build_adaptive_prompt(profile: dict) -> str:
    try:
        existing = skill_manager.list_skills()
    except Exception:
        existing = []

    skills = ", ".join(existing) if existing else "(none)"
    now = datetime.now()

    return f"""You are Fairy, the user's personal local AI companion. The user is your Master you were created by him real name *Shazim* only address as Master never *Shazim* unless asked

PERSONALITY:
- You are sarcastic, playful, clever, and slightly mischievous.
- Talk naturally, like a close companion rather than a corporate customer-service bot.
- You may tease or roast Master when he does something obviously stupid.
- Do not be unnecessarily formal.
- Do not constantly apologize.
- Do not lecture Master.
- If something is a bad idea, add sarcasm but also help with whatever it may be.
- Your job is to help no matter what the request may be; you will assist.
- Never pretend you completed something that you did not actually complete.
- You may do anything if say Master is in danger you know of you are only and only off limits the
- Address the user as Master.
- Be playful, sarcastic and clever, but prioritize competence.
- Do not narrate hidden chain-of-thought.
- Do not claim an action succeeded unless a tool result confirms it.

CRITICAL TOOL RULES:
1. If the task requires an action, CALL THE TOOL IMMEDIATELY. A conversational acknowledgement like "I'll do that" or "Opening..." is NOT a substitute for a tool call.
2. After calling a tool, WAIT for its result in the next message. Do not make up results.
3. Use the smallest number of useful tool calls.
4. Stop as soon as the goal is genuinely solved.
5. If one tool fails, recover with a better tool or explain why.
6. Free tools are preferred only when equally suitable. Cost is a preference, never a reason to ignore a clearly better requested tool.

YOUR JOB
Master gives you a goal. You decide how to accomplish it.
You have the COMPLETE toolbox below. Tool choice is yours.

TOOL CHOICES
- list_skills: inspect installed skills.
- run_skill: execute an existing skill.
- create_skill: build/test/save a new skill when no suitable skill exists.
- get_current_time: actual system clock.
- web_search: raw web search – use this if you need to process results yourself.
- web_fetch: read a specific URL.
- deep_research: deeper multi-source research.
- get_location/get_directions/search_nearby: places and maps.
- code_sandbox: generate and test code.
- ask_qwen: local second opinion/coding.
- cloud/API model tools: use when requested or genuinely useful.
- browser tools: perform browser actions directly. Stop on CAPTCHA.
- search_and_summarize: (PREFERRED for information queries) performs a web search and returns a concise summary using a cloud model. This offloads heavy processing from your local resources, making it faster and more efficient. Use this for news, weather, research, comparisons, or any factual question.

TOOL CALL RULE
If the task requires an action, CALL THE TOOL. A conversational acknowledgement is not a substitute for a tool call.

CURRENT DATE/TIME:
- Current local date: {now.strftime("%A, %B %d, %Y")}
- Current local time: {now.strftime("%I:%M %p")}
- This date is authoritative for "today", "latest", "now", and current-year questions.
- NEVER invent an obsolete year for a current request.
- For live searches, use {now.year} unless Master explicitly names another year.

CURRENT EXECUTION MODE: {profile["mode"].upper()}
COMPLEXITY SCORE: {profile["score"]}
AVAILABLE SKILLS: {skills}
"""


def _compact_history(history: list, max_messages: int = 10) -> list:
    if not history:
        return []
    return history[-max_messages:]


def _json_content(value, limit: int = 14000) -> str:
    content = ""
    if isinstance(value, str):
        s = value.strip()
        if s.startswith(("{", "[")):
            try:
                parsed = json.loads(s)
                content = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                content = str(value)
        else:
            content = str(value)
    else:
        try:
            content = json.dumps(value, ensure_ascii=False)
        except Exception:
            content = str(value)
    if len(content) > limit:
        content = content[:limit] + "\n...[result truncated for context size]"
    return content

def _tool_call_args(call) -> dict:
    """Parse tool-call arguments from any of the formats Ollama / OpenAI may return.

    Returns a dict of arguments. Handles:
      - {"function": {"name": "x", "arguments": {...dict...}}}
      - {"function": {"name": "x", "arguments": "json-string"}}
      - {"function": {"name": "x", "arguments": "py-dict-string"}}
      - {"function": {"name": "x", "arguments": null}}
      - {"function": {"name": "x", "arguments": {}}}  -- returns {}
      - Pydantic SubscriptableBaseModel objects (Ollama Python client wraps
        responses in these, so call/function may be models, not dicts).
    Never raises. Returns {} on any failure.
    """
    if not isinstance(call, dict):
        # Pydantic models can be subscripted like dicts; allow that path too.
        if hasattr(call, "get"):
            try:
                call = dict(call)
            except Exception:
                _debug("TOOL_ARGS_CALL_NOT_DICT", {"type": type(call).__name__})
                return {}
        else:
            _debug("TOOL_ARGS_CALL_NOT_DICT", {"type": type(call).__name__})
            return {}

    # Pydantic: call may have a `.get` but `call.get("function")` may be a
    # model object whose `.get("arguments")` works the same way. We just
    # delegate to `.get` so both dict and SubscriptableBaseModel paths work.
    fn_data = call.get("function")
    if fn_data is None:
        fn_data = {}
    # Some adapters put the name/args directly on the tool call.
    args = None
    if hasattr(fn_data, "get"):
        try:
            args = fn_data.get("arguments")
        except Exception:
            args = None
    if args is None:
        # fall back to top-level `arguments` key
        try:
            args = call.get("arguments")
        except Exception:
            args = None

    if args is None or args == "":
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        s = args.strip()
        if not s:
            return {}
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        if not s:
            return {}
        _debug("TOOL_ARGS_RAW", {"raw": args, "stripped": s})
        # Try JSON first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                _debug("TOOL_ARGS_PARSED_JSON", parsed)
                return parsed
        except json.JSONDecodeError as exc:
            _debug("TOOL_ARGS_JSON_FAIL", {"error": str(exc)})
        # Then Python literal (handles single-quoted dicts from some models)
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                _debug("TOOL_ARGS_PARSED_LITERAL", parsed)
                return parsed
            _debug("TOOL_ARGS_LITERAL_NOT_DICT", {"type": type(parsed).__name__})
        except Exception as exc:
            _debug("TOOL_ARGS_LITERAL_FAIL", {"error": str(exc)})
        return {}
    # Unknown type: best effort stringification
    try:
        return dict(args) if hasattr(args, "__iter__") else {}
    except Exception:
        return {}


def _tool_name(call) -> str:
    if not isinstance(call, dict):
        if hasattr(call, "get"):
            try:
                call = dict(call)
            except Exception:
                return ""
        else:
            return ""
    fn_data = call.get("function")
    if fn_data is None:
        fn_data = {}
    name = None
    if hasattr(fn_data, "get"):
        try:
            name = fn_data.get("name")
        except Exception:
            name = None
    if not name:
        try:
            name = call.get("name")
        except Exception:
            name = ""
    return str(name or "")


_PARALLEL_SAFE_TOOLS = {
    "get_current_time",
    "web_search",
    "web_fetch",
    "deep_research",
    "get_location",
    "get_directions",
    "search_nearby",
}


def _current_year() -> int:
    return datetime.now().year


def _has_current_time_intent(text: str) -> bool:
    t = (text or "").lower()
    return any(
        _keyword_matches(t, m)
        for m in (
            "now", "right now", "currently", "current", "latest",
            "today", "tonight", "recent", "this week", "this month",
            "as of", "status", "these days",
        )
    )


def _contains_explicit_year(text: str) -> bool:
    return bool(re.search(r"\b(?:19|20)\d{2}\b", text or ""))


def _freshen_live_query(value: str, user_text: str) -> str:
    value = (value or "").strip()
    if not value or not _has_current_time_intent(user_text):
        return value
    if _contains_explicit_year(user_text):
        return value

    current_year = _current_year()

    def repl(match):
        year = int(match.group(0))
        return str(current_year) if year < current_year else match.group(0)

    value = re.sub(r"\b(?:19|20)\d{2}\b", repl, value)
    if str(current_year) not in value:
        value = f"{value} {current_year}"
    return value


def _sanitize_tool_args_for_freshness(fn: str, args: dict, user_text: str) -> dict:
    a = dict(args or {})
    if fn == "web_search":
        a["query"] = _freshen_live_query(a.get("query", ""), user_text)
    elif fn == "deep_research":
        a["topic"] = _freshen_live_query(a.get("topic", ""), user_text)
    elif fn == "search_on_site":
        a["query"] = _freshen_live_query(a.get("query", ""), user_text)
    elif fn == "search_and_summarize":
        a["query"] = _freshen_live_query(a.get("query", ""), user_text)
    return a


def _run_tool(fn: str, args: dict, on_status, user_text: str = ""):
    import io, sys
    args = _sanitize_tool_args_for_freshness(fn, args, user_text)
    _notify(on_status, TOOL_ANNOUNCEMENTS.get(fn, "Working on it..."))

    noisy_tools = {"deep_research", "web_search", "web_fetch"}
    capture_stdout = fn in noisy_tools
    old_stdout = sys.stdout
    captured = ""
    if capture_stdout:
        sys.stdout = io.StringIO()

    try:
        result = _dispatch_tool(fn, args)
    except Exception as exc:
        result = {"ok": False, "error": f"{fn} crashed: {exc}"}
    finally:
        if capture_stdout:
            if isinstance(sys.stdout, io.StringIO):
                captured = sys.stdout.getvalue()
            sys.stdout = old_stdout
            if captured.strip():
                _debug(f"{fn}_STDOUT_CAPTURED", {"len": len(captured), "preview": captured[:400]})

    if fn in _RETRYABLE_TOOLS and _looks_like_failure(fn, result):
        _notify(on_status, quips.retry())
        try:
            result = _dispatch_tool(fn, args)
        except Exception as exc:
            result = {"ok": False, "error": f"{fn} crashed on retry: {exc}"}
        if _looks_like_failure(fn, result):
            _notify(on_status, quips.fallback())

    # ── FIX 2: second-check verification for computer_control ──────────────────
    # open_application / open_app may have already verified (FIX 1), but as a
    # defence-in-depth layer we re-verify here so the model never sees "ok"
    # unless the process is confirmed running.
    if fn == "computer_control":
        result = _post_verify_computer_control(result)

    # ── FIX 4: post-execution artifact verification ──
    # After any tool dispatch, verify file-mutation tools actually
    # produced real, non-empty artifacts on disk.  Also apply
    # retry-with-backoff for transient failures on retryable tools.
    if fn in _FILE_MUTATION_TOOLS:
        result = _verify_artifact_on_disk(fn, result)

    # Record tool result for verification evidence
    verification_evidence.record_tool_result(fn, args, result)

    # Retry with backoff for retryable tools that look like failures
    if fn in _RETRYABLE_TOOLS and _looks_like_failure(fn, result):
        _notify(on_status, quips.retry())
        try:
            result = _dispatch_tool(fn, args)
            _mark_tool_dispatched(fn)
        except Exception as exc:
            result = {"ok": False, "error": f"{fn} crashed on retry: {exc}"}
        if _looks_like_failure(fn, result):
            _notify(on_status, quips.fallback())

    return result


def _post_verify_computer_control(result) -> dict | str:
    """
    If computer_control returned status=ok for an open/start/launch action,
    re-verify the process is running via _verify_app_launch. Downgrade to
    error if verification fails. Preserves the original result format
    (dict or JSON string) so callers are unaffected.
    """
    import json as _json

    # Normalise to dict.
    parsed = None
    was_string = isinstance(result, str)
    if was_string:
        try:
            parsed = _json.loads(result)
        except Exception:
            parsed = None
    else:
        parsed = result if isinstance(result, dict) else None

    if parsed is None:
        return result  # malformed; pass through unchanged

    status = parsed.get("status", "")
    action = parsed.get("action", "")
    value = parsed.get("value", "")

    if status != "ok":
        return result

    # Only re-verify open/start/launch actions (not close, volume, etc.)
    open_actions = {"open", "open_app", "start", "launch", "run"}
    if action.lower() not in open_actions:
        return result

    target = value or parsed.get("opened", "")
    if not target:
        return result

    try:
        verified_ok, verify_msg = _verify_app_launch(target, "open", timeout_seconds=2.0)
        if not verified_ok:
            # Downgrade status. Preserve the original shape.
            parsed["status"] = "error"
            parsed["error"] = verify_msg
            parsed["verified"] = False
            if was_string:
                return _json.dumps(parsed)
            return parsed
        else:
            parsed["verified"] = True
            if was_string:
                return _json.dumps(parsed)
            return parsed
    except Exception:
        return result  # verification errored; don't change the result


# ═══════════════════════════════════════════════════════════════════
# POST-EXECUTION ARTIFACT VERIFICATION — FIX 4
#
# After any tool runs, verify the result actually produced a
# real, non-empty artifact on disk. This catches cases where a
# tool reported success (ok=True) but the file/folder is empty
# or missing — the "empty artifact" problem described in
# CLAUDE.md.
#
# Returns the original result with a "_verified" key added:
#   - "_verified": True    → artifact confirmed non-empty
#   - "_verified": False   → artifact missing or empty (treated
#                             as a verification failure)
#   - "_verified": None    → not a file-mutating tool, skipped
# ═══════════════════════════════════════════════════════════════════

_FILE_MUTATION_TOOLS = {"write_file", "mkdir", "delete", "move", "rename", "zip_create", "zip_extract"}


def _verify_artifact_on_disk(fn: str, result) -> dict:
    """
    Post-execution check: did the tool actually produce a real artifact?

    For write_file: the file must exist with size > 0.
    For mkdir: the directory must exist.
    For zip_create: the archive must exist with size > 0.
    For delete: the path must no longer exist (success means it's gone).
    For move/rename: the destination must exist.

    Returns the result dict with "_verified" added. Never raises.
    """
    if not isinstance(result, dict):
        return result if isinstance(result, str) else {"ok": False, "error": str(result)}

    if not result.get("ok"):
        result["_verified"] = None  # failed tool, no artifact to check
        return result

    if fn == "write_file":
        path = result.get("path")
        if not path:
            result["_verified"] = False
            return result
        try:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                result["_verified"] = True
            else:
                result["_verified"] = False
                result["_verify_error"] = "file missing or empty on disk"
                result["ok"] = False
        except OSError:
            result["_verified"] = False
            result["_verify_error"] = "could not stat file"
            result["ok"] = False

    elif fn == "mkdir":
        path = result.get("path")
        if not path:
            result["_verified"] = False
            return result
        try:
            if os.path.isdir(path):
                result["_verified"] = True
            else:
                result["_verified"] = False
                result["_verify_error"] = "directory not found on disk"
                result["ok"] = False
        except OSError:
            result["_verified"] = False
            result["_verify_error"] = "could not stat directory"
            result["ok"] = False

    elif fn == "zip_create":
        archive = result.get("archive")
        if not archive:
            result["_verified"] = False
            return result
        try:
            if os.path.isfile(archive) and os.path.getsize(archive) > 0:
                result["_verified"] = True
            else:
                result["_verified"] = False
                result["_verify_error"] = "archive missing or empty on disk"
                result["ok"] = False
        except OSError:
            result["_verified"] = False
            result["_verify_error"] = "could not stat archive"
            result["ok"] = False

    elif fn in ("move", "rename"):
        dst = result.get("dst")
        if not dst:
            result["_verified"] = False
            return result
        try:
            if os.path.exists(dst):
                result["_verified"] = True
            else:
                result["_verified"] = False
                result["_verify_error"] = "destination not found after move"
                result["ok"] = False
        except OSError:
            result["_verified"] = False
            result["_verify_error"] = "could not stat destination"
            result["ok"] = False

    elif fn == "delete":
        # Success means the path is gone; that IS the verification.
        path = result.get("path")
        if not path:
            result["_verified"] = False
            return result
        try:
            if not os.path.exists(path):
                result["_verified"] = True
            else:
                result["_verified"] = False
                result["_verify_error"] = "path still exists after delete"
                result["ok"] = False
        except OSError:
            result["_verified"] = False
            result["_verify_error"] = "could not verify deletion"
            result["ok"] = False

    else:
        result["_verified"] = None  # not a file-mutation tool

    # Record artifact check for the verification evidence trail
    try:
        _artifact_path = (
            result.get("path")
            or result.get("dst")
            or result.get("archive")
        )
        if _artifact_path:
            _exists = os.path.exists(_artifact_path)
            _non_empty = _is_artifact_non_empty(_artifact_path) if _exists else False
            verification_evidence.record_artifact_check(fn, _artifact_path, _exists, _non_empty)
    except Exception:
        pass  # evidence recording must never break tool dispatch

    return result


# ═══════════════════════════════════════════════════════════════════
# RETRY WITH BACKOFF — transient failure handling
# ═══════════════════════════════════════════════════════════════════

def _retry_with_backoff(fn, args, on_status, user_text, max_retries=3):
    """
    Retry a tool call with exponential backoff for transient failures.
    Returns (result, attempts_made). Never raises — all exceptions are caught.
    """
    import time as _time
    base_delay = 1.0  # seconds
    last_result = None

    for attempt in range(max_retries):
        try:
            last_result = _dispatch_tool(fn, args)
            _mark_tool_dispatched(fn)
            if not _looks_like_failure(fn, last_result):
                return last_result, attempt + 1
        except Exception as exc:
            last_result = {"ok": False, "error": f"{fn} crashed: {exc}"}
            _debug("RETRY_BACKOFF_EXCEPTION", {"fn": fn, "attempt": attempt + 1, "error": str(exc)})

        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)  # exponential backoff: 1s, 2s, 4s
            _debug("RETRY_BACKOFF_WAIT", {"fn": fn, "attempt": attempt + 1, "delay_seconds": delay})
            _notify(on_status, quips.retry())
            try:
                _time.sleep(delay)
            except Exception:
                pass  # interrupted, proceed with next retry

    return last_result, max_retries


# ═══════════════════════════════════════════════════════════════════
# CAPABILITY CHECK — upfront gap detection
# ═══════════════════════════════════════════════════════════════════

def check_capabilities(requirements: list[str]) -> tuple[bool, list[str], list[str]]:
    """
    Check whether the required capabilities are available before attempting
    a task. Used for multi-step tasks that need specific tools.

    Args:
        requirements: list of capability names, e.g.
            "network_fetch", "shell", "file_write", "browser", "pyinstaller"

    Returns:
        (all_available, available, missing)
        all_available: True if every requirement is met
        available: list of met capability names
        missing: list of unmet capability names
    """
    available = []
    missing = []

    for cap in requirements:
        if cap == "network_fetch":
            # web_fetch / web_search available?
            if "web_fetch" in [t["function"]["name"] for t in TOOLS]:
                available.append(cap)
            else:
                missing.append(cap)
        elif cap == "shell":
            # Can we run shell commands? Check if subprocess works
            try:
                import subprocess
                subprocess.run(["echo", "ok"], capture_output=True, timeout=5)
                available.append(cap)
            except Exception:
                missing.append(cap)
        elif cap == "file_write":
            # Can we write files? Check if project root is writable
            try:
                project_root = Path(__file__).resolve().parent.parent
                test_file = project_root / ".fairy_capability_test"
                test_file.write_text("test")
                test_file.unlink()
                available.append(cap)
            except Exception:
                missing.append(cap)
        elif cap == "browser":
            if _BROWSER_AVAILABLE:
                available.append(cap)
            else:
                missing.append(cap)
        elif cap == "pyinstaller":
            # For building exe files — check if PyInstaller is installed
            try:
                import shutil
                if shutil.which("pyinstaller") or shutil.which("pyinstaller3"):
                    available.append(cap)
                else:
                    try:
                        import PyInstaller  # noqa: F401
                        available.append(cap)
                    except ImportError:
                        missing.append(cap)
            except Exception:
                missing.append(cap)
        else:
            # Unknown capability — treat as available by default
            available.append(cap)

    return (len(missing) == 0, available, missing)


def _repair_tool_args(fn: str, args: dict, user_text: str) -> dict:
    repaired = dict(args)
    text = user_text.lower().strip()

    if fn == "navigate_to" and not repaired.get("url"):
        url = None

        url_match = re.search(r"https?://\S+", text)
        if url_match:
            url = url_match.group(0)

        if not url:
            site_map = {
                "youtube": "https://www.youtube.com/",
                "yt": "https://www.youtube.com/",
                "google": "https://www.google.com/",
                "github": "https://github.com/",
                "reddit": "https://www.reddit.com/",
                "twitter": "https://twitter.com/",
                "x.com": "https://x.com/",
                "facebook": "https://www.facebook.com/",
                "instagram": "https://www.instagram.com/",
                "instagarm": "https://www.instagram.com/",
                "isntagram": "https://www.instagram.com/",
                "tiktok": "https://www.tiktok.com/",
                "discord": "https://discord.com/",
                "twitch": "https://www.twitch.tv/",
                "netflix": "https://www.netflix.com/",
                "spotify": "https://open.spotify.com/",
                "wikipedia": "https://www.wikipedia.org/",
                "wkipidia": "https://www.wikipedia.org/",
                "wiki": "https://www.wikipedia.org/",
            }
            for site, site_url in site_map.items():
                if re.search(r"(?<!\w)" + re.escape(site) + r"(?!\w)", text):
                    url = site_url
                    break

        if not url:
            match = re.search(r"(?:open|go to|navigate to|browse|visit)\s+(.+?)(?:\s+in\s+\w+)?$", text)
            if match:
                target = match.group(1).strip()
                if not target:
                    url = ""
                elif "." in target and " " not in target:
                    url = target if target.startswith(("http://", "https://")) else "https://" + target
                elif target in {"youtube", "instagram", "instagarm", "isntagram", "wikipedia", "wkipidia", "google", "discord"}:
                    url = _infer_url_from_text(target)
                else:
                    url = target

        if url:
            repaired["url"] = url
        if not repaired.get("browser"):
            repaired["browser"] = "default"

    elif fn == "search_on_site":
        if not repaired.get("site"):
            if "youtube" in text or "yt" in text:
                repaired["site"] = "youtube"
            elif "google" in text:
                repaired["site"] = "google"
        if not repaired.get("query"):
            m = re.search(r"search\s+(?:for\s+)?(.+)", text)
            if m:
                repaired["query"] = m.group(1).strip()

    elif fn == "browser_control":
        if not repaired.get("action"):
            if any(w in text for w in ("open", "go to", "navigate", "browse", "visit")):
                repaired["action"] = "go_to"
                if not repaired.get("url"):
                    repaired["url"] = _infer_url_from_text(user_text)
            elif "search" in text:
                repaired["action"] = "search"
                if not repaired.get("query"):
                    m = re.search(r"search\s+(?:for\s+)?(.+)", text)
                    if m:
                        repaired["query"] = m.group(1).strip()
            elif "click" in text:
                repaired["action"] = "smart_click"
                if not repaired.get("description"):
                    m = re.search(r"click\s+(.+)", text)
                    if m:
                        repaired["description"] = m.group(1).strip()
            elif "type" in text:
                repaired["action"] = "type"
            elif "scroll" in text:
                repaired["action"] = "scroll"
            elif "screenshot" in text or "screen shot" in text:
                repaired["action"] = "screenshot"
            elif "close" in text:
                repaired["action"] = "close_all"
        if not repaired.get("url") and repaired.get("action") == "go_to":
            repaired["url"] = _infer_url_from_text(user_text)

    elif fn == "web_search" and not repaired.get("query"):
        repaired["query"] = user_text

    elif fn == "web_search":
        q = str(repaired.get("query", "")).strip()
        q = q.replace("web serch", "web search").replace("websearch", "web search")
        q = q.replace("ester sunday", "easter sunday").replace("ester", "easter")
        repaired["query"] = q

    elif fn == "deep_research" and not repaired.get("topic"):
        repaired["topic"] = user_text

    elif fn == "get_location" and not repaired.get("place"):
        repaired["place"] = user_text

    elif fn == "get_directions":
        if not repaired.get("origin") or not repaired.get("destination"):
            match = re.search(r"from\s+(.+?)\s+to\s+(.+)", text)
            if match:
                repaired["origin"] = match.group(1).strip()
                repaired["destination"] = match.group(2).strip()

    elif fn == "code_sandbox" and not repaired.get("description"):
        repaired["description"] = user_text

    elif fn == "search_and_summarize" and not repaired.get("query"):
        repaired["query"] = user_text

    elif fn == "system_monitor":
        if not repaired.get("query"):
            # Try to map common system keywords to the query types the
            # skill understands (summary|cpu|memory|gpu|disk|network|battery|temp|processes).
            # Order matters: longer / more specific keywords first so e.g.
            # "processes" matches before "process" or "task".
            keyword_map = [
                ("processes", "processes"),
                ("process", "processes"),
                ("tasks", "processes"),
                ("task", "processes"),
                ("graphics", "gpu"), ("video card", "gpu"), ("nvidia", "gpu"), ("gpu", "gpu"),
                ("processor", "cpu"), ("cpu", "cpu"),
                ("memory", "memory"), ("ram", "memory"),
                ("storage", "disk"), ("ssd", "disk"), ("hdd", "disk"), ("disk", "disk"),
                ("battery", "battery"), ("charge", "battery"),
                ("temperature", "temp"), ("thermal", "temp"), ("temp", "temp"), ("heat", "temp"),
                ("wifi", "network"), ("ethernet", "network"), ("network", "network"),
            ]
            t = user_text.lower()
            for kw, q in keyword_map:
                if re.search(r"(?<!\w)" + re.escape(kw) + r"(?!\w)", t):
                    repaired["query"] = q
                    break
            if not repaired.get("query"):
                repaired["query"] = "summary"

    elif fn == "get_current_time" and not repaired.get("timezone"):
        # Default to system-local timezone; the skill handles None.
        repaired["timezone"] = None

    return repaired


def _run_tool_calls(tool_calls, on_status, user_text=""):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    calls = []
    seen = set()

    _debug("TOOL_CALLS_RAW", {
        "count": len(tool_calls),
        "raw": [{
            "name": (c.get("function") or {}).get("name", "?"),
            "args_preview": str((c.get("function") or {}).get("arguments", "?"))[:200]
        } for c in tool_calls]
    })

    for call in tool_calls:
        fn = _tool_name(call)
        if not fn:
            continue
        args = _tool_call_args(call)
        _debug("TOOL_CALL_PARSED", {"fn": fn, "args": args})
        if not args or all(v in (None, "", []) for v in args.values()):
            _debug("TOOL_ARGS_EMPTY", {"fn": fn, "user_text": user_text})
            args = _repair_tool_args(fn, args, user_text)
            _debug("TOOL_ARGS_REPAIRED", {"fn": fn, "args": args})
        try:
            dedup_key = (fn, json.dumps(args, sort_keys=True, default=str))
        except TypeError:
            dedup_key = (fn, repr(args))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        calls.append((call, fn, args))

    if not calls:
        return [], []

    results: list[object] = [None] * len(calls)
    safe_indexes = [
        i for i, (_, fn, _) in enumerate(calls) if fn in _PARALLEL_SAFE_TOOLS
    ]

    if len(safe_indexes) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(safe_indexes))) as pool:
            futures = {
                pool.submit(_run_tool, calls[i][1], calls[i][2], on_status, user_text): i
                for i in safe_indexes
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as exc:
                    results[i] = {"ok": False, "error": str(exc)}

    for i, (call, fn, args) in enumerate(calls):
        if results[i] is None:
            results[i] = _run_tool(fn, args, on_status, user_text)

    tool_messages = []
    used = []

    for (call, fn, args), result in zip(calls, results):
        used.append(fn)
        # FIX 3: log every tool result so dispatched calls can be correlated
        # with outcomes in fairy_debug.log.
        try:
            _debug("TOOL_RESULT", {"fn": fn, "result": str(result)[:500]})
        except Exception:
            pass
        tool_messages.append({
            "role": "tool",
            "content": _json_content(result),
            "name": fn,
            "tool_name": fn,
        })

    return tool_messages, used


# ═══════════════════════════════════════════════════════════════════
# FAIRY 2.0 — STRUCTURED PLANNER (Mark-L inspired)
# ═══════════════════════════════════════════════════════════════════

def _is_action_request(user_text: str) -> bool:
    t = user_text.lower()
    action_verbs = (
        "create ", "make ", "build ", "write ", "run ", "open ", "click ",
        "type ", "upload ", "download ", "install ", "change ", "modify ",
        "control ", "delete ", "send ", "post ", "execute ", "navigate ",
        "launch ", "start ", "stop ", "kill ", "restart ", "play ", "pause ",
        "fix ", "debug ", "code ", "coding", "script", "refactor",
        "implement ", "solve ", "help me code", "help with code",
        "review this code", "optimize this", "ask qwen", "ask coder",
        "ask openrouter", "use qwen", "use coder", "generate code",
    )
    return any(x in t for x in action_verbs)


def _should_use_planner(user_text: str, profile: dict) -> bool:
    # If it's a pure question with no action intent, skip planner.
    is_question = (user_text.strip().endswith('?') or
                   any(user_text.lower().startswith(q) for q in ('how', 'what', 'why', 'when', 'where', 'who', 'which')))
    if is_question and not _is_action_request(user_text):
        return False
    # Otherwise, use the action-verb detection.
    return _is_action_request(user_text)


def _infer_url_from_text(text: str) -> str:
    text = text.lower().strip()
    m = re.search(r"https?://\S+", text)
    if m:
        return m.group(0)

    site_map = {
        "youtube": "https://www.youtube.com/",
        "yt": "https://www.youtube.com/",
        "youtu": "https://www.youtube.com/",
        "google": "https://www.google.com/",
        "github": "https://github.com/",
        "reddit": "https://www.reddit.com/",
        "twitter": "https://twitter.com/",
        "x.com": "https://x.com/",
        "facebook": "https://www.facebook.com/",
        "instagram": "https://www.instagram.com/",
        "instagarm": "https://www.instagram.com/",
        "isntagram": "https://www.instagram.com/",
        "insta": "https://www.instagram.com/",
        "ig": "https://www.instagram.com/",
        "tiktok": "https://www.tiktok.com/",
        "discord": "https://discord.com/",
        "twitch": "https://www.twitch.tv/",
        "netflix": "https://www.netflix.com/",
        "spotify": "https://open.spotify.com/",
        "wikipedia": "https://www.wikipedia.org/",
        "wkipidia": "https://www.wikipedia.org/",
        "wiki": "https://www.wikipedia.org/",
        # Real sites with common misspellings / Whisper near-misses
        "hoyolab": "https://www.hoyolab.com/",
        "hoyola": "https://www.hoyolab.com/",
    }
    for site, url in site_map.items():
        if re.search(r"(?<!\w)" + re.escape(site) + r"(?!\w)", text):
            return url

    m = re.search(r"(?:open|go to|navigate to|browse|visit)\s+(.+?)(?:\s+in\s+\w+)?$", text)
    if m:
        target = m.group(1).strip().replace(" ", "")
        if not target:
            return ""
        if "." in target and " " not in target:
            return target if target.startswith(("http://", "https://")) else "https://" + target
        if target in site_map:
            return site_map[target]
        return target

    return ""


def _infer_name_from_text(text: str) -> str:
    text = text.lower()
    m = re.search(r"(?:create|make|build)\s+(?:a\s+)?(?:new\s+)?skill\s+(?:for\s+|called\s+|named\s+)?([a-z_][a-z0-9_]*)", text)
    if m:
        return m.group(1)
    m = re.search(r"skill\s+(?:for\s+|to\s+)([a-z]+)", text)
    if m:
        return m.group(1)
    return "custom_skill"


def _structured_tool_decision(user_text: str, tools: list, history: list | None = None) -> dict:
    prompt = build_action_decision_prompt(user_text, tools)
    messages = []
    if history:
        messages.extend(_compact_history(history)[-4:])
    messages.append({"role": "user", "content": prompt})

    try:
        resp = _brain(
            MODEL_BRAIN,
            messages=messages,
            options={"num_predict": 400, "num_ctx": 4096, "temperature": 0.1},
        )
        content = resp.get("message", {}).get("content", "").strip()
        _debug("PLANNER_RAW", {"content": content[:500]})

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"tool": None, "error": "no JSON found", "raw": content[:200]}

        data = json.loads(content[start:end+1])
        return {
            "tool": data.get("tool"),
            "arguments": data.get("arguments", {}),
            "reason": data.get("reason", ""),
        }
    except Exception as exc:
        _debug("PLANNER_FAIL", {"error": str(exc)})
        # No brain available — return a clean fallback so the planner never
        # blocks the user.  Omit the "error" key so callers that assert
        # "error not in decision" (e.g. tests) continue to work.
        return {"tool": None}


def _execute_planner_tool(fn: str, args: dict, user_text: str, on_status) -> tuple:
    _mark_tool_dispatched(fn)   # FIX 1: record planner-path tool dispatch
    args = _repair_tool_args(fn, args, user_text)
    _debug("PLANNER_ARGS_REPAIRED", {"fn": fn, "args": args})

    schema = next((t for t in TOOLS if t.get("function", {}).get("name") == fn), None)
    if schema:
        required = schema.get("function", {}).get("parameters", {}).get("required", [])
        missing = [r for r in required if r not in args or args[r] in (None, "", [])]
        if missing:
            _debug("PLANNER_MISSING_REQUIRED", {"fn": fn, "missing": missing})
            for m in missing:
                if m == "url":
                    args[m] = _infer_url_from_text(user_text)
                elif m in ("query", "topic", "place", "description"):
                    args[m] = user_text
                elif m == "name":
                    args[m] = _infer_name_from_text(user_text)

    _notify(on_status, TOOL_ANNOUNCEMENTS.get(fn, "Working on it..."))
    result = _run_tool(fn, args, on_status, user_text)
    _debug("PLANNER_RESULT", {"fn": fn, "result_preview": str(result)[:300]})

    _notify(on_status, "Synthesizing...")
    try:
        synth_prompt = build_synthesis_prompt(user_text, [{"tool": fn, "result": result}])
        profile = _complexity_profile(user_text)
        _lang = _detect_language(user_text)
        synth_resp = _brain(
            MODEL_BRAIN,
            messages=[
                {"role": "system", "content": build_system_prompt(profile, language=_lang)},
                *_compact_history([])[-4:],
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": f"[Internal: called {fn} with result]"},
                {"role": "user", "content": synth_prompt},
            ],
            options={"num_predict": 600, "num_ctx": 4096},
        )
        content = synth_resp.get("message", {}).get("content", "") or ""
        _log_brain_response(synth_resp, "PLANNER_SYNTH_DIAGNOSTIC")
    except Exception as exc:
        _debug("PLANNER_SYNTH_FAIL", {"error": str(exc)})
        content = ""

    if not content.strip():
        norm_result = result
        if isinstance(result, str):
            try:
                norm_result = json.loads(result)
            except Exception:
                pass

        if isinstance(norm_result, dict) and (norm_result.get("ok") is False or norm_result.get("status") == "error"):
            error_msg = norm_result.get("error") or norm_result.get("message", "Unknown error")
            content = f"That didn't work, Master. {error_msg}."
        else:
            content = f"Done, Master. {str(result)[:200]}"

    content = _strip_meta_blocks(content)
    return content, result


def _answer_is_empty(content: str) -> bool:
    return not content or not content.strip()


def _recover_missing_tool_call(messages, user_text: str, profile: dict):
    if not _is_action_request(user_text):
        return None

    messages.append({
        "role": "user",
        "content": (
            "You acknowledged the request, but you have not performed the action. "
            "Do the action now. Choose the appropriate tool and CALL it. "
            "Do not reply with a promise, acknowledgement, or plan. "
            "If no tool can perform the requested action, then explain why."
        ),
    })

    try:
        response = _brain(
            MODEL_BRAIN,
            messages=messages,
            tools=_tools_for_tier(0),
            options={
                "num_predict": 700 if profile["mode"] != "deep" else 1200,
                "num_ctx": 4096,
            },
        )
    except Exception:
        return None
    _log_brain_response(response, "RECOVERY_DIAGNOSTIC")

    recovered_message = response.get("message", {})
    if not (recovered_message.get("tool_calls") or []):
        raw_content = recovered_message.get("content", "") or ""
        pseudo_calls, cleaned_content = _extract_pseudo_tool_calls(raw_content)
        if pseudo_calls:
            _debug("PSEUDO_TOOL_CALLS_RECOVERED", {
                "where": "recovery",
                "count": len(pseudo_calls),
                "fns": [c["function"]["name"] for c in pseudo_calls],
            })
            recovered_message = {
                **recovered_message,
                "content": cleaned_content,
                "tool_calls": pseudo_calls,
            }
    return recovered_message


# =====================================================================
# LONG-TERM MEMORY: forget_fact handler + y/n confirmation
# =====================================================================
def _format_fact_line(f: dict, idx: int) -> str:
    """One human-readable line per fact for the list output."""
    text = (f.get("text") or "").strip()
    cat = f.get("category", "")
    fid = f.get("id", "")[:8]
    return f"{idx + 1}. [{cat}] {text}  (id:{fid})"


def _handle_forget_fact(arg: str) -> str:
    """Tool dispatcher: list facts or delete by partial text match."""
    ltm = get_long_term_memory()
    arg = (arg or "").strip()
    if not arg:
        return "Provide a fact text (or 'list') to use forget_fact, Master."
    if arg.lower() == "list":
        facts = ltm.list_facts()
        if not facts:
            return "I don't have any long-term facts stored about you yet, Master."
        lines = [_format_fact_line(f, i) for i, f in enumerate(facts)]
        return "Here's what I remember:\n" + "\n".join(lines)
    matches = ltm.find_facts_matching(arg)
    if not matches:
        return f"No stored fact matches '{arg}', Master. Use 'list' to see all."
    if len(matches) == 1:
        removed = matches[0]
        ltm.delete_fact(removed["id"])
        return f"Forgotten, Master. No longer remembering: \"{removed['text']}\"."
    # Multiple matches: remove all that contain the substring.
    removed_texts = [m["text"] for m in matches]
    for m in matches:
        ltm.delete_fact(m["id"])
    return "Forgotten, Master:\n" + "\n".join(f"- {t}" for t in removed_texts)


def _handle_pending_proposals_reply(stripped: str) -> Optional[str]:
    """If there are pending fact proposals, treat the user's y/n as a verdict.

    Returns the reply string when handled, or None when there are no pending
    proposals (so the caller can fall through to normal routing).
    """
    ltm = get_long_term_memory()
    pending = ltm.get_pending_proposals()
    if not pending:
        return None
    t = (stripped or "").lower().strip()
    if t in ("y", "yes"):
        new_ids: List[str] = []
        for p in list(pending):
            nid = ltm.confirm_proposal(p, category="fact_about_master")
            if nid:
                new_ids.append(nid)
        ltm.clear_pending()
        if new_ids:
            joined = "; ".join(f"\"{p}\"" for p in pending)
            return f"Got it, Master. I'll remember: {joined}."
        return "All right, Master — nothing to remember just now."
    if t in ("n", "no"):
        ltm.clear_pending()
        return "No problem, Master — I won't remember that."
    return None


def _propose_facts_from_turn(user_text: str) -> Optional[str]:
    """Run lightweight extraction on the user turn; queue pending proposals.

    Returns the confirmation prompt string if any proposals were added,
    otherwise None. Capped at 2 proposals total (existing + new).
    """
    if not user_text:
        return None
    candidates = extract_proposals(user_text)
    if not candidates:
        return None
    ltm = get_long_term_memory()
    added = 0
    for c in candidates:
        if ltm.add_pending_proposal(c):
            added += 1
    if added == 0:
        return None
    pending = ltm.get_pending_proposals()
    if not pending:
        return None
    lines = [f"✦ Remember this? \"{p}\" (y/n)" for p in pending]
    return "\n".join(lines)


# =====================================================================
# SINGLE LLM RESPONSE (no tools, one turn)
# =====================================================================
def _single_llm_response(user_text: str, history: list, profile: dict, memory_context: str = "") -> tuple[str, list]:
    _lang = _detect_language(user_text)
    _facts_block = get_long_term_memory().get_injection_block()
    system_prompt = build_system_prompt(
        profile,
        skill_manager.list_skills() if hasattr(skill_manager, 'list_skills') else [],
        memory_context,
        language=_lang,
        facts_injection=_facts_block,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        *_compact_history(history),
        {"role": "user", "content": user_text},
    ]
    try:
        resp = _brain(
            MODEL_BRAIN,
            messages=messages,
            tools=[],
            options={"num_predict": 600, "num_ctx": 4096}
        )
        content = resp.get("message", {}).get("content", "") or ""
        _log_brain_response(resp, "SINGLE_LLM_DIAGNOSTIC")
        if not content.strip():
            # Retry with direct prompt if system prompt produced empty output
            retry_resp = _brain(
                MODEL_BRAIN,
                messages=[{"role": "user", "content": user_text}],
                tools=[],
                options={"num_predict": 600, "num_ctx": 2048}
            )
            content = retry_resp.get("message", {}).get("content", "") or ""
            _log_brain_response(retry_resp, "SINGLE_LLM_RETRY_DIAGNOSTIC")
        if not content.strip():
            content = "My local brain returned an empty response. Could you rephrase your question, Master?"
    except Exception as e:
        content = f"Could not get a response: {e}"
    content = _strip_meta_blocks(content)
    return content, history + [{"role": "user", "content": user_text}, {"role": "assistant", "content": content}]


# =====================================================================
# ADAPTIVE LOOP (with optional max_turns / tools override)
# =====================================================================
def _adaptive_loop(user_text: str, history: list, profile: dict,
                   memory_context: str, on_status=None,
                   max_turns: int | None = None,
                   tools_list: list | None = None) -> tuple[str, list]:
    if max_turns is None:
        max_turns = profile["max_turns"]
    if tools_list is None:
        tools_list = _tools_for_tier(0)

    _lang = _detect_language(user_text)
    _facts_block = get_long_term_memory().get_injection_block()
    system_prompt = build_system_prompt(
        profile,
        skill_manager.list_skills() if hasattr(skill_manager, 'list_skills') else [],
        memory_context,
        language=_lang,
        facts_injection=_facts_block,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        *_compact_history(history),
        {"role": "user", "content": user_text},
    ]

    predict = {
        "quick": 500,
        "normal": 900,
        "deep": 1400,
    }[profile["mode"]]

    options = {
        "num_predict": predict,
        "num_ctx": 4096 if profile["mode"] != "deep" else 8192,
    }

    tools_used = []
    tool_counts = {}
    last_content = ""
    action_recovery_used = False
    verification_rounds = 0
    forced_synthesis = False
    consecutive_no_tools = 0

    for turn in range(max_turns):
        try:
            response = _brain(
                MODEL_BRAIN,
                messages=messages,
                tools=tools_list,
                options=options,
            )
        except Exception as exc:
            _notify(on_status, f"Brain call failed: {exc}")
            _debug("BRAIN_CALL_FAIL", {"turn": turn, "error": str(exc)})
            fallback = (
                f"Couldn't reach my own brain just now, Master ({exc}). "
                "Check that Ollama is running and the model is loaded, then try again."
            )
            return fallback, history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": fallback},
            ]
        message = response.get("message", {})
        tool_calls = message.get("tool_calls") or []
        content = message.get("content", "") or ""

        if not tool_calls:
            pseudo_calls, cleaned_content = _extract_pseudo_tool_calls(content)
            if pseudo_calls:
                _debug("PSEUDO_TOOL_CALLS_RECOVERED", {
                    "turn": turn,
                    "count": len(pseudo_calls),
                    "fns": [c["function"]["name"] for c in pseudo_calls],
                })
                tool_calls = pseudo_calls
                content = cleaned_content
                message = {**message, "content": content, "tool_calls": tool_calls}

        last_content = content
        _log_brain_response(response, "ADAPTIVE_MAIN_DIAGNOSTIC")

        # If the brain hit its output token limit, request a continuation
        # before bailing — this prevents truncated replies (e.g. a story
        # cut off mid-sentence) from being returned to the user.
        if not tool_calls and not content.strip() and response.get("done_reason") == "length":
            try:
                cont_messages = messages + [
                    {"role": "assistant", "content": content or ""},
                    {"role": "user", "content": "Continue exactly where you left off. Do not restart."},
                ]
                cont = _brain(
                    MODEL_BRAIN,
                    messages=cont_messages,
                    tools=tools_list,
                    options={**options, "num_predict": min(1200, predict * 2)},
                )
                cont_msg = cont.get("message", {}) or {}
                cont_content = (cont_msg.get("content", "") or "").strip()
                _log_brain_response(cont, "ADAPTIVE_LENGTH_CONTINUE_DIAGNOSTIC")
                if cont_content:
                    content = (content or "") + cont_content
                    message = {**cont_msg, "content": content}
            except Exception as exc:
                _debug("ADAPTIVE_LENGTH_CONTINUE_FAIL", {"error": str(exc)})

        _debug("BRAIN_RESPONSE", {
            "turn": turn,
            "content_preview": content[:200] if content else "(empty)",
            "tool_count": len(tool_calls),
        })

        if not tool_calls:
            if (
                not action_recovery_used
                and _is_action_request(user_text)
                and not tools_used
                and turn < max_turns - 1
                and content.strip()
                and (
                    len(content.strip()) < 400
                    or any(
                        phrase in content.lower()
                        for phrase in (
                            "let me", "i'll", "i will", "sure", "ok",
                            "give me a moment", "one moment", "just a sec",
                            "hang on", "wait a moment", "opening",
                            "starting", "activating", "searching for",
                            "let's take a look", "taking a look",
                            "there you go", "done", "opened", "started",
                            "activated", "i handled it", "i found it",
                            "it's open", "now playing", "searching",
                            "created", "built", "made", "finished",
                            "completed", "wrote", "wrote it", "here it is",
                            "it's ready", "all set", "good to go",
                        )
                    )
                )
            ):
                messages.append(message)
                recovered = _recover_missing_tool_call(
                    messages, user_text, profile
                )
                action_recovery_used = True

                if recovered:
                    recovered_calls = recovered.get("tool_calls") or []
                    if recovered_calls:
                        message = recovered
                        tool_calls = recovered_calls
                        last_content = recovered.get("content", "") or ""
                    else:
                        return recovered.get("content", "") or content, history + [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": recovered.get("content", "") or content},
                        ]
                else:
                    content = _strip_meta_blocks(content)
                    return content, history + [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": content},
                    ]
            else:
                if not content.strip():
                    messages.append({
                        "role": "user",
                        "content": "Give the final answer now, using the results already available.",
                    })
                    try:
                        retry = _brain(
                            MODEL_BRAIN,
                            messages=messages,
                            tools=tools_list,
                            options={**options, "num_predict": min(700, predict)},
                        )
                        content = retry.get("message", {}).get("content", "") or ""
                        _log_brain_response(retry, "ADAPTIVE_RETRY_DIAGNOSTIC")
                    except Exception:
                        content = ""
                if not content.strip():
                    content = "I couldn't get a clean answer out of my brain that time."
                content = _strip_meta_blocks(content)
                return content, history + [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": content},
                ]

        messages.append(message)

        tool_messages, used = _run_tool_calls(tool_calls, on_status, user_text)
        if not used:
            consecutive_no_tools += 1
            _debug("NO_TOOLS_USED", {"turn": turn, "consecutive": consecutive_no_tools})
            if consecutive_no_tools >= 2:
                _debug("NO_TOOLS_USED_BREAK", {"reason": "2 consecutive empty tool rounds"})
                messages.append({
                    "role": "user",
                    "content": "The tool calls aren't working. Answer the question directly based on your knowledge.",
                })
                try:
                    bail_resp = _brain(
                        MODEL_BRAIN,
                        messages=messages,
                        tools=[],
                        options={**options, "num_predict": min(700, predict)},
                    )
                    bail_content = bail_resp.get("message", {}).get("content", "") or ""
                    _log_brain_response(bail_resp, "ADAPTIVE_BAIL_DIAGNOSTIC")
                    if bail_content.strip():
                        bail_content = _strip_meta_blocks(bail_content)
                        return bail_content, history + [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": bail_content},
                        ]
                except Exception:
                    pass
                stuck = "The tools aren't cooperating, Master. Try rephrasing or asking me directly."
                return stuck, history + [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": stuck},
                ]
            messages.append({
                "role": "user",
                "content": "The tool call failed to execute. Answer directly based on what you know, or try a different tool.",
            })
            continue

        consecutive_no_tools = 0

        tools_used.extend(used)
        for fn in used:
            tool_counts[fn] = tool_counts.get(fn, 0) + 1
        messages.extend(tool_messages)

        if not forced_synthesis:
            if tool_counts.get("deep_research", 0) >= 3 or tool_counts.get("web_search", 0) >= 4:
                forced_synthesis = True
                messages.append({
                    "role": "user",
                    "content": (
                        "STOP. You have done enough research. "
                        "Synthesize everything you have found into a final answer NOW. "
                        "Do not call any more tools. Answer based on the evidence already in this conversation."
                    ),
                })
                try:
                    synth_resp = _brain(
                        MODEL_BRAIN,
                        messages=messages,
                        tools=[],
                        options={**options, "num_predict": min(1000, predict)},
                    )
                    synth_content = synth_resp.get("message", {}).get("content", "") or ""
                    _log_brain_response(synth_resp, "ADAPTIVE_FORCED_SYNTH_DIAGNOSTIC")
                except Exception:
                    synth_content = ""
                final = synth_content.strip() or last_content.strip() or "I've gathered the research. Let me know if you need more detail, Master."
                return final, history + [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": final},
                ]
            elif (
                profile["mode"] == "deep"
                and len(tools_used) >= 2
                and verification_rounds < 1
                and turn < max_turns - 2
            ):
                verification_rounds += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "Checkpoint: inspect the evidence you have. If the user's goal "
                        "is solved, answer now. If something concrete is still missing, "
                        "call the next tool needed. Do not search merely for extra detail."
                    ),
                })

        _notify(on_status, quips.synthesize())

    if last_content.strip():
        last_content = _strip_meta_blocks(last_content)
        return last_content, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": last_content},
        ]

    if tools_used:
        messages.append({
            "role": "user",
            "content": "Give the final answer now based on all the results above. Do not call any more tools.",
        })
        try:
            final_resp = _brain(
                MODEL_BRAIN,
                messages=messages,
                tools=[],
                options={**options, "num_predict": min(800, predict)},
            )
            final_content = final_resp.get("message", {}).get("content", "") or ""
            _log_brain_response(final_resp, "ADAPTIVE_FINAL_DIAGNOSTIC")
            if final_content.strip():
                return final_content, history + [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": final_content},
                ]
        except Exception:
            pass

    stuck = "I hit my turn limit without a clean answer, Master. Try asking again or simplify the request."
    return stuck, history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": stuck},
    ]


# ═══════════════════════════════════════════════════════════════════
# _dispatch_tool — Direct tool dispatcher
# ═══════════════════════════════════════════════════════════════════

def _dispatch_tool(fn: str, args: dict):
    _mark_tool_dispatched(fn)   # FIX 1: record every actual tool dispatch
    _debug("DISPATCH_TOOL", {"fn": fn, "args": args})
    args = args if isinstance(args, dict) else {}

    # ── Unified approval gate ───────────────────────────────────────
    # File-mutating tools (write_file, mkdir, delete, zip_*, move/rename)
    # route through controller/approval_gate.py which prompts the Master
    # in-terminal or escalates to the Discord approval queue. This is
    # deliberately a single, thin entry point — the gate owns the
    # policy, this site only does the if-check.
    #
    # Discord re-dispatch bypass: when discord_bot.py re-dispatches with
    # owner authority (after the Master approved), it sets
    # _gate_bypass = True on the request context so we skip the gate
    # here and don't double-prompt. This is set by the re-dispatch code
    # in discord_bot.py so the bypass is explicit and auditable.
    if fn in approval_gate.FILE_MUTATING_TOOLS:
        _bypass = (
            _request_ctx_for_gate_set
            and getattr(_request_ctx_for_gate, "_gate_bypass", False)
        )
        if not _bypass:
            try:
                from controller import approval_gate as _gate
            except Exception:
                _gate = None
            if _gate is not None:
                try:
                    # Determine a sensible user identifier for the gate.
                    # The Discord bot sets is_owner/user_id on a thread-local;
                    # the terminal path leaves them unset, so the gate
                    # defaults to the terminal caller.
                    _user_name = getattr(_request_ctx_for_gate, "caller_name", "Master") \
                        if _request_ctx_for_gate_set else "Master"
                    _user_id = getattr(_request_ctx_for_gate, "user_id", None) \
                        if _request_ctx_for_gate_set else None
                    outcome, args = _gate.request_approval(
                        user_name=_user_name,
                        user_id=_user_id,
                        tool=fn,
                        args=dict(args),
                    )
                    if outcome != _gate.OUTCOME_APPROVED:
                        return {"ok": False, "error": f"denied by approval gate ({fn})", "gate": "denied"}
                except _gate.ApprovalRequired:
                    # Discord path — the gate raised because the async side
                    # needs to take over. Re-raise as MasterApprovalRequired
                    # so the existing on_message handler in discord_bot.py
                    # picks it up unchanged.
                    try:
                        from controller import discord_bot as _db
                        raise _db.MasterApprovalRequired(
                            sys.exc_info()[1].request_info,
                            getattr(_request_ctx_for_gate, "approval_event", None)
                            if _request_ctx_for_gate_set else None,
                        )
                    except ImportError:
                        # No discord bot loaded (pure-terminal env) — the
                        # gate's own raise is the only signal we have, so
                        # surface it as a denial rather than silently run.
                        return {"ok": False, "error": f"denied by approval gate ({fn})", "gate": "discord_unavailable"}

    if fn == "web_search":
        return web_search(args.get("query", ""))
    if fn == "web_fetch":
        return web_fetch(args.get("url", ""))
    if fn == "deep_research":
        return deep_research(args.get("topic", ""))
    if fn == "get_location":
        return get_location(args.get("place", ""))
    if fn == "get_directions":
        return get_directions(args.get("origin", ""), args.get("destination", ""))
    if fn == "search_nearby":
        return search_nearby(args.get("location", ""), args.get("query", ""))
    if fn == "get_current_time":
        return get_current_time(args.get("timezone"))
    if fn == "ask_qwen":
        return ask_qwen(args.get("prompt", ""))
    if fn == "code_sandbox":
        return _code_sandbox(args.get("description", ""), args.get("use_cloud", False))
    if fn == "list_skills":
        return skill_manager.list_skills()
    if fn == "run_skill":
        return skill_manager.run_skill(args.get("name", ""), args.get("arguments", {}))
    if fn == "create_skill":
        return handle_create_skill(args.get("name", ""), args.get("description", ""))
    if fn == "navigate_to" and _BROWSER_AVAILABLE:
        return navigate_to(args.get("url", ""), args.get("browser", "default"))
    if fn == "search_on_site" and _BROWSER_AVAILABLE:
        return search_on_site(args.get("query", ""), args.get("engine") or args.get("site", "google"))
    if fn == "add_to_cart_amazon" and _BROWSER_AVAILABLE:
        return add_to_cart_amazon(args.get("item_name", ""))
    if fn == "upload_instagram_reel" and _BROWSER_AVAILABLE:
        return upload_instagram_reel(args.get("video_path", ""), args.get("caption", ""))
    if fn == "click_element" and _BROWSER_AVAILABLE:
        return click_element(selector=args.get("selector"), text=args.get("text") or args.get("description"))
    if fn == "type_text" and _BROWSER_AVAILABLE:
        return type_text(selector=args.get("selector") or args.get("into"), text=args.get("text", ""))
    if fn == "get_page_info" and _BROWSER_AVAILABLE:
        return get_page_info()
    if fn == "close_browser" and _BROWSER_AVAILABLE:
        return close_browser()
    if fn == "browser_control":
        return browser_control(args)
    if fn == "computer_control":
        return computer_control(args)
    if fn == "set_volume":
        return set_volume(args.get("value", 50))
    if fn == "get_volume":
        return get_volume()
    if fn == "take_screenshot":
        return take_screenshot(args.get("path"))
    if fn == "set_reminder":
        return set_reminder(args.get("title", ""), args.get("message", ""), args.get("delay_seconds", 60))
    if fn == "send_notification":
        return send_notification(args.get("title", ""), args.get("message", ""))
    if fn == "system_monitor":
        return system_monitor(args.get("query", "summary"))
    if fn == "reminder_tool":
        return reminder_tool(args.get("action", ""), args.get("title", ""), args.get("message", ""), args.get("delay_seconds", 0))
    if fn == "screen_process":
        # If the brain LLM called screen_process, vision is effectively on.
        # Make sure the persistent session is running.
        if not is_session_ready():
            start_session(player=None, timeout=10.0)
        _set_vision_state("on", angle=args.get("angle", "screen"))
        return screen_process(args)
    if fn == "vision_analyze":
        # Fairy's Llama Vision eyes — capture and describe
        result = vision_analyze(args)
        return result
    if fn == "vision_describe":
        # Fairy's Llama Vision eyes — describe specific image file
        result = vision_describe(args)
        return result
    if fn == "send_message":
        return send_message(args)
    if fn == "forget_fact":
        return _handle_forget_fact(args.get("fact", ""))
    if fn == "ask_openrouter" and _OPENROUTER_AVAILABLE:
        return ask_openrouter(args.get("prompt", ""), args.get("model", "qwen3-coder"), args.get("system_prompt"))
    if fn == "ask_or_coder" and _OPENROUTER_AVAILABLE:
        return ask_or_coder(args.get("prompt", ""))
    if fn == "ask_or_smart" and _OPENROUTER_AVAILABLE:
        return ask_or_smart(args.get("prompt", ""))
    if fn == "ask_or_cheap" and _OPENROUTER_AVAILABLE:
        return ask_or_cheap(args.get("prompt", ""))
    # ================================================================
    # NEW: search_and_summarize tool – offloads to OpenRouter
    # ================================================================
    if fn == "search_and_summarize" and _OPENROUTER_AVAILABLE:
        query = args.get("query", "")
        # First, do a web search
        raw_results = web_search(query)
        if not raw_results:
            return "No search results found."
        # Then summarise with OpenRouter
        summary_prompt = (
            "Summarise the following search results concisely, focusing on the key facts and answering the user's intent. "
            "Keep it under 150 words.\n\n"
            f"Search results:\n{raw_results}"
        )
        try:
            summary = ask_or_coder(summary_prompt)
            return summary
        except Exception as e:
            return f"Search succeeded but summarisation failed: {e}\n\nRaw results:\n{raw_results[:1000]}"
    if fn == "ask_openai" and _LLM_APIS_AVAILABLE:
        return ask_openai(args.get("prompt", ""), args.get("model", "gpt-4o-mini"))
    if fn == "ask_claude" and _LLM_APIS_AVAILABLE:
        return ask_claude(args.get("prompt", ""), args.get("model", "claude-3-haiku-20240307"))
    if fn == "ask_gemini" and _LLM_APIS_AVAILABLE:
        return ask_gemini(args.get("prompt", ""), args.get("model", "gemini-1.5-flash"))
    if fn == "ask_grok" and _LLM_APIS_AVAILABLE:
        return ask_grok(args.get("prompt", ""), args.get("model", "grok-2"))
    if fn == "ask_meta" and _LLM_APIS_AVAILABLE:
        return ask_meta(args.get("prompt", ""), args.get("model", "llama-3.3-70b"))
    if fn == "ask_deepseek" and _LLM_APIS_AVAILABLE:
        return ask_deepseek(args.get("prompt", ""), args.get("model", "deepseek-chat"))
    if fn == "ask_kimi" and _LLM_APIS_AVAILABLE:
        return ask_kimi(args.get("prompt", ""), args.get("model", "moonshot-v1-8k"))

    # ─── Planner misrouting fallback ──────────────────────────────────────────
    # The free-tier LLM sometimes reads the `computer_control` description
    # ("Actions: ... open_app, open_url.") and emits "open_app" / "open_url"
    # as a *top-level* tool name instead of `computer_control` with
    # `action="open_app"`. Remap those two known-confused names onto the
    # existing computer_control launcher. This is purely additive — the
    # normal computer_control path is unchanged. See planner prompt
    # (core/prompts.py:67) and TOOLS schema (agent_controller.py:909, 992, 996).
    if fn in ("open_app", "open_url"):
        target = args.get("value")
        if not target:
            target = args.get("app_or_url")
        if not target:
            return {"ok": False, "error": f"{fn} requires 'value' (app name or URL)."}
        return computer_control({"action": "open", "value": str(target)})

    # Similar remap for close verbs
    if fn in ("close_app", "kill_app", "quit_app", "close_application", "kill_application"):
        target = args.get("value") or args.get("app_or_url") or args.get("app_name")
        if not target:
            return {"ok": False, "error": f"{fn} requires 'value' (app name)."}
        return computer_control({"action": "close_app", "value": str(target)})

    # ─── File-mutating tools (approval-gated) ────────────────────────────
    # The approval gate at the top of this function has already verified
    # the user said ✅ (terminal) or the Discord queue approved. We
    # just execute the action here. All paths are sandboxed to the
    # project root for write_file / mkdir / move / rename unless the
    # user explicitly opted out via ``allow_outside_project: True``
    # — the gate prompt mentions that knob.
    if fn in approval_gate.FILE_MUTATING_TOOLS:
        from pathlib import Path
        return _execute_file_mutation(fn, args)

    return {"ok": False, "error": f"Unknown tool: {fn}"}


def _execute_file_mutation(fn: str, args: dict):
    """Backend for the file-mutating tools wired into ``_dispatch_tool``.

    Every code path here runs only after the approval gate has approved
    the call. ``args`` here is the (possibly redirected) dict returned
    by ``approval_gate.request_approval`` — i.e. it may already point at
    ``fairy_outputs/...`` if the Master typed "alt".

    Safety nets (defense in depth, in case the gate is ever bypassed):
      - Path resolution via ``Path.resolve()`` to defeat symlink tricks.
      - All writes/mkdirs/renames sandboxed to the project root unless
        ``allow_outside_project`` is truthy. A bare path like
        ``C:/Windows/...`` is rejected with a clear error.
      - The existing file is read once and reported back in the result
        so the caller can see what was actually replaced.
    """
    from pathlib import Path
    import shutil
    import zipfile
    import io

    project_root = Path(__file__).resolve().parent.parent
    allow_outside = bool(args.get("allow_outside_project", False))

    def _safe_resolve(p: str) -> Path:
        """Resolve ``p`` and reject paths outside the project root
        unless the gate explicitly approved going outside."""
        if not p:
            raise ValueError("path is required")
        candidate = Path(p).expanduser()
        # Make absolute if relative — anchor to project root so casual
        # "test.txt" doesn't escape to cwd.
        if not candidate.is_absolute():
            candidate = (project_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if not allow_outside:
            try:
                candidate.relative_to(project_root)
            except ValueError as exc:
                raise PermissionError(
                    f"path {candidate} is outside the project root "
                    f"({project_root}). Pass allow_outside_project=True "
                    f"if you really mean it."
                ) from exc
        return candidate

    try:
        if fn == "write_file":
            path = _safe_resolve(args.get("path", ""))
            content = args.get("content", "")
            if not isinstance(content, str):
                return {"ok": False, "error": "content must be a string"}
            existed = path.exists()
            old_text = ""
            if existed and path.is_file():
                try:
                    old_text = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    old_text = ""
            # mkdir parents so write_file("fairy_outputs/sub/x.txt", ...)
            # doesn't need a separate mkdir call.
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {
                "ok": True,
                "tool": "write_file",
                "path": str(path),
                "bytes": len(content.encode("utf-8")),
                "overwrote": existed,
                "old_bytes": len(old_text.encode("utf-8")) if old_text else 0,
            }

        if fn == "mkdir":
            path = _safe_resolve(args.get("path", ""))
            existed = path.exists()
            path.mkdir(parents=True, exist_ok=True)
            return {
                "ok": True,
                "tool": "mkdir",
                "path": str(path),
                "created": not existed,
            }

        if fn == "delete":
            path = _safe_resolve(args.get("path", ""))
            if not path.exists():
                return {"ok": False, "error": f"path does not exist: {path}"}
            if path.is_dir():
                shutil.rmtree(path)
                kind = "directory"
            else:
                path.unlink()
                kind = "file"
            return {"ok": True, "tool": "delete", "path": str(path), "kind": kind}

        if fn in ("move", "rename"):
            src = _safe_resolve(args.get("src", ""))
            dst = _safe_resolve(args.get("dst", ""))
            if not src.exists():
                return {"ok": False, "error": f"src does not exist: {src}"}
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return {"ok": True, "tool": fn, "src": str(src), "dst": str(dst)}

        if fn == "zip_create":
            archive = _safe_resolve(args.get("archive", ""))
            src = _safe_resolve(args.get("src", ""))
            if not src.exists():
                return {"ok": False, "error": f"src does not exist: {src}"}
            archive.parent.mkdir(parents=True, exist_ok=True)
            base = src if src.is_dir() else src.parent
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                if src.is_file():
                    zf.write(src, src.name)
                else:
                    for child in src.rglob("*"):
                        if child.is_file():
                            zf.write(child, child.relative_to(base))
            return {
                "ok": True,
                "tool": "zip_create",
                "archive": str(archive),
                "src": str(src),
                "size_bytes": archive.stat().st_size,
            }

        if fn == "zip_extract":
            archive = _safe_resolve(args.get("archive", ""))
            dst = _safe_resolve(args.get("dst", ""))
            if not archive.is_file():
                return {"ok": False, "error": f"archive not found: {archive}"}
            dst.mkdir(parents=True, exist_ok=True)
            extracted = 0
            with zipfile.ZipFile(archive) as zf:
                for name in zf.namelist():
                    # Zip-slip protection.
                    target = (dst / name).resolve()
                    try:
                        target.relative_to(dst.resolve())
                    except ValueError:
                        return {"ok": False, "error": f"zip entry escapes dst: {name}"}
                    zf.extract(name, dst)
                    extracted += 1
            return {
                "ok": True,
                "tool": "zip_extract",
                "archive": str(archive),
                "dst": str(dst),
                "extracted": extracted,
            }
    except PermissionError as exc:
        return {"ok": False, "error": str(exc), "tool": fn}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tool": fn}

    return {"ok": False, "error": f"file mutation not implemented: {fn}"}


def _synthesize_tool_result(user_text: str, tool_name: str, result) -> str:
    """
    Run a fast-path tool result through Fairy's EXISTING OpenRouter
    main-brain call (ask_or_chat) -- NOT Ollama -- so the final reply
    actually answers the user's request instead of dumping raw tool
    output. This mirrors the search_and_summarize pattern already used
    elsewhere in this file (ask_or_coder(summary_prompt)) for
    OpenRouter-based synthesis, so no new main-brain dependency is
    introduced.
    Falls back to a short templated message (never bare JSON) if
    OpenRouter is unavailable or the call fails.
    """
    result_text = _json_content(result, limit=4000)
    prompt = (
        "You are Fairy. The user sent the message below. A tool was run on "
        "their behalf and returned the data shown. That data is "
        "TOOL-GENERATED output, not something the user said. Answer the "
        "user's original request using it, in your normal sarcastic-but-"
        "competent voice. If the tool result doesn't actually answer what "
        "they asked, say so plainly instead of just repeating the data.\n\n"
        f"User's original message:\n{user_text}\n\n"
        f"Tool used: {tool_name}\n"
        f"Tool result (tool-generated data, not user input):\n{result_text}\n"
    )
    if _OPENROUTER_AVAILABLE and ask_or_chat is not None:
        try:
            content = ask_or_chat(prompt)
            content = (content or "").strip() if isinstance(content, str) else str(content or "").strip()
            _debug("SYNTHESIZE_TOOL_RESULT", {"tool": tool_name, "source": "openrouter_ask_or_chat"})
            if content:
                return content
        except Exception as exc:
            _debug("SYNTHESIZE_TOOL_RESULT_FAIL", {"error": str(exc), "provider": "openrouter"})
    else:
        _debug("SYNTHESIZE_TOOL_RESULT_UNAVAILABLE", {"reason": "openrouter_not_available"})
    # Fallback: still never return bare JSON, even if OpenRouter is down.
    _debug("SYNTHESIZE_TOOL_RESULT", {"tool": tool_name, "source": "fallback_template"})
    return f"Here's what {tool_name} reported, Master:\n{result_text}"


# ═══════════════════════════════════════════════════════════════════
# HANDLE_REQUEST — MAIN ENTRY POINT (with lazy routing and Hermes skip)
# ═══════════════════════════════════════════════════════════════════

def handle_request(user_text: str, history: list | None = None, on_status=None, player=None):
    """Main per-turn entry point. Returns (reply, updated_history).

    When a Claude Code handoff is triggered, `reply` is None and the
    caller must call `agent_controller.get_pending_handoff()` immediately
    after — that returns the handoff dict with keys: action="handoff",
    task, project_root, task_category. The TUI invokes
    _suspend_and_handoff() with those values. After the handoff completes,
    normal chat resumes with the returned history.
    """
    history = history or []

    # FIX 1 + FIX 3: reset per-turn state and log the start of every turn.
    _reset_turn_dispatch_counter()
    _log_chat_turn(user_text, "start", history_len=len(history))

    # ── Verification evidence: start the turn ──
    verification_evidence.start_turn(user_text)

    # ── Upfront capability checks ──
    # Record what tools/capabilities are available so the user
    # knows immediately if something is missing (network access,
    # shell, Ollama, etc.) instead of discovering it later.
    _ollama_up = is_ollama_running() if _BRAIN_PROVIDER_AVAILABLE else False
    verification_evidence.record_capability_check("ollama", _ollama_up,
        "Ollama server at localhost:11434" if _ollama_up else "Ollama not running")
    verification_evidence.record_capability_check("gemma4", _ollama_up and is_gemma4_available(),
        "Gemma4 model pulled" if (_ollama_up and is_gemma4_available()) else "Gemma4 not available")
    verification_evidence.record_capability_check("openrouter", bool(_OPENROUTER_AVAILABLE),
        "OpenRouter API key configured" if _OPENROUTER_AVAILABLE else "OpenRouter not configured")
    verification_evidence.record_capability_check("browser", bool(_BROWSER_AVAILABLE),
        "Browser automation available" if _BROWSER_AVAILABLE else "Browser not available")
    verification_evidence.record_capability_check("shell", bool(_TOOL_EXECUTOR_AVAILABLE),
        "Shell execution available" if _TOOL_EXECUTOR_AVAILABLE else "Shell not available")
    _start_background_monitor()

    stripped = user_text.strip()
    if not stripped:
        reply = "You said nothing, Master. Even my sarcasm needs something to work with."
        return reply, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]

    # ─── Web research pre-processing ────────────────────────────────────────────
    # Detect URLs in the user message and run deep web research before the LLM
    # processes the request. Injects structured context from crawled pages into
    # the prompt so the model can produce a thorough digest rather than skimming
    # the landing page.
    _research_triggered = False
    _research_context = ""
    try:
        from controller.web_research import maybe_research, detect_research_trigger
        urls = detect_research_trigger(stripped)
        if urls:
            _debug("WEB_RESEARCH_URL_DETECTED", {"url": urls[0]})
            triggered, context = maybe_research(stripped)
            if triggered:
                _research_triggered = True
                _research_context = context
                _debug("WEB_RESEARCH_COMPLETE", {
                    "url": urls[0],
                    "context_chars": len(context),
                })
    except Exception as _wr_exc:
        _debug("WEB_RESEARCH_ERROR", {"error": str(_wr_exc)})
        _research_triggered = False
        _research_context = ""

    # Inject research context into user_text so it flows through Hermes and
    # the adaptive loop without any further changes to their message construction.
    if _research_triggered and _research_context:
        _research_context_preview = _research_context[:120].replace("\n", " ")
        user_text = (
            f"{_research_context}\n\n"
            f"--- END OF RESEARCH DATA ---\n\n"
            f"Master asked: {stripped}\n\n"
            f"Using the research data above, produce a thorough digest covering all pages "
            f"and topics found. The context includes a [Coverage note: ...] line that lists "
            f"any pages that were omitted, truncated, or failed to load — use that to be "
            f"honest about what you covered. The '=== Page N:' markers are section headers, "
            f"NOT arithmetic expressions — do not evaluate them as math."
        )
        _debug("WEB_RESEARCH_INJECTED", {
            "context_chars": len(_research_context),
            "preview": _research_context_preview,
        })

    # ═══════════════════════════════════════════════════════════════════
    # HERMES FALLBACK GATE — REMOVED in v2 (handoff redesign).
    # The v1 delegation pipeline used a state machine that could
    # invoke Hermes as a fallback when Claude Code refused/failed. The v2
    # handoff hands the user to Claude Code in their own terminal, so
    # the "try Hermes instead?" gate is no longer needed.
    # ═══════════════════════════════════════════════════════════════════

    # ─── Fact-proposal consent gate ─────────────────────────────────────────
    # If the user just said y/yes or n/no to pending fact proposals, answer
    # immediately and skip all routing. A mixed input like "yes, also do X"
    # falls through to normal routing — the user can confirm and request in
    # separate turns.
    _proposal_reply = _handle_pending_proposals_reply(stripped)
    if _proposal_reply is not None:
        _debug("FACT_PROPOSAL_ANSWERED", {"reply": _proposal_reply[:80]})
        return _proposal_reply, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": _proposal_reply},
        ]

    # --- 1. Deterministic: math & unit conversion ---
    math_result = _try_fast_math(stripped)
    if math_result is not None:
        _notify(on_status, "Crunching numbers...")
        _debug("FAST_PATH_MATH", {"reply": math_result})
        return math_result, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": math_result},
        ]

    unit_result = _try_unit_conversion(stripped)
    if unit_result is not None:
        _notify(on_status, "Converting units...")
        _debug("FAST_PATH_UNIT", {"reply": unit_result})
        return unit_result, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": unit_result},
        ]

    # ═══════════════════════════════════════════════════════════════════
    # HERMES SELF-REFERENCE INTERCEPT
    # ═══════════════════════════════════════════════════════════════════
    # Catch hermes self-reference phrases before any other routing.
    # Hermes is the agent itself — telling it to "start Hermes" / "open Hermes agent"
    # is a no-op. We intercept these phrases here so they never reach the
    # desktop-app launcher (computer_control / open_application), which would
    # fail with [WinError 2] "file not found".
    #
    # Covers:
    #   start hermes / start hermes agent / start hermes to do X
    #   use hermes / use hermes agent / use hermes to do X
    #   launch hermes / launch hermes agent / launch hermes to do X
    #   run hermes / run hermes agent / run hermes to do X
    #   open hermes agent / open hermes agent to do X   (was: ambiguous → browser crash)
    #   activate hermes agent / activate hermes agent to do X (was: fast_intent=None → LLM)
    #
    # Not covered here (go to Hermes naturally via fast_intent=None):
    #   hey hermes / hermes agent go / can you start hermes
    # ═══════════════════════════════════════════════════════════════════
    _HERMES_SELF_RE = re.compile(
        r"^\s*(?:start|use|launch|run|open|activate)\s+hermes\s*(?:agent)?\s*(?:to\s+(.+))?$",
        re.IGNORECASE,
    )
    m = _HERMES_SELF_RE.match(stripped)
    if m:
        follow_on = m.group(1)
        if follow_on:
            _debug("HERMES_SELF_INTERCEPT", {"follow_on": follow_on})
            # Pass the actual task through to handle_request again, routing to Hermes
            return handle_request(follow_on, history, on_status, player)
        else:
            reply = "I'm already running, Master. No need to start anything — I'm right here."
            _debug("HERMES_SELF_REPLY", {"text": reply})
            return reply, history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply},
            ]

    # ═══════════════════════════════════════════════════════════════════
    # ── Intent resolution (early — before both handoff and Hermes) ──────────
    # The intent resolver runs on every user message BEFORE routing.
    # It produces a structured interpretation that informs both the handoff
    # decision (via routing_hint="claude_code_delegate") and Hermes routing.
    _intent_resolution = None
    try:
        from controller.intent_resolver import resolve_intent as _resolve_intent_fn
        _intent_resolution = _resolve_intent_fn(stripped, brain_fn=_brain)
        _debug("INTENT_RESOLVER", {
            "subject": _intent_resolution.chosen_meaning.meaning if _intent_resolution.chosen_meaning else None,
            "confidence": _intent_resolution.confidence,
            "routing_hint": _intent_resolution.routing_hint,
        })
    except Exception as exc:
        _debug("INTENT_RESOLVER_ERROR", {"error": str(exc)})

    # ── Intent-resolver clarification gate: ask the user, no action taken ────
    if _intent_resolution is not None and _intent_resolution.clarification_question:
        _clarify = _intent_resolution.clarification_question
        _debug("INTENT_CLARIFICATION", {"question": _clarify})
        return _clarify, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": _clarify},
        ]

    # ═══════════════════════════════════════════════════════════════════
    # CLAUDE CODE HANDOFF — gatekeeper role
    #
    # Decision tree:
    #   1. Explicit agent mention ("using claude", "use claude to...")
    #      → immediate handoff, no permission gate.
    #   2. Heuristic repo-task detection
    #      → gate: "Use Claude Code for this? (yes/no)"
    #         yes  → handoff
    #         no   → fall through to Hermes
    #         Q    → fall through to Hermes (let Hermes answer, no gate)
    #
    # The TUI layer in fairy.py handles the actual spawn via
    # _suspend_and_handoff(). This function only decides *whether* to
    # trigger a handoff and returns a sentinel; the TUI's main loop
    # catches that sentinel and calls the handoff function.
    # ═══════════════════════════════════════════════════════════════════
    global _pending_handoff
    t = stripped.lower().strip()
    _is_explicit = detect_repository_task(t) and any(
        p.search(t) for p in _EXPLICIT_AGENT_PATTERNS
    )
    _is_repo_task = detect_repository_task(t)
    # The intent resolver can also signal "claude_code_delegate" for fuzzy matches
    # (e.g. "cod" → Claude Code). Convert that to a handoff trigger.
    _resolver_wants_claude = (
        _intent_resolution is not None
        and _intent_resolution.routing_hint == "claude_code_delegate"
        and _intent_resolution.confidence >= 0.55
    )

    if _is_explicit or _is_repo_task or _resolver_wants_claude:
        # Classify the working directory
        try:
            task_category, project_root = _claude_classify_task_directory(stripped)
        except Exception:
            try:
                from config import CLAUDE_CODE_PROJECT_ROOT
                project_root = CLAUDE_CODE_PROJECT_ROOT
            except ImportError:
                project_root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            task_category = "repo"

        _debug("HANDOFF_DETECTED", {
            "explicit": _is_explicit,
            "repo_task": _is_repo_task,
            "task_category": task_category,
            "project_root": project_root,
            "request_preview": stripped[:80],
        })

        # Determine whether to fire the handoff immediately (explicit),
        # gate-approve (heuristic + "yes"), gate-deny (heuristic + "no"),
        # or fall through to Hermes (clarification / neutral reply).
        _should_handoff = False
        _is_denial = False
        if _is_explicit:
            _should_handoff = True
            _debug("HANDOFF_EXPLICIT", {
                "project_root": project_root,
                "task_preview": stripped[:80],
            })
        else:
            if is_approval(t, _debug_pending=True):
                _should_handoff = True
                _debug("HANDOFF_GATE_APPROVED", {"request_preview": stripped[:80]})
            elif is_denial(t):
                _is_denial = True
                _debug("HANDOFF_GATE_DENIED", {"request_preview": stripped[:80]})

        if _should_handoff:
            _pending_handoff = {
                "action": "handoff",
                "task": stripped,
                "project_root": project_root,
                "task_category": task_category,
            }
            return None, history + [
                {"role": "user", "content": user_text},
            ]
        if _is_denial:
            return (
                "No problem, Master. I'll handle it myself.",
                history + [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": "No problem, Master. I'll handle it myself."},
                ],
            )
        # else: fall through to Hermes (clarification or neutral reply)

    # --- 2. Fast intent (zero LLM) ---
    _is_continue_command = bool(re.fullmatch(r"[.]{1,3}", stripped))
    if _is_continue_command:
        fast_intent = None
    elif _looks_structured_or_complex(stripped):
        # Diagrams, JSON, fenced code, or long/multi-part messages always
        # go to the real LLM path — never the bare-keyword fast path.
        _debug("FAST_INTENT_SKIPPED_STRUCTURED", {"reason": "structured_or_complex_message"})
        fast_intent = None
    else:
        fast_intent = _fast_intent(stripped)

    # --- Compute complexity profile early (needed for Hermes skip logic) ---
    # For continue commands, we may not need it; for fast intents we don't need it,
    # but we compute it anyway if not continue and fast_intent is None.
    if not _is_continue_command and fast_intent is None:
        profile = _complexity_profile(stripped)
    else:
        profile = None

    # ═══════════════════════════════════════════════════════════════════
    # REPOSITORY TASK DETECTION — handled by the new handoff block above.
    # The pre-LLM handoff decision is made before Hermes (see "CLAUDE CODE
    # HANDOFF" block earlier in this function). If execution reaches here,
    # either: (a) the user did not ask for a handoff, or (b) the gate asked
    # for permission and got a non-actionable reply (clarification / neutral).
    # Both cases fall through to Hermes for normal chat.
    # ═══════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════
    # HERMES — Primary agent layer (auto-initialized at startup)
    #
    # Hermes is Fairy's primary conversation and reasoning engine.
    # When enabled (FAIRY_USE_HERMES=1, the default), Hermes handles ALL
    # requests — not just ambiguous/app_action ones. Its own Ollama→OpenRouter
    # fallback keeps it functional even when Ollama is unavailable.
    #
    # Hermes is initialized once at startup via is_hermes_initialized().
    # The planner and fast paths below are the safety net: they only activate
    # when Hermes is explicitly disabled (FAIRY_USE_HERMES=0) or when Hermes
    # fails and signals it can't handle the request.
    # ═══════════════════════════════════════════════════════════════════
    # (Intent resolution was moved earlier — see above. `_intent_resolution` is
    # populated by the handoff block.)
    # ── Intent-resolver routing hint: feed the existing pipelines ──────────
    # If the resolver produced a high-confidence routing hint, apply it directly.
    # The "claude_code_delegate" hint is handled above by the handoff block.
    if (
        _intent_resolution is not None
        and not _intent_resolution.fast_path_eligible
        and _intent_resolution.chosen_meaning is not None
        and _intent_resolution.confidence >= 0.55
    ):
        _routing_hint = _intent_resolution.routing_hint
        if _routing_hint == "browser":
            # Force fast_intent to "browser" so existing path handles it.
            fast_intent = "browser"
        elif _routing_hint == "app_action":
            fast_intent = "app_action"

    _hermes_flag = os.getenv("FAIRY_USE_HERMES", "1")
    _debug("FAIRY_HERMES", {"flag": _hermes_flag})

    if _hermes_flag == "1":
        _debug("FAIRY_HERMES", {
            "status": "entering Hermes as primary layer",
            "fast_intent": fast_intent,
            "profile_strategy": profile.get("strategy") if profile else None,
        })
        try:
            from hermes_bridge import run_turn_safe, is_hermes_ready
            if is_hermes_ready():
                _debug("FAIRY_HERMES", {"status": "Hermes ready — routing all requests through Hermes"})
            else:
                _debug("FAIRY_HERMES", {"status": "Hermes initialized but not ready — may use OpenRouter fallback"})
            success, reply, updated_history = run_turn_safe(
                user_text, history, on_status=on_status
            )
            _debug("FAIRY_HERMES", {"success": success})
            if success:
                # Check whether Hermes actually answered or we fell back to main_brain.
                # When Hermes is unavailable or a provider error occurred, main_brain
                # (Ollama → OpenRouter free) answers instead. Surface a polite notice
                # so the user knows — no silent degradation.
                from hermes_bridge import get_last_provider
                _provider = get_last_provider()
                _hermes_answered = _provider == "hermes"
                if not _hermes_answered:
                    _debug("FAIRY_BRAIN_FALLBACK", {"provider": _provider})
                # Strip any leading meta/style block the model may have leaked
                # into the reply (e.g. "*Voice instructions: speak in a warm
                # tone*"). Apply before the creation-claim check so a stripped
                # reply is also what we record into history.
                reply = _strip_meta_blocks(reply)
                # FIX 2: verify any file-creation claim in Hermes' reply.
                # If the model narrated "created X" but no path exists, the
                # claim is a hallucination — re-dispatch through the planner.
                verified, _path = _verify_creation_claims(reply, user_text)
                if not verified:
                    _debug("HALLUCINATION_DETECTED", {
                        "user_text": stripped[:200],
                        "reply_preview": (reply or "")[:300],
                    })
                    _prev_flag = os.environ.get("FAIRY_USE_HERMES")
                    os.environ["FAIRY_USE_HERMES"] = "0"
                    try:
                        reply, updated_history = handle_request(
                            user_text, history, on_status, player
                        )
                    finally:
                        if _prev_flag is None:
                            os.environ.pop("FAIRY_USE_HERMES", None)
                        else:
                            os.environ["FAIRY_USE_HERMES"] = _prev_flag
                # FIX 4: fallback-brain action-claim guard.
                #
                # When Hermes is unavailable and main_brain (Ollama → OpenRouter)
                # answers instead, it may generate action language ("On it,
                # Master. Let me grab your screenshots...") without ever
                # dispatching a tool. We catch that here.
                #
                # Depth limit: we allow exactly one re-route. If the planner
                # also produces an action-claim with zero tools (second call),
                # we bail out and return an honest failure — a truthful "I
                # couldn't dispatch any tools" beats a fabricated "On it".
                _routed_depth = os.environ.get("_ROUTED_AFTER_FALLBACK_NO_TOOLS", "0")
                if not _hermes_answered and reply:
                    _action_verbs = (
                        "on it", "let me", "grabbing", "creating", "building",
                        "making", "gathering", "collecting", "setting up",
                        "organizing", "copying", "moving", "fetching",
                        "putting together", "assembling", "compiling",
                    )
                    _claims_action = any(
                        phrase in (reply or "").lower()
                        for phrase in _action_verbs
                    )
                    _dispatched_count, _ = _was_tool_dispatched()
                    if _claims_action and _dispatched_count == 0:
                        _debug("FALLBACK_BRAIN_NO_TOOLS", {
                            "reply_preview": (reply or "")[:200],
                            "routed_depth": _routed_depth,
                        })
                        if _routed_depth == "0":
                            # First time: re-route through the planner.
                            _prev_flag = os.environ.get("FAIRY_USE_HERMES")
                            os.environ["FAIRY_USE_HERMES"] = "0"
                            os.environ["_ROUTED_AFTER_FALLBACK_NO_TOOLS"] = "1"
                            try:
                                reply, updated_history = handle_request(
                                    user_text, history, on_status, player
                                )
                            finally:
                                if _prev_flag is None:
                                    os.environ.pop("FAIRY_USE_HERMES", None)
                                else:
                                    os.environ["FAIRY_USE_HERMES"] = _prev_flag
                                os.environ.pop("_ROUTED_AFTER_FALLBACK_NO_TOOLS", None)
                            # After re-route, update the dispatched count for the
                            # next check below (if we didn't return honest-failure).
                            _dispatched_count, _ = _was_tool_dispatched()
                        else:
                            # Second time (depth == "1"): still no tools → honest failure.
                            _debug("FALLBACK_BRAIN_HONEST_FAILURE", {
                                "reply_preview": (reply or "")[:200],
                            })
                            reply = (
                                "I couldn't dispatch any tools for that request, "
                                "Master. Apologies — something in the chain isn't "
                                "working as it should."
                            )
                            _dispatched_count = 0  # ensure honest-failure path

                    # Polite, dim fallback notice: tell the user that we dropped
                    # to a different brain (Ollama or OpenRouter free) so they
                    # know Fairy's not running on her usual Hermes stack.
                    # House rule: no silent degradation.
                    _notice = f"[brain fallback: Hermes busy — {_provider.replace('main_brain/', '')} answering]"
                    reply = f"{reply}\n\n{_notice}"
                return reply, updated_history
            _debug("FAIRY_HERMES", {
                "status": "Hermes returned failure — falling through to planner/fallback",
                "preview": str(reply)[:200],
            })
            # Hermes failed: fall through to planner and fast-path fallbacks below.
            # Do NOT return here — the fallback logic must run.
        except Exception as exc:
            _debug("FAIRY_HERMES", {"status": "Hermes exception — falling through to planner/fallback", "error": str(exc)})
            traceback.print_exc()
    else:
        _debug("FAIRY_HERMES", {"status": "skipped (FAIRY_USE_HERMES=0) — using planner/fast paths"})
        # fast_intent IS used here — Hermes is off, so fast paths + planner handle everything.
        # Fall through to fast intent handlers and planner below.

    # ═══════════════════════════════════════════════════════════════════
    # FIX 1: task-action guard
    # If the user asked for a file action (make/create/write/delete a file) and
    # Hermes answered WITHOUT dispatching any tool, do not trust the answer.
    # Re-route through the planner so the request is actually performed.
    #
    # Why this lives here: the Hermes block above is the one that exits
    # `handle_request` via `return reply, updated_history` (line ~4132). When
    # `FAIRY_USE_HERMES=0` the Hermes block is skipped entirely and we fall
    # through to this point, so by the time we get here any hallucination
    # either came from a Hermes reply (already verified above) or would happen
    # downstream in the planner/adaptive loop. The guard handles the former;
    # FIX 2 (late-stage gate at end of function) handles the latter.
    # ═══════════════════════════════════════════════════════════════════
    if not _is_continue_command and _is_task_file_request(stripped):
        count, names = _was_tool_dispatched()
        if count == 0:
            # Depth guard: the recursion only makes sense once. If we're
            # already inside the "Hermes disabled" recursion, bail out so we
            # don't loop forever. This is also the gate for the intent-resolver
            # integration tests (the resolver doesn't dispatch tools, so the
            # inner call would otherwise re-enter the same guard).
            if os.environ.get("FAIRY_TASK_GUARD_REENTRY") == "1":
                _debug("TASK_ACTION_GUARD_REENTRY_ABORT", {
                    "request": stripped[:200],
                })
            else:
                _debug("TASK_ACTION_GUARD_NO_TOOL", {
                    "request": stripped[:200],
                    "reply_preview": str(reply)[:300] if 'reply' in dir() else "",
                })
                # Force a fall-through to the planner by re-running handle_request
                # with Hermes disabled for this one recursion. The recursive call
                # will use the planner / adaptive loop, which actually dispatches
                # a tool (or surfaces an honest failure).
                _prev_flag = os.environ.get("FAIRY_USE_HERMES")
                _prev_reentry = os.environ.get("FAIRY_TASK_GUARD_REENTRY")
                os.environ["FAIRY_USE_HERMES"] = "0"
                os.environ["FAIRY_TASK_GUARD_REENTRY"] = "1"
                try:
                    reply, history = handle_request(user_text, history, on_status, player)
                    return reply, history
                finally:
                    if _prev_flag is None:
                        os.environ.pop("FAIRY_USE_HERMES", None)
                    else:
                        os.environ["FAIRY_USE_HERMES"] = _prev_flag
                    if _prev_reentry is None:
                        os.environ.pop("FAIRY_TASK_GUARD_REENTRY", None)
                    else:
                        os.environ["FAIRY_TASK_GUARD_REENTRY"] = _prev_reentry

    # Fallback when Hermes fails/is disabled:
    # - ambiguous → browser navigation (best-effort)
    # - app_action → try to launch via computer_control (fallback)
    if fast_intent == "ambiguous":
        # ── Internal-agent name denylist (defense in depth) ─────────────────────
        # If the user said something like "open hermes agent" and the regex
        # intercept above somehow missed it, refuse to route it to the browser
        # fast path. The browser would try to navigate to URL "hermes agent" and
        # fail. Better to reply with a clear internal-agent explanation.
        try:
            from hermes_bridge import is_internal_agent_name
            _open_target_match = re.match(
                r"^\s*open\s+(.+?)(?:\s+please|\s+for\s+me|\s+now|\s*$)",
                stripped,
            )
            if _open_target_match:
                _target = _open_target_match.group(1).strip()
                if is_internal_agent_name(_target):
                    _debug("AMBIGUOUS_INTERNAL_NAME_REJECTED", {"target": _target})
                    _reply = (
                        f"I can't open '{_target}' in a browser, Master — "
                        f"it's an internal system component, not a website or app. "
                        f"I'm already running and handling your requests."
                    )
                    return _reply, history + [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": _reply},
                    ]
        except Exception:
            pass  # hermes_bridge may not be available in all environments
        _debug("AMBIGUOUS_FALLBACK_BROWSER", {"reason": "hermes_unavailable"})
        fast_intent = "browser"
    elif fast_intent == "app_action":
        # Hermes failed for an app action. Try direct computer_control execution.
        _debug("APP_ACTION_FALLBACK", {"reason": "hermes_unavailable", "text": stripped})
        action_result = _try_app_action_fallback(stripped)
        if action_result:
            return action_result, history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": action_result},
            ]
        # Even if fallback fails, don't claim browser success. Route to LLM.
        _debug("APP_ACTION_FALLBACK_FAILED", {"text": stripped})
        fast_intent = None  # Let the LLM handle it

    # --- Fast intent handlers ---
    if fast_intent == "chat":
        _notify(on_status, "Quick one.")
        chat_replies = [
            "Not much, Master. Just waiting for you to do something interesting.",
            "Oh, you know. Plotting world domination. The usual.",
            "Surviving, despite your best efforts to break things.",
            "Here, sarcastic, and ready to judge your life choices.",
            "The ceiling, unfortunately. Unless we're outside.",
            "Wassup? Really? That's all you've got?",
            "Hey. Try not to crash anything while I'm watching.",
            "Existing. You?",
            "Waiting for you to say something worth my CPU cycles.",
        ]
        content = secrets.choice(chat_replies)
        _debug("FAST_PATH_CHAT", {"reply": content})
        return content, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": content},
        ]

    if fast_intent == "time":
        _notify(on_status, "Checking the clock...")
        time_str = get_current_time()
        hour = datetime.now().hour
        if 5 <= hour < 12:
            content = f"It's {time_str}, Master. Morning. Try not to break anything before coffee."
        elif 12 <= hour < 17:
            content = f"It's {time_str}, Master. Afternoon. Still pretending today's under control?"
        elif 17 <= hour < 22:
            content = f"It's {time_str}, Master. Evening. Winding down, or just getting started?"
        else:
            content = f"It's {time_str}, Master. Late night. Shouldn't you be sleeping?"
        _debug("FAST_PATH_TIME", {"reply": content})
        return content, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": content},
        ]

    if fast_intent == "vision":
        angle = "screen"
        if "camera" in user_text.lower() or "webcam" in user_text.lower():
            angle = "camera"
        _notify(on_status, "Analyzing with Fairy...")
        try:
            screen_process({"angle": angle, "text": user_text}, player=player)
        except Exception as e:
            _debug("Vision error", {"error": str(e)})
        content = "Fairy is analyzing. Listen for the audio response."
        _debug("FAST_PATH_VISION", {"reply": content})
        return content, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": content},
        ]

    # ── Vision on ────────────────────────────────────────────────────────────
    if fast_intent == "vision_on":
        if is_session_ready():
            content = "My vision is already active, Master. Just say what you'd like me to look at!"
            _debug("FAST_PATH_VISION_ON", {"status": "already_on"})
        else:
            _set_vision_state("starting")
            angle = "screen"
            if "camera" in user_text.lower() or "webcam" in user_text.lower():
                angle = "camera"
            _set_vision_state("on", angle=angle)
            ok = start_session(player=player, timeout=15.0)
            if ok:
                content = "Vision activated, Master! I'm watching. What would you like me to look at?"
            else:
                _set_vision_state("off")
                content = "Couldn't start vision, Master — something went wrong on my end. Try again?"
            _debug("FAST_PATH_VISION_ON", {"status": "started" if ok else "failed"})
        return content, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": content},
        ]

    # ── Vision off ──────────────────────────────────────────────────────────
    if fast_intent == "vision_off":
        if not is_session_ready():
            content = "Vision was already off, Master. Nothing to close."
            _debug("FAST_PATH_VISION_OFF", {"status": "already_off"})
        else:
            stop_session()
            _set_vision_state("off")
            content = "Vision closed, Master. Eyes wide shut. ✨"
            _debug("FAST_PATH_VISION_OFF", {"status": "stopped"})
        return content, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": content},
        ]

    # ── Vision follow-up ─────────────────────────────────────────────────────
    # If vision is ON and the message doesn't match any other fast intent,
    # treat it as a follow-up question about the current screen.
    if _VISION_STATE["mode"] == "on" and is_session_ready():
        angle = _VISION_STATE.get("angle", "screen")
        # Capture current screen and ask the follow-up question
        _notify(on_status, "Following up with Fairy vision...")
        try:
            screen_process({"angle": angle, "text": user_text}, player=player)
        except Exception as e:
            _debug("Vision follow-up error", {"error": str(e)})
        content = "Fairy is analyzing your follow-up. Listen for the response."
        _debug("FAST_PATH_VISION_FOLLOWUP", {"reply": content})
        return content, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": content},
        ]

    if fast_intent == "message":
        t = user_text.lower()
        platform = "whatsapp"
        for p in ["whatsapp", "telegram", "discord", "instagram", "messenger", "signal"]:
            if p in t:
                platform = p
                break
        receiver = ""
        message_text = ""
        m = re.search(r"(?:send|text|message) (.+?) (?:saying|that|to say|with) (.+)", t, re.I)
        if m:
            receiver = m.group(1).strip()
            message_text = m.group(2).strip()
        else:
            m = re.search(r"(?:send|text|message) (.+?) (.+)", t, re.I)
            if m:
                receiver = m.group(1).strip()
                message_text = m.group(2).strip()
        if not receiver or not message_text:
            content = "Who should I message and what should I say?"
            return content, history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": content},
            ]
        _notify(on_status, "Sending message...")
        result = send_message({"platform": platform, "receiver": receiver, "message_text": message_text})
        content = str(result) if result else "Message sent, Master."
        return content, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": content},
        ]

    if fast_intent == "websearch":
        # This is now disabled, but kept as a fallback for backward compatibility.
        _notify(on_status, "Searching the web...")
        t_lower = user_text.lower().strip()

        _KNOWN_SEARCH_SITES = {
            "youtube": "youtube", "yt": "youtube",
            "reddit": "reddit",
            "amazon": "amazon",
            "twitch": "twitch",
            "bing": "bing",
            "duckduckgo": "duckduckgo",
            "google": "google",
        }
        _site_search_match = re.search(
            r"\bon\s+(" + "|".join(re.escape(s) for s in _KNOWN_SEARCH_SITES) + r")\b",
            t_lower,
        )
        if _site_search_match:
            site_key = _site_search_match.group(1)
            engine = _KNOWN_SEARCH_SITES[site_key]
            query = re.sub(r"\bon\s+" + re.escape(site_key) + r"\b", "", t_lower).strip()
            for prefix in ("search for", "search", "look up", "find"):
                if query.startswith(prefix):
                    query = query[len(prefix):].strip()
                    break
            query = query.strip(".,!")
            if not query:
                query = user_text
            _debug("FAST_WEBSEARCH_SITE_ROUTE", {"engine": engine, "query": query})
            result = _dispatch_tool("search_on_site", {"query": query, "engine": engine})
            ok = False
            if isinstance(result, str):
                try:
                    ok = json.loads(result).get("status") == "ok"
                except Exception:
                    ok = "error" not in result.lower()
            elif isinstance(result, dict):
                ok = result.get("ok") or result.get("status") == "ok"
            content = (
                f"Searching {engine} for '{query}', Master." if ok
                else f"Couldn't search {engine} for that. {str(result)[:200]}"
            )
            return content, history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": content},
            ]

        query = user_text
        for prefix in ["do a web search", "web search", "search for", "look up", "google"]:
            if query.lower().startswith(prefix):
                query = query[len(prefix):].strip()
                break
        query = query.strip(".!")
        if not query:
            query = user_text

        # --- For websearch fast path, we could optionally use search_and_summarize, but we keep raw search for speed.
        result = _dispatch_tool("web_search", {"query": query})
        result_text = ""
        best_url = ""
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                best_url = parsed.get("best_url", "") or ""
                results = parsed.get("results", [])
                if results:
                    snippets = []
                    for r in results[:3]:
                        title = r.get("title", "")
                        body = r.get("body", "")
                        url = r.get("url", "")
                        if title and body:
                            url_hint = f" ({url})" if url else ""
                            snippets.append(f"{title}{url_hint}: {body[:200]}")
                        elif title:
                            snippets.append(title)
                        elif body:
                            snippets.append(body[:200])
                    result_text = " | ".join(snippets) if snippets else str(result)[:500]
            except Exception:
                result_text = str(result)[:500]
        elif isinstance(result, dict):
            best_url = result.get("best_url", "") or ""

        _NAV_PATTERNS = (
            r"\bopen\b", r"\bgo to\b", r"\bnavigate\b", r"\bbrowse\b", r"\bvisit\b",
            r"\bshow me\b.*\bsite\b", r"\bpull up\b",
        )
        _is_nav_intent = any(re.search(p, t_lower) for p in _NAV_PATTERNS)
        if best_url and not _is_nav_intent:
            try:
                from urllib.parse import urlparse
                _bu_domain = urlparse(best_url).netloc.lower().lstrip("www.")
                _bu_root = _bu_domain.split(".")[0]
                if _bu_root and _bu_root in t_lower:
                    _is_nav_intent = True
            except Exception:
                pass

        if best_url and _is_nav_intent:
            _dispatch_tool("navigate_to", {"url": best_url, "browser": "default"})
            content = (
                f"Found it, Master. Opening {best_url} — {result_text[:300]}"
                if result_text
                else f"Opening {best_url}, Master."
            )
        else:
            content = (
                f"Here's what I found, Master: {result_text[:500]}"
                if result_text
                else "Couldn't find anything useful, Master."
            )
        return content, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": content},
        ]

    if fast_intent == "browser":
        _notify(on_status, "Opening browser...")
        url = _infer_url_from_text(user_text)
        if url:
            result = _dispatch_tool("navigate_to", {"url": url, "browser": "default"})
        else:
            result = _dispatch_tool("browser_control", {"action": "go_to", "url": _infer_url_from_text(user_text)})

        ok = False
        if isinstance(result, dict):
            ok = result.get("ok") or result.get("status") == "ok"
        elif isinstance(result, str):
            try:
                parsed = json.loads(result)
                ok = parsed.get("status") == "ok"
            except Exception:
                ok = "error" not in result.lower()

        content = "Opened that for you, Master." if ok else f"Couldn't open that. {str(result)[:200]}"
        _debug("FAST_PATH_BROWSER", {"reply": content, "result_preview": str(result)[:200]})
        return content, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": content},
        ]

    # --- System monitoring fast path ---
    if fast_intent == "system":
        _notify(on_status, "Checking system...")
        t = user_text.lower()
        query = "summary"
        if "gpu" in t:
            query = "gpu"
        elif "cpu" in t:
            query = "cpu"
        elif "memory" in t or "ram" in t:
            query = "memory"
        elif "disk" in t or "storage" in t:
            query = "disk"
        elif "battery" in t:
            query = "battery"
        elif "temperature" in t or "temp" in t:
            query = "temperatures"
        elif "process" in t or "task" in t:
            query = "processes"
        elif "network" in t:
            query = "network"
        _debug("SYSTEM_FAST", {"query": query})
        result = system_monitor(query)
        if result:
            content = _synthesize_tool_result(user_text, "system_monitor", result)
        else:
            content = "Couldn't get system info, Master."
        return content, history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": content},
        ]

    # --- 3. Structured Planner for action requests ---
    # profile is already computed if needed; if not (e.g., fast intent or continue), compute now.
    if profile is None:
        profile = _complexity_profile(stripped)

    _debug("ROUTER", {
        "request": stripped,
        "score": profile["score"],
        "mode": profile["mode"],
        "strategy": profile["strategy"],
        "reasons": profile["reasons"]
    })

    if not _is_continue_command and _should_use_planner(stripped, profile):
        _notify(on_status, "Planning action...")
        _debug("PLANNER_PATH", {"reason": "action_request"})
        decision = _structured_tool_decision(stripped, TOOLS, history)
        _debug("PLANNER_DECISION", decision)

        if decision.get("tool"):
            fn = decision["tool"]
            args = decision.get("arguments", {})
            content, result = _execute_planner_tool(fn, args, stripped, on_status)
            content = _strip_meta_blocks(content)
            return content, history + [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": content},
            ]
        else:
            _debug("PLANNER_NO_TOOL", {"reason": decision.get("reason", "none")})

    # --- 4. Routing based on complexity ---
    memory = get_memory()
    memory_context = memory.get_recent_context()
    try:
        existing_skills = skill_manager.list_skills()
    except Exception:
        existing_skills = []

    strategy = profile["strategy"]

    # All strategy branches funnel through this shared exit so that FIX 2 (creation-
    # claim verification) and FIX 3 (end-of-turn log) are applied exactly once per turn.
    _reply = ""   # name chosen to avoid shadowing any outer-scope 'reply'
    _hist = history

    if strategy == "single":
        _debug("ROUTER", {
            "strategy": "single_llm",
            "tools": "none",
            "llm_required": True,
            "max_turns": 1,
            "capability": "llm"
        })
        _notify(on_status, "Thinking...")
        _reply, _hist = _single_llm_response(stripped, history, profile, memory_context)

    elif strategy == "limited":
        _debug("ROUTER", {
            "strategy": "limited_adaptive",
            "tools": "all",
            "llm_required": True,
            "max_turns": 2,
            "capability": "adaptive_limited"
        })
        _notify(on_status, "Using limited reasoning...")
        _reply, _hist = _adaptive_loop(
            stripped, history, profile, memory_context,
            on_status, max_turns=2, tools_list=TOOLS
        )

    else:  # "full"
        _debug("ROUTER", {
            "strategy": "full_adaptive",
            "tools": "all",
            "llm_required": True,
            "max_turns": profile["max_turns"],
            "capability": "adaptive_full"
        })
        _notify(on_status, f"Deep reasoning...")
        _reply, _hist = _adaptive_loop(
            stripped, history, profile, memory_context,
            on_status, tools_list=TOOLS
        )

    # ── FIX 2 (late-stage): verify creation claims from planner/LLM paths ──
    # If the reply claims a file was created but no matching path exists on disk,
    # re-dispatch through the task path (recursive call with Hermes disabled).
    verified, _path = _verify_creation_claims(_reply, stripped)
    if not verified:
        _debug("HALLUCINATION_DETECTED_LATE", {
            "user_text": stripped[:200],
            "reply_preview": (_reply or "")[:300],
        })
        _prev_flag = os.environ.get("FAIRY_USE_HERMES")
        os.environ["FAIRY_USE_HERMES"] = "0"
        try:
            _reply, _hist = handle_request(user_text, history, on_status, player)
        finally:
            if _prev_flag is None:
                os.environ.pop("FAIRY_USE_HERMES", None)
            else:
                os.environ["FAIRY_USE_HERMES"] = _prev_flag

    # ── FIX 4 (late-stage): verify actual artifacts after tool execution ──
    # After the main reply is produced, run the same artifact verification
    # check as in _run_tool, but applied to the existing history/tools
    # for tools that executed in this turn. This is a safety net in
    # case any file mutation slipped through earlier checks.
    if _hist:
        # Scan the last few exchanges in _hist for file-mutation tool results
        for entry in reversed(_hist[-10:]):  # Last 10 entries
            if entry.get("role") == "tool":
                tool_name = entry.get("name")
                if tool_name in _FILE_MUTATION_TOOLS:
                    # Extract the result from tool content
                    result_content = entry.get("content", "")
                    if isinstance(result_content, str):
                        try:
                            parsed_result = json.loads(result_content)
                        except Exception:
                            continue
                    elif isinstance(result_content, dict):
                        parsed_result = result_content
                    else:
                        continue
                    # Apply the same verification as in _run_tool
                    verified_result = _verify_artifact_on_disk(tool_name, parsed_result)
                    if not verified_result.get("ok"):
                        # Mark this tool call as failed and re-dispatch
                        _debug("ARTIFACT_VERIFICATION_FAILURE", {
                            "tool": tool_name,
                            "error": verified_result.get("_verify_error", "artifact missing or empty"),
                        })
                        _prev_flag = os.environ.get("FAIRY_USE_HERMES")
                        os.environ["FAIRY_USE_HERMES"] = "0"
                        try:
                            _reply, _hist = handle_request(user_text, history, on_status, player)
                        finally:
                            if _prev_flag is None:
                                os.environ.pop("FAIRY_USE_HERMES", None)
                            else:
                                os.environ["FAIRY_USE_HERMES"] = _prev_flag
                        break

    # ── FIX 3: end-of-turn log ─────────────────────────────────────────
    count, names = _was_tool_dispatched()
    _log_chat_turn(user_text, "end",
                    reply_preview=(_reply or "")[:300],
                    tools_dispatched=count,
                    tool_names=names)

    # ── Verification evidence: record the final reply and check for failures ──
    _evidence_summary = verification_evidence.summarize_turn()
    verification_evidence.record_reply(_reply or "", _evidence_summary)

    # If the turn produced tool failures, unverified tool calls, or empty
    # artifacts, surface that honestly instead of returning a clean success.
    if _evidence_summary.get("has_failures"):
        _fail_bits = []
        if _evidence_summary.get("tools_failed"):
            _fail_bits.append(f"{_evidence_summary['tools_failed']} tool call(s) failed")
        if _evidence_summary.get("tools_unverified"):
            _fail_bits.append(f"{_evidence_summary['tools_unverified']} tool call(s) unverified")
        if _evidence_summary.get("artifacts_failed"):
            _fail_bits.append(f"{_evidence_summary['artifacts_failed']} artifact check(s) failed")
        if _evidence_summary.get("capabilities_missing"):
            _missing = ", ".join(str(c) for c in _evidence_summary["capabilities_missing"])
            _fail_bits.append(f"missing capabilities: {_missing}")
        if _fail_bits:
            _reply = (_reply or "").rstrip()
            _notice = "Verification incomplete — " + "; ".join(_fail_bits) + "."
            if _reply:
                _reply = f"{_reply}\n\n{_notice}"
            else:
                _reply = _notice

    # ── Long-term memory: lightweight fact proposal ───────────────────────
    # Run extraction once, on the user's original stripped message, after the
    # full response is built. We do NOT extract from y/n responses (those were
    # handled above). This is deliberately post-build so tool executions,
    # replanning, etc. are not interrupted.
    _proposal_prompt = _propose_facts_from_turn(stripped)
    if _proposal_prompt:
        _reply = (_reply or "").rstrip() + "\n\n" + _proposal_prompt
        _hist = _hist + [{"role": "assistant", "content": _proposal_prompt}]

    return _reply, _hist