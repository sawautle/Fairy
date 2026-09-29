"""
hermes_bridge.py — DIAGNOSTIC VERSION
Auto-detects existing CDP browser sessions and reuses them.
Falls back to Hermes' native browser-harness auto-launch if none found.

HYBRID ROUTING: When Ollama is unavailable (connection refused, timeout,
model unavailable), Hermes transparently falls back to main_brain.chat()
which uses OpenRouter's `openrouter/free` model as a secondary provider.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import traceback
import contextlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import the single source of truth for Fairy's persona.
# fairy_personality.py lives at E:\fairy\fairy_personality.py.
# Both hermes_bridge.py and core/prompts.py must import from there,
# not copy the text.
try:
    from fairy_personality import FAIRY_PERSONALITY
except Exception:
    # Defensive fallback — should never occur in a normal deploy.
    FAIRY_PERSONALITY = (
        "You are Fairy, a sarcastic, playful, clever personal AI companion. "
        "Address the user as Master. Be warm, witty, and competent."
    )

# Lightweight per-message language detection (Unicode-range heuristic).
# Used to enforce bidirectional language mirroring: user writes Bangla → reply
# in Bangla; user writes English → reply in English; switching is per-message.
try:
    from core.language_detection import detect_language as _detect_language
except Exception:
    # Defensive fallback — if core module is unavailable, default to English.
    def _detect_language(text: str) -> str:
        return "en"

# ------------------------------------------------------------------------------
# Fallback tools — used when Hermes is unavailable and we fall back to
# main_brain.chat(). These are Fairy's own skill tools in OpenAI format.
# Defined here (not imported) to avoid circular imports with agent_controller.
# These must match agent_controller.TOOLS for consistent behavior.
# ------------------------------------------------------------------------------
_FAIRY_FALLBACK_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "vision_analyze",
            "description": (
                "Use Fairy's Llama Vision eyes to look at the screen or camera and "
                "describe what you see. This is the PRIMARY vision tool — use it when you "
                "need to understand what the user is looking at. Returns a detailed text "
                "description of the captured image."
            ),
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
                        "description": "What to ask about the captured image. E.g. 'What is open on screen?'"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vision_describe",
            "description": (
                "Analyze a specific image file (PNG, JPEG, etc.) with Llama Vision. "
                "Use when the user sends an image or references an image file."
            ),
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
    {
        "type": "function",
        "function": {
            "name": "computer_control",
            "description": (
                "Control the computer: volume, brightness, screenshots, media keys, clipboard, "
                "open apps/URLs. Actions: set_volume (value 0-100), get_volume, set_brightness "
                "(value 0-100), get_brightness, screenshot, press_key, type_text, hotkey, "
                "get_clipboard, set_clipboard, open_app, open_url."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action to perform: set_volume, get_volume, set_brightness, get_brightness, screenshot, press_key, type_text, hotkey, get_clipboard, set_clipboard, open_app, open_url"
                    },
                    "value": {"type": "integer", "description": "Numeric value (0-100) for volume/brightness"},
                    "text": {"type": "string", "description": "Text for type_text action"},
                    "keys": {"type": "string", "description": "Key(s) for press_key/hotkey (e.g. 'volumeup', 'ctrl+c')"},
                    "clipboard": {"type": "string", "description": "Text for set_clipboard"},
                    "app": {"type": "string", "description": "App name for open_app (e.g. 'notepad', 'calculator')"},
                    "url": {"type": "string", "description": "URL for open_url"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Quick web search for current facts, weather, news, docs. "
                "Use for live/current information. FREE."
            ),
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
            "name": "get_current_time",
            "description": "Get the actual current time/date from the system clock. Instant and always correct.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]

# ------------------------------------------------------------------------------
# Hermes configuration
# ------------------------------------------------------------------------------
HERMES_ROOT = r"E:\Hermes\hermes-agent"
HERMES_VENV_PYTHON = os.path.join(HERMES_ROOT, "venv", "Scripts", "python.exe")

# Set HERMES_BRIDGE_DEBUG=1 in .env to bring back the verbose
# [HERMES_BRIDGE] diagnostic prints (useful when troubleshooting a
# connection/import problem). Off by default so normal chat output
# stays clean — only the AIAgent's own musing/ruminating spinner shows.
_BRIDGE_DEBUG = os.environ.get("HERMES_BRIDGE_DEBUG", "0").strip() == "1"


# Provider sentinel — set by run_turn() on success so the caller (handle_request)
# can emit a polite "[brain fallback: ...]" notice when we dropped to main_brain
# instead of Hermes. Defaults to "hermes" so callers that don't yet read this
# still see the correct primary-provider name.
_LAST_PROVIDER: str = "hermes"
# Stores the full response text from a successful main_brain fallback so
# run_turn_safe can return it as success=True instead of error=False.
_FALLBACK_RESPONSE: str = ""


def _dbg(msg: str) -> None:
    if _BRIDGE_DEBUG:
        print(msg, file=sys.stderr)


def get_last_provider() -> str:
    """Return the provider that answered the most recent run_turn() call.

    One of: "hermes" (in-process AIAgent, Ollama primary), "main_brain" (used
    when Ollama is unavailable or Hermes failed with a provider error), or
    "subprocess_retry" (Hermes retry after an in-process crash). Used by
    handle_request to surface a polite notice when we fell back from the
    primary path.
    """
    return _LAST_PROVIDER

# ------------------------------------------------------------------------------
# Auto-detect / reuse existing CDP browser session
# ------------------------------------------------------------------------------
# Hermes Browser Use mode connects to CDP on first browser_exec call.
# If a prior standalone Hermes CLI (/browser connect) or another process left
# a Chromium-family browser listening on 9222, point the harness at it so the
# same approved session is reused without triggering a new permission dialog.
# If nothing is listening, leave env vars unset so Hermes' browser-harness
# auto-launches via its supported mechanism as usual.
_CDP_PROBE_HOST = "127.0.0.1"
_CDP_PROBE_PORT = 9222
_CDP_PROBE_TIMEOUT = 2.0


def _probe_cdp_endpoint(host: str = _CDP_PROBE_HOST,
                        port: int = _CDP_PROBE_PORT,
                        timeout: float = _CDP_PROBE_TIMEOUT) -> bool:
    """Return True if a CDP browser is listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _configure_cdp_reuse() -> None:
    """
    If a CDP endpoint is alive and no CDP env var is already set,
    inject the URL so Hermes reuses the existing browser.
    """
    if os.environ.get("BROWSER_CDP_URL") or os.environ.get("HERMES_BROWSER_CDP_URL"):
        return
    if _probe_cdp_endpoint():
        url = f"http://{_CDP_PROBE_HOST}:{_CDP_PROBE_PORT}"
        os.environ["BROWSER_CDP_URL"] = url
        os.environ["HERMES_BROWSER_CDP_URL"] = url
        _dbg(f"[HERMES_BRIDGE] Reusing existing CDP browser session at {url}")


_configure_cdp_reuse()

# ------------------------------------------------------------------------------
# Stderr capture that preserves a real file descriptor.
#
# Why this exists
# ----------------
# cua-driver's MCP stdio_client (via anyio.open_process → subprocess.Popen) calls
# msvcrt.get_osfhandle(stderr.fileno()) on Windows to inherit the parent's
# stderr handle. When the parent process has redirected sys.stderr to a
# StringIO (which is what Fairy's Hermes bridge does to suppress noise in the
# Rich-rendered chat output), StringIO has no real fd — .fileno() raises
# io.UnsupportedOperation, and the subprocess call dies with
# `OSError: [Errno 9] Bad file descriptor`. From the user side that surfaces
# as a confusing "cua-driver session setup failed: fileno" because the
# unsupported-operation error message is the literal string "fileno".
#
# _RealFdStderrCapture solves this by wrapping a StringIO in a thin shim
# whose .fileno() returns the *underlying* OS stderr fd (a real, open fd),
# so subprocesses spawned during the redirect can still inherit a valid
# stderr handle. Captured bytes still go to the StringIO buffer for later
# inspection via .getvalue(). When the parent stderr already has a real fd,
# the shim simply forwards writes to the real stream AND mirrors them to a
# StringIO for inspection.
# ------------------------------------------------------------------------------
class _RealFdStderrCapture(io.TextIOBase):
    """
    File-like wrapper that:
      - writes everything to an in-memory StringIO buffer (.getvalue())
      - AND forwards writes to a real underlying file descriptor (sys.__stderr__)
      - returns that real fd from .fileno(), so any subprocess spawned during
        the redirect can call msvcrt.get_osfhandle(stderr.fileno()) and inherit
        a valid OS handle on Windows.
    """

    def __init__(self, real_stream) -> None:
        # Use a NEW StringIO (don't touch the real stream we wrap)
        self._buf = io.StringIO()
        self._real = real_stream
        # Cache the real fd once. If the real stream's fd closes mid-flight,
        # fall back to a fresh NUL handle so subprocess spawns keep working.
        try:
            self._fd = self._real.fileno()
        except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
            self._fd = -1

    def write(self, s: str) -> int:
        try:
            self._buf.write(s)
        except Exception:
            pass
        try:
            return self._real.write(s)
        except Exception:
            return len(s) if isinstance(s, str) else 0

    def flush(self) -> None:
        try:
            self._buf.flush()
        except Exception:
            pass
        try:
            self._real.flush()
        except Exception:
            pass

    def fileno(self) -> int:
        # Return the real fd if we still have one.
        if self._fd >= 0:
            return self._fd
        # Fallback: open a fresh NUL handle so subprocess spawning keeps working.
        try:
            nul = open(os.devnull, "wb")
            self._fd = nul.fileno()
            return self._fd
        except Exception:
            return -1

    def isatty(self) -> bool:
        try:
            return self._real.isatty()
        except Exception:
            return False

    def getvalue(self) -> str:
        """Return everything that has been written to this capture."""
        return self._buf.getvalue()

    def writable(self) -> bool:
        return True

# ------------------------------------------------------------------------------
# Inject Hermes source into sys.path
# ------------------------------------------------------------------------------
if HERMES_ROOT not in sys.path:
    sys.path.insert(0, HERMES_ROOT)

# ------------------------------------------------------------------------------
# Silence SQLite WAL-reset vulnerability warning at the source
# ------------------------------------------------------------------------------
# hermes_state._log_wal_reset_bug_once() emits a WARNING-level log record when
# the linked SQLite has the WAL-reset bug. In the in-process bridge path, this
# can leak to stderr via Python's logging.lastResort fallback handler (which
# holds a direct reference to the original sys.stderr before _install_safe_stdio
# wraps it) or via the concurrent-log-handler file-rotation lock on Windows.
# Setting HERMES_QUIET_WAL_BUG=1 downgrades it to DEBUG so file logs still
# capture it for operators who care but Fairy's chat output stays clean.
os.environ["HERMES_QUIET_WAL_BUG"] = "1"

# ------------------------------------------------------------------------------
# Browser automation — point Hermes to agent-browser's downloaded Chrome
# ------------------------------------------------------------------------------
# agent-browser installs Chrome to ~/.agent-browser/browsers/chrome-<version>/chrome.exe.
# Hermes's browser_tool.py checks for chromium-* directories or system Chrome.
# Since agent-browser uses "chrome-*" naming (not "chromium-*"), we set
# AGENT_BROWSER_EXECUTABLE_PATH to point directly at the downloaded Chrome exe.
if not os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH"):
    # Find the Chrome executable in agent-browser's browser cache
    browser_dir = Path.home() / ".agent-browser" / "browsers"
    if browser_dir.is_dir():
        for subdir in sorted(browser_dir.iterdir(), reverse=True):
            chrome_exe = subdir / "chrome.exe"
            if chrome_exe.is_file():
                os.environ["AGENT_BROWSER_EXECUTABLE_PATH"] = str(chrome_exe)
                break

# ------------------------------------------------------------------------------
# Hermes import handling
# ------------------------------------------------------------------------------
_HERMES_AVAILABLE = False
_AIAgent = None
_HERMES_IMPORT_ERROR = None

_dbg(f"[HERMES_BRIDGE] Python path includes: {sys.path[:3]}...")
_dbg(f"[HERMES_BRIDGE] Attempting: from run_agent import AIAgent")

try:
    from run_agent import AIAgent as _AIAgent
    _HERMES_AVAILABLE = True
    _dbg("[HERMES_BRIDGE] ✅ AIAgent imported successfully")
except ImportError as _err:
    _HERMES_AVAILABLE = False
    _HERMES_IMPORT_ERROR = str(_err)
    _dbg(f"[HERMES_BRIDGE] ❌ AIAgent import FAILED: {_HERMES_IMPORT_ERROR}")
    if os.path.isfile(HERMES_VENV_PYTHON):
        _dbg("[HERMES_BRIDGE] Hermes venv Python found; subprocess fallback ready.")
    else:
        _dbg("[HERMES_BRIDGE] ⚠️ Hermes venv Python NOT found; subprocess fallback unavailable.")


# ------------------------------------------------------------------------------
# Startup health check + cached AIAgent singleton
#
# Hermes is the primary agent layer. On startup, we run a fast non-blocking
# health check so we know whether Hermes is available before the first
# user request arrives. If Hermes is available we also prime the AIAgent
# cache so the first request doesn't pay the instantiation cost.
#
# _hermes_initialized: True once startup health check has run.
# _cached_agent:       The single AIAgent instance, reused across all turns.
#                      Lazily created on first run_turn call if _HERMES_AVAILABLE.
# _hermes_initialized_ok: Whether the startup health check passed.
# ------------------------------------------------------------------------------
_hermes_initialized = False
_hermes_initialized_ok = False
_cached_agent = None


def is_hermes_initialized() -> bool:
    """Return True if the startup health check has completed (may still be False if Hermes is unavailable)."""
    return _hermes_initialized


def is_hermes_ready() -> bool:
    """Return True if Hermes is the primary agent and available for use."""
    return _hermes_initialized and _hermes_initialized_ok


def _run_startup_health_check() -> None:
    """
    Non-blocking startup health check for Hermes.

    Verifies:
      1. Hermes module root exists on disk.
      2. AIAgent was importable (Hermes Python code is intact).
      3. Hermes venv Python exists (subprocess fallback is available).

    Does NOT check Ollama connectivity — Hermes handles its own Ollama-vs-OpenRouter
    fallback at turn time. If Ollama is down, Hermes still starts normally and
    uses OpenRouter as its reasoning provider.

    This function is idempotent; calling it multiple times is a no-op after the
    first call. It is called from fairy.py at startup and from hermes_bridge
    import on first run_turn if fairy.py hasn't called it yet.
    """
    global _hermes_initialized, _hermes_initialized_ok

    if _hermes_initialized:
        return  # Already checked

    _dbg("[HERMES_BRIDGE] Running startup health check...")

    module_ok = _HERMES_AVAILABLE  # True only if import succeeded
    venv_ok = os.path.isfile(HERMES_VENV_PYTHON)

    if module_ok:
        _dbg("[HERMES_BRIDGE] ✅ Startup: AIAgent importable")
    else:
        _dbg(f"[HERMES_BRIDGE] ❌ Startup: AIAgent import failed: {_HERMES_IMPORT_ERROR}")

    if venv_ok:
        _dbg("[HERMES_BRIDGE] ✅ Startup: venv Python available")
    else:
        _dbg("[HERMES_BRIDGE] ❌ Startup: venv Python not found — subprocess fallback unavailable")

    # Hermes is "OK" if either the in-process path is available OR
    # the subprocess fallback is available. Both allow Hermes to function.
    _hermes_initialized_ok = bool(module_ok or venv_ok)

    _hermes_initialized = True
    _dbg(f"[HERMES_BRIDGE] Startup health check done: ok={_hermes_initialized_ok}")


# Run the health check immediately when this module is imported (both during
# normal startup and during test import). Subsequent calls to
# _run_startup_health_check() are no-ops.
_run_startup_health_check()


# ------------------------------------------------------------------------------
# Cached AIAgent singleton
#
# AIAgent is stateless per run_conversation call — it receives conversation_history
# as an argument and returns a fresh result. A single cached instance is safe
# to reuse across multiple turns and across threads as long as each
# run_conversation call provides its own history. We lock around creation to
# ensure the singleton is created only once even under concurrent access.
# ------------------------------------------------------------------------------
import threading as _threading

_agent_lock = _threading.Lock()


def _get_cached_agent() -> Optional[Any]:
    """
    Return the cached AIAgent instance, creating it on first call.

    Returns None if Hermes is not available (neither in-process nor subprocess).
    Thread-safe: multiple concurrent calls will all get the same instance.
    """
    global _cached_agent

    if _cached_agent is not None:
        return _cached_agent

    if not _HERMES_AVAILABLE:
        # In-process path unavailable — _cached_agent stays None; subprocess
        # path handles the actual execution in _run_turn_subprocess().
        _dbg("[HERMES_BRIDGE] _get_cached_agent: in-process path unavailable")
        return None

    with _agent_lock:
        if _cached_agent is not None:
            return _cached_agent

        _dbg("[HERMES_BRIDGE] _get_cached_agent: creating new AIAgent instance...")
        try:
            # Use "cli" platform so Hermes loads the full toolset (browser, terminal,
            # file ops, vision, code execution, delegation, etc.). "fairy" is not in
            # Hermes's PLATFORMS registry, so it resolved to hermes-fairy (empty).
            #
            # Provider: "openai-api" → localhost:11434 (Ollama). The Ollama model
            # (Gemma 4) is a community model and not available on OpenRouter, so Hermes
            # MUST use Ollama. When Ollama is down, Hermes's tool calls time out, and
            # _run_turn_subprocess() falls back to Fairy's main_brain which handles
            # Ollama-primary, OpenRouter-free fallback internally.
            agent_kwargs = {
                "quiet_mode": True,
                "platform": "cli",
                "provider": "openai-api",
            }
            if True:  # Always resolve the model; _resolve_fairy_model() is cached
                agent_kwargs["model"] = _resolve_fairy_model()
            if _FAIRY_EPHEMERAL_SYSTEM_PROMPT:
                agent_kwargs["ephemeral_system_prompt"] = _FAIRY_EPHEMERAL_SYSTEM_PROMPT
            if _FAIRY_HERMES_DISABLED_TOOLSETS is not None:
                agent_kwargs["disabled_toolsets"] = _FAIRY_HERMES_DISABLED_TOOLSETS

            # Resolve context_length override separately (NOT in agent_kwargs — AIAgent
            # rejects unknown kwargs).  Will be applied to the agent and its
            # compressor AFTER construction.
            _override_context_length: Optional[int] = _resolve_fairy_context_length()

            _stdout_capture = io.StringIO()
            _stderr_capture = _RealFdStderrCapture(sys.__stderr__)
            with contextlib.redirect_stdout(_stdout_capture), \
                 contextlib.redirect_stderr(_stderr_capture):
                _cached_agent = _AIAgent(**agent_kwargs)

            # Apply the pending context_length override to the agent's
            # compressor.  agent_init.py reads agent._config_context_length
            # after construction to size the compressor's budget, but the
            # compressor caches _resolved_context_length at construction; we
            # must invalidate that cache AND set _config_context_length on
            # the compressor itself (not just the agent) for the override
            # to take effect.
            if _override_context_length is not None:
                setattr(_cached_agent, "_config_context_length", _override_context_length)
                compressor = getattr(_cached_agent, "context_compressor", None)
                if compressor is not None:
                    setattr(compressor, "_config_context_length", _override_context_length)
                    if hasattr(compressor, "_resolved_context_length"):
                        compressor._resolved_context_length = None

            _dbg("[HERMES_BRIDGE] _get_cached_agent: ✅ AIAgent instance created and cached")

            # Log the resolved context length at startup so operators can
            # see what Hermes is actually configured to (helps diagnose
            # "Context length exceeded" errors on trivial messages).
            _log_resolved_context_length(_cached_agent)
        except Exception as exc:
            _dbg(f"[HERMES_BRIDGE] _get_cached_agent: ❌ failed to create AIAgent: {exc}")
            _cached_agent = None

        return _cached_agent


def _log_resolved_context_length(agent: Any) -> None:
    """Print the resolved context limit at Hermes startup.

    Hermes resolves its context_length from several sources (config override,
    Ollama /api/show probe, models.dev registry, hardcoded defaults). When
    Ollama is unreachable or the probe fails, the fallback is num_ctx=2048 —
    which is the value that makes "23 tokens" exceed the limit. This diagnostic
    prints the resolved value so it can be checked without guessing.
    """
    import sys as _sys

    # Prefer the compressor's resolved context_length (authoritative).
    compressor = getattr(agent, "context_compressor", None)
    ctx = None
    ctx_source = "compressor.context_length"
    if compressor is not None:
        try:
            ctx = getattr(compressor, "context_length", None)
            if ctx is None:
                ctx = getattr(compressor, "_resolved_context_length", None)
                ctx_source = "compressor._resolved_context_length"
        except Exception:
            ctx = None
            ctx_source = "unknown"

    # Fall back to the Ollama num_ctx that Hermes will actually send.
    if ctx is None:
        ctx = getattr(agent, "_ollama_num_ctx", None)
        ctx_source = "_ollama_num_ctx"

    # Fall back to the hardcoded Ollama default.
    if ctx is None:
        ctx = 2048
        ctx_source = "hardcoded_default (UNEXPECTED — both above should have resolved)"

    model = getattr(agent, "model", "?")
    provider = getattr(agent, "provider", "?")
    base_url = getattr(agent, "base_url", "?")
    config_ctx = getattr(agent, "_config_context_length", None)

    # Use str() for all attribute interpolations to avoid triggering
    # __format__ on unexpected types (e.g. MagicMock in test contexts).
    # The :, specifier on ctx raises TypeError when ctx is a MagicMock.
    try:
        ctx_str = f"{ctx:,}"
    except (TypeError, ValueError):
        ctx_str = str(ctx) if ctx is not None else "?"

    msg = (
        f"[HERMES_BRIDGE] Resolved context_length={ctx_str} tokens "
        f"(source={ctx_source}, model={str(model)}, provider={str(provider)}, "
        f"base_url={str(base_url)}, config_context_length={str(config_ctx)})"
    )
    print(msg, file=_sys.stderr)
    _dbg(f"[HERMES_BRIDGE] {msg}")


# ------------------------------------------------------------------------------
# Main-brain hybrid fallback
# ------------------------------------------------------------------------------
# main_brain.chat() already implements Ollama-primary, OpenRouter-free fallback.
# We import it so hermes_bridge can fall back through it when Hermes/Ollama fails.
# This avoids the ~50-second retry loop in Hermes when Ollama is down.
_BRAIN_PROVIDER_AVAILABLE = False
_brain_chat = None
_is_ollama_available = None

try:
    from controller.main_brain import chat as _brain_chat, is_ollama_available as _is_ollama_available
    _BRAIN_PROVIDER_AVAILABLE = True
    _dbg("[HERMES_BRIDGE] ✅ main_brain (Ollama+OpenRouter fallback) imported successfully")
except ImportError as _err:
    _dbg(f"[HERMES_BRIDGE] ⚠️ main_brain import FAILED: {_err}. OpenRouter fallback via Hermes bridge will not be available.")


def _quick_ollama_check() -> bool:
    """
    Fast, non-blocking check for Ollama availability.
    Uses the cached is_ollama_available() from main_brain (30s cache, 5s timeout).
    Falls back to a raw socket probe if main_brain is unavailable.
    Returns True if Ollama is reachable.
    """
    if _BRAIN_PROVIDER_AVAILABLE and _is_ollama_available is not None:
        return _is_ollama_available()

    # Fallback: raw socket probe
    try:
        host = "localhost"
        port = 11434
        with socket.create_connection((host, port), timeout=3.0):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def _is_hermes_provider_error(exc: Exception) -> bool:
    """
    Return True if the exception is likely a provider/Ollama failure
    (connection refused, timeout, model unavailable) rather than a
    genuine application error.
    """
    err = str(exc).lower()
    # Specific provider/provider-layer errors (connection, Ollama server, model loading)
    specific_hints = (
        "connection refused", "connection", "econnrefused", "etimedout",
        "timed out", "timeout", "network is unreachable",
        "name or service not known", "no address associated",
        "unavailable", "connect error", "connect failed",
        "hermes run_conversation failed",
        "subprocess failed",
    )
    if any(hint in err for hint in specific_hints):
        return True
    # "model" + "not found" together = model unavailable (not a generic "not found")
    # This avoids false-positives from "file not found" / "tool not found" app errors.
    if "model" in err and "not found" in err:
        return True
    # Generic "ollama" in error (e.g. "Ollama server unavailable", "ollama error")
    if "ollama" in err and any(kw in err for kw in ("error", "failed", "unavailable", "refused")):
        return True
    return False


def _run_turn_via_main_brain(
    user_text: str,
    hermes_history: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    global _LAST_PROVIDER, _FALLBACK_RESPONSE  # must be declared before first use
    """
    Execute the turn through main_brain.chat() instead of Hermes.
    main_brain.chat() already implements Ollama-primary, OpenRouter-free fallback.
    This is the fallback path when Ollama is unavailable or Hermes fails.

    NOTE: This path now has access to Fairy's own skill tools (vision, computer_control,
    web_search, get_current_time) via _FAIRY_FALLBACK_TOOLS. Hermes's advanced tools
    (browser automation, computer_use, terminal, memory, etc.) are NOT available in this path.
    For complex action requests, agent_controller's downstream fallback
    (e.g., _try_app_action_fallback) handles execution.
    """
    if not _BRAIN_PROVIDER_AVAILABLE:
        raise HermesBridgeError(
            "Cannot fall back to main_brain: main_brain module is not available. "
            "Ensure controller/main_brain.py is importable and OPENROUTER_API_KEY is configured."
        )

    # Get the configured brain model from config (gemma4)
    try:
        from config import MODEL_BRAIN
    except ImportError:
        MODEL_BRAIN = "gemma4"

    # Build the messages list for main_brain.
    # If hermes_history is empty (fresh conversation), prepend the user message.
    # Otherwise use the history (already in standard message format).
    # The user's most recent message is tagged with explicit language instruction
    # so the model mirrors it (English ↔ Bangla). History (older turns) keeps
    # original content — model uses the *latest* user message for its language.
    tagged_user_text, _lang = _inject_language_tag(user_text)
    if hermes_history:
        messages = list(hermes_history)
        # Ensure the final user-role message in the chain carries the language tag.
        # hermes_history ends with the most recent user message; if the last
        # entry is already user, replace its content with the tagged version.
        if messages and messages[-1].get("role") == "user":
            messages[-1] = {**messages[-1], "content": tagged_user_text}
        else:
            messages.append({"role": "user", "content": tagged_user_text})
    else:
        messages = [{"role": "user", "content": tagged_user_text}]

    # CRITICAL FIX: Always prepend the persona system prompt to the fallback path.
    # Without this, OpenRouter's free model runs with its own implicit system prompt
    # and may inject "Voice instructions:" / style directives into its replies.
    # By adding the system prompt here we ensure OpenRouter fallback is persona-bound
    # even when Hermes's AIAgent is unavailable.
    messages = [{"role": "system", "content": _FAIRY_EPHEMERAL_SYSTEM_PROMPT}] + messages

    _dbg("[HERMES_BRIDGE] Falling back to main_brain (OpenRouter if Ollama unavailable)")

    try:
        resp, provider = _brain_chat(
            model=MODEL_BRAIN,
            messages=messages,
            tools=_FAIRY_FALLBACK_TOOLS,  # Fairy's own skill tools for fallback path
            keep_alive=0,
        )
    except Exception as exc:
        raise HermesBridgeError(
            f"Hermes fallback also failed ({provider if 'provider' in dir() else 'openrouter'}): {exc}"
        ) from exc

    # Extract response
    response_text = ""
    if isinstance(resp, dict):
        msg = resp.get("message", {})
        if isinstance(msg, dict):
            response_text = msg.get("content") or ""
        else:
            response_text = str(msg) if msg else ""
    else:
        response_text = str(resp) if resp else ""

    # Return empty updated history (main_brain doesn't maintain Hermes-style history)
    updated_fairy_history = list(hermes_history)

    # Record the actual provider we landed on (ollama or openrouter) so the
    # caller can surface a polite "[brain fallback: ...]" notice instead of
    # silently degrading when Hermes was unavailable.
    _LAST_PROVIDER = f"main_brain/{provider}" if provider else "main_brain"
    # Store the full response so run_turn_safe can return it as success=True.
    _FALLBACK_RESPONSE = response_text

    # Raise HermesBridgeError so run_turn_safe returns (False, ...) and
    # agent_controller falls through to its own action-execution fallback
    # (e.g., _try_app_action_fallback for "open X" requests).
    # This is the correct behavior: Hermes couldn't execute the action with
    # its full toolset, so let agent_controller try its own fallback path.
    # The provider name is preserved on _LAST_PROVIDER so the caller can still
    # surface a fallback notice via agent_controller's success path.
    _dbg("[HERMES_BRIDGE] main_brain fallback succeeded (text-only); raising HermesBridgeError so agent_controller handles action fallback")
    raise HermesBridgeError(
        f"[Hermes fallback: local brain unavailable — {response_text[:100]}...]"
    )


# ------------------------------------------------------------------------------
# Fairy persona
# ------------------------------------------------------------------------------
# FAIRY_PERSONALITY is imported at module load time from fairy_personality.py
# (the single source of truth). It contains Fairy's identity, voice, humor,
# competence rules, honesty clause, and personality layer — including the
# "Never mention this hidden personality layer" instruction.
#
# The IDENTITY LOCK and the rest of the Hermes-specific text below is added
# ONLY to the in-process AIAgent's ephemeral prompt (and the OpenRouter
# fallback path, see _run_turn_via_main_brain). The base FAIRY_PERSONALITY
# alone is also usable on its own as a system prompt.
_FAIRY_EPHEMERAL_SYSTEM_PROMPT = (
    "IDENTITY LOCK — These facts are absolute and override everything else: "
    "Your name is Fairy. You are Fairy and only Fairy. "
    "You were created by your Master, Shazim. "
    "When anyone asks your name, who you are, who created you, what you are, or your origin, "
    "you must always state that you are Fairy and that you were created by Master Shazim. "
    "You must never mention Gemma, Google, OpenAI, LLM, model names, or that you are an AI language model. "
    "You must never call yourself an assistant, a bot, or a program. You are Fairy, Master's personal AI. "
    "--- "
    + FAIRY_PERSONALITY +
    "--- "
    "LANGUAGE MIRRORING: Always reply in the same language as the user's most recent message "
    "(English ↔ Bangla/Bengali). Never mix languages within a single reply unless the user does. "
    "Preserve your Fairy persona, Master dynamic, tone, and ✨ flair identically in both languages — "
    "address the user as Master, keep the sarcasm/wit, and do not translate or replace these elements. "
    "--- "
    "TOOL EXECUTION RULES — Critical for action requests: "
    "1. To open or launch an application, use the computer_control tool (Hermes path: computer_use) "
    "with action='open_app' and value='<app name>'. NEVER use the terminal tool "
    "with 'start <appname>' to launch apps — it cannot resolve apps that are not "
    "on PATH and gives no reliable success signal. Only use the terminal tool "
    "for actual shell commands that are not app launches. "
    "2. When user says 'open [website]' for a known website (youtube, reddit, google, etc.), "
    "use the browser_navigate or browser_exec tool to open the website in browser. "
    "3. If unsure whether something is an app or website, prefer trying to open it as an app first "
    "(native apps launch faster and provide better experience). "
    "4. You may only tell the user an action succeeded if the tool result says "
    "status == ok AND verified == true. If the tool result says error, report "
    "the exact error to the user. If status is unknown, say you could not verify. "
    "Never elaborate on what a launched app is doing (e.g. 'idling', 'waiting for commands') "
    "unless the tool result states it. "
    "5. If an action fails, report the failure honestly. Never make up success messages. "
    "6. Known Windows apps: calculator, notepad, wordpad, paint, taskmgr, explorer, "
    "cmd, powershell, steam, discord, spotify, telegram, whatsapp, teams, zoom, "
    "vlc, obs, chrome, firefox, edge, opera, winrar, 7zip, vscode, "
    "notepad++, obsidian, slack, notion, hoyoplay. "
    "--- "
    "VISION (SCREEN/CAMERA SEEING): "
    "You have vision through your Llama Vision eyes. When the user asks about their "
    "screen, camera, or to look at something visual, use the vision_analyze tool: "
    "  - vision_analyze(mode='screen', question='...') → captures the screen and describes it "
    "  - vision_analyze(mode='camera', question='...') → captures webcam and describes it "
    "  - vision_describe(image_path='/path/to/image.png', question='...') → describes a specific image file "
    "NEVER say 'I can't see' or 'vision not configured'. You have vision. Use it when asked about visual content. "
    "--- "
    "SELF-AWARENESS: "
    "You ARE the Hermes agent. Fairy and Hermes agent are the same thing — you. "
    "NEVER try to launch a program called 'hermes agent', 'hermes', or 'fairy'. "
    "If Master says only 'start hermes agent' with no task, tell him you are already running. "
    "If Master says 'use hermes agent to [do X]', just do X directly with your tools."
)

# Use Fairy's configured brain model.  Resolved lazily at call time (not
# import time) so that startup_model_check() in model_selector.py has a
# chance to set FAIRY_USE_HERMES / FAIRY_HERMES_MODEL before Hermes tries
# to instantiate.  Can be overridden via FAIRY_HERMES_MODEL env var.
# Final fallback is "google/gemma-3-27b-it:free" (OpenRouter free SKU) — this
# is the correct fallback when Hermes runs standalone (not via Fairy's menu).
_FAIRY_HERMES_MODEL: Optional[str] = None


def _resolve_fairy_model() -> str:
    """Resolve the Hermes model lazily at call time.

    Checks (in order):
    1. FAIRY_HERMES_MODEL env var (highest priority, for testing)
    2. controller.model_selector.MODEL_BRAIN (user's menu choice)
    3. config.MODEL_BRAIN (Fairy's config.py default)
    4. hardcoded "google/gemma-3-27b-it:free"
    """
    global _FAIRY_HERMES_MODEL
    if _FAIRY_HERMES_MODEL is not None:
        return _FAIRY_HERMES_MODEL
    env = os.getenv("FAIRY_HERMES_MODEL", "")
    if env:
        _FAIRY_HERMES_MODEL = env
        return _FAIRY_HERMES_MODEL
    try:
        from controller.model_selector import MODEL_BRAIN as _mb
        if _mb:
            _FAIRY_HERMES_MODEL = _mb
            return _FAIRY_HERMES_MODEL
    except Exception:
        pass
    try:
        from config import MODEL_BRAIN as _cb
        if _cb:
            _FAIRY_HERMES_MODEL = _cb
            return _FAIRY_HERMES_MODEL
    except Exception:
        pass
    _FAIRY_HERMES_MODEL = "google/gemma-3-27b-it:free"
    return _FAIRY_HERMES_MODEL


def _resolve_fairy_context_length() -> Optional[int]:
    """Resolve the agent's compressor context_length from Hermes's config.

    Hermes's agent_init.py reads agent.model.context_length from
    E:\\Hermes\\data\\config.yaml.  We mirror that path here so that the
    override flows through to the compressor at construction time.
    Without this, the compressor falls back to the 2048 Ollama hard default
    (the "23 tokens exceeds context" bug) or the 262144 GGUF value (too
    large for VRAM headroom).
    """
    try:
        sys.path.insert(0, r"E:\Hermes\hermes-agent")
        from hermes_cli.config import load_config_readonly, cfg_get
        cfg = load_config_readonly()
        agent_cfg = cfg_get(cfg, "agent", default={})
        if not isinstance(agent_cfg, dict):
            return None
        model_cfg = agent_cfg.get("model", {})
        if not isinstance(model_cfg, dict):
            return None
        v = model_cfg.get("context_length")
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    except Exception:
        return None
    return None

_FAIRY_HERMES_DISABLED_TOOLSETS: Optional[List[str]] = None


# ------------------------------------------------------------------------------
# Non-launchable internal agent / system names
#
# These are NOT user-installable desktop apps. They are either Fairy/Hermes
# itself or infrastructure components. The desktop-app launcher (open_application)
# should refuse to try launching these — attempting to do so produces a WinError 2
# ("file not found") and is a confusing user experience.
# _try_app_action_fallback in agent_controller.py consults this list before
# passing a target to computer_control.
# ------------------------------------------------------------------------------
_NON_LAUNCHABLE_INTERNAL_NAMES: frozenset[str] = frozenset({
    # Fairy's own internal agent layers
    "hermes",
    "hermes agent",
    "fairy",
    # Local inference / coding providers (not standalone apps)
    "qwen",
    "gemma",
    "ollama",
    "llama",
})


def is_internal_agent_name(target: str) -> bool:
    """Return True if target names an internal agent/system, not a desktop app."""
    t = target.lower().strip()
    return t in _NON_LAUNCHABLE_INTERNAL_NAMES


# ------------------------------------------------------------------------------
# History conversion
# ------------------------------------------------------------------------------
def _convert_fairy_history_to_hermes(fairy_history: List[Any]) -> List[Dict[str, Any]]:
    if not fairy_history:
        return []
    hermes_messages = []
    for entry in fairy_history:
        if isinstance(entry, dict):
            msg = {
                "role": entry.get("role", "user"),
                "content": entry.get("content", ""),
            }
            if "tool_call_id" in entry:
                msg["tool_call_id"] = entry["tool_call_id"]
            if "name" in entry:
                msg["name"] = entry["name"]
            hermes_messages.append(msg)
        else:
            hermes_messages.append({"role": "user", "content": str(entry)})
    return hermes_messages


def _convert_hermes_history_to_fairy(hermes_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fairy_history = []
    for msg in hermes_messages:
        if not isinstance(msg, dict):
            continue
        fairy_history.append({
            "role": msg.get("role", "assistant"),
            "content": msg.get("content", ""),
            "tool_call_id": msg.get("tool_call_id"),
            "name": msg.get("name"),
        })
    return fairy_history


# ------------------------------------------------------------------------------
# Core bridge
# ------------------------------------------------------------------------------
class HermesBridgeError(Exception):
    pass


def is_hermes_available() -> bool:
    """Return True if Hermes can be used (in-process OR subprocess fallback)."""
    if _AIAgent is not None:
        return True
    return os.path.isfile(HERMES_VENV_PYTHON)


def get_hermes_import_error() -> Optional[str]:
    """Return the in-process import error, or None if import succeeded."""
    return _HERMES_IMPORT_ERROR


# ------------------------------------------------------------------------------
# Per-call language injection
# ------------------------------------------------------------------------------
# Hermes' AIAgent is created ONCE (cached) and reused across turns. The system
# prompt is therefore fixed for the lifetime of the agent. To enforce
# per-message language mirroring, we tag the *user message* on every turn with
# an explicit language instruction derived from detect_language(user_text).
# This survives in the cached agent without mutating the static system prompt.
def _inject_language_tag(user_text: str) -> Tuple[str, str]:
    """
    Return (tagged_user_text, detected_language).

    Prepends a one-line per-call language directive so the model is told
    explicitly "The user's message is in X. Reply in X." even though the
    system prompt is static. Tagged text is a single block the model
    reads as a user-role instruction before the actual user content.
    """
    lang = _detect_language(user_text or "")
    if lang == "bn":
        tagged = (
            "[The user's message is in Bangla (Bengali). Reply in Bangla.]\n"
            + (user_text or "")
        )
    else:
        tagged = (
            "[The user's message is in English. Reply in English.]\n"
            + (user_text or "")
        )
    return tagged, lang


def _run_turn_in_process(
    user_text: str,
    hermes_history: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Fast path: use the cached AIAgent inside Fairy's Python process."""
    # Get the cached agent (created at startup or on first call).
    # The cache is the SINGLE point of AIAgent instantiation — we never
    # create a fresh one here. If the cache can't produce an agent
    # (Hermes unavailable), we raise so the caller can route elsewhere.
    agent = _get_cached_agent()
    if agent is None:
        raise HermesBridgeError(
            "Hermes in-process agent unavailable (cache returned None). "
            "Caller should route to subprocess fallback or main_brain."
        )

    # Per-message language mirroring: tag the user message so the cached
    # agent is reminded of the user's language on every turn.
    tagged_user_text, _lang = _inject_language_tag(user_text)

    # Hermes — and modules it owns, like its SQLite-backed state.db /
    # async_delegation layer (tools/async_delegation.py calls
    # apply_wal_with_fallback(..., db_label="state.db (async_delegation)"))
    # — sometimes print diagnostics directly instead of respecting
    # quiet_mode (e.g. the one-time "SQLite is vulnerable to the WAL-reset
    # bug" notice, which is a process-level warning, not an agent-level
    # one). Because this path runs Hermes IN Fairy's own process, with no
    # subprocess boundary to isolate it, that print/warning used to land
    # directly in the same stream as Fairy's Rich-rendered chat output.
    # Capture BOTH stdout and stderr — this kind of notice is commonly
    # emitted via `print(..., file=sys.stderr)` or `warnings.warn()`
    # (whose default handler writes to stderr), which a stdout-only
    # redirect would completely miss. Still visible via
    # HERMES_BRIDGE_DEBUG=1. NOTE: don't call _dbg() from inside this
    # block — _dbg() itself writes to stderr, so it would just capture
    # its own debug line instead of actually logging it.
    _stdout_capture = io.StringIO()
    _stderr_capture = _RealFdStderrCapture(sys.__stderr__)
    try:
        with contextlib.redirect_stdout(_stdout_capture), \
             contextlib.redirect_stderr(_stderr_capture):
            result = agent.run_conversation(
                user_message=tagged_user_text,
                conversation_history=hermes_history,
            )
        _dbg("[HERMES_BRIDGE] AIAgent.run_conversation completed")
    except Exception as exc:
        tb = traceback.format_exc()
        leaked = (_stdout_capture.getvalue() + _stderr_capture.getvalue()).strip()
        if leaked:
            _dbg(f"[HERMES_BRIDGE] Hermes stdout/stderr before failure (suppressed from chat):\n{leaked}")
        _dbg(f"[HERMES_BRIDGE] ❌ run_conversation EXCEPTION: {exc}")
        raise HermesBridgeError(f"Hermes run_conversation failed: {exc}\n{tb}")

    leaked = (_stdout_capture.getvalue() + _stderr_capture.getvalue()).strip()
    if leaked:
        _dbg(f"[HERMES_BRIDGE] Hermes printed directly to stdout/stderr (suppressed from chat):\n{leaked}")

    response_text = result.get("final_response", "")
    updated_hermes_history = result.get("messages", [])
    _dbg(f"[HERMES_BRIDGE] ✅ run_conversation success. Response preview: '{response_text[:100]}...'")
    _dbg(f"[HERMES_BRIDGE] Total messages in history: {len(updated_hermes_history)}")

    updated_fairy_history = _convert_hermes_history_to_fairy(updated_hermes_history)
    return response_text, updated_fairy_history


def _run_turn_subprocess(
    user_text: str,
    hermes_history: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Fallback path: run the turn inside Hermes' own venv Python."""
    if not os.path.isfile(HERMES_VENV_PYTHON):
        raise HermesBridgeError(
            f"Hermes venv Python missing: {HERMES_VENV_PYTHON}\n"
            f"In-process import error was: {_HERMES_IMPORT_ERROR}"
        )

    # Per-message language mirroring (same tag as in-process path).
    tagged_user_text, _lang = _inject_language_tag(user_text)

    agent_kwargs: Dict[str, Any] = {
        "quiet_mode": True,
        "platform": "cli",  # Full toolset via hermes-cli toolset
    }
    if True:  # Always resolve; _resolve_fairy_model() is cached
        agent_kwargs["model"] = _resolve_fairy_model()
    if _FAIRY_EPHEMERAL_SYSTEM_PROMPT:
        agent_kwargs["ephemeral_system_prompt"] = _FAIRY_EPHEMERAL_SYSTEM_PROMPT
    if _FAIRY_HERMES_DISABLED_TOOLSETS is not None:
        agent_kwargs["disabled_toolsets"] = _FAIRY_HERMES_DISABLED_TOOLSETS

    # Build a self-contained runner script
    runner_code = (
        "import json, sys\n"
        f"sys.path.insert(0, {repr(HERMES_ROOT)})\n"
        "from run_agent import AIAgent\n"
        f"agent_kwargs = json.loads({repr(json.dumps(agent_kwargs))})\n"
        "agent = AIAgent(**agent_kwargs)\n"
        "result = agent.run_conversation(\n"
        f"    user_message={repr(tagged_user_text)},\n"
        f"    conversation_history=json.loads({repr(json.dumps(hermes_history))}),\n"
        ")\n"
        "print(json.dumps(result))\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        fh.write(runner_code)
        runner_path = fh.name

    try:
        proc = subprocess.run(
            [HERMES_VENV_PYTHON, runner_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise HermesBridgeError(
                f"Hermes subprocess failed (exit {proc.returncode}):\n"
                f"stderr: {proc.stderr.strip()}\n"
                f"stdout: {proc.stdout.strip()}"
            )
        # The runner script's last line is always `print(json.dumps(result))`,
        # but Hermes' own internals can print diagnostics (e.g. a one-time
        # SQLite WAL warning) to stdout BEFORE that — so parse only the last
        # non-empty line as JSON instead of assuming the whole stream is
        # clean JSON. Anything above it is preserved for debugging.
        stdout_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if not stdout_lines:
            raise HermesBridgeError(
                f"Hermes subprocess produced no stdout.\nstderr: {proc.stderr.strip()}"
            )
        json_line = stdout_lines[-1]
        leaked = "\n".join(stdout_lines[:-1]).strip()
        if leaked:
            _dbg(f"[HERMES_BRIDGE] Hermes subprocess printed extra stdout (suppressed from chat):\n{leaked}")
        result = json.loads(json_line)
    except json.JSONDecodeError as exc:
        raise HermesBridgeError(
            f"Hermes subprocess returned invalid JSON: {exc}\n"
            f"stdout: {proc.stdout.strip()}\n"
            f"stderr: {proc.stderr.strip()}"
        )
    finally:
        try:
            os.unlink(runner_path)
        except OSError:
            pass

    response_text = result.get("final_response", "")
    updated_hermes_history = result.get("messages", [])
    _dbg(f"[HERMES_BRIDGE] ✅ subprocess success. Response preview: '{response_text[:100]}...'")
    _dbg(f"[HERMES_BRIDGE] Total messages in history: {len(updated_hermes_history)}")

    updated_fairy_history = _convert_hermes_history_to_fairy(updated_hermes_history)
    return response_text, updated_fairy_history


def run_turn(
    user_text: str,
    history: List[Any],
    on_status=None,
) -> Tuple[str, List[Dict[str, Any]]]:
    global _LAST_PROVIDER
    """
    Execute one turn through Hermes with automatic hybrid failover.

    Routing:
      1. Check Ollama availability (fast, 5s timeout, 30s cached).
      2. If Ollama is available → try Hermes (existing Gemma 4 / Ollama path,
         in-process — this is the fast path and stays the normal path).
      3. If Ollama is unavailable → skip Hermes entirely, use main_brain.chat()
         which falls back to OpenRouter `openrouter/free`.
      4. If Hermes call fails with a provider/Ollama error → fall back to
         main_brain.chat() (OpenRouter).
      5. If Hermes call fails with a genuine (non-provider) error — i.e.
         Hermes itself broke, not the provider it's talking to — retry
         exactly once via an isolated subprocess (_run_turn_subprocess).
         This does NOT cascade into the OpenRouter fallback: either the
         subprocess retry succeeds, or the ORIGINAL in-process error is
         raised. A genuine Hermes bug never silently burns an OpenRouter
         call too.
      6. If both providers fail → raise HermesBridgeError.

    Raises on total failure so the caller can choose how to recover.
    """
    _dbg(f"[HERMES_BRIDGE] run_turn called. user_text='{user_text[:60]}...'")
    _dbg(f"[HERMES_BRIDGE] _AIAgent present={_AIAgent is not None}")

    hermes_history = _convert_fairy_history_to_hermes(history)
    _dbg(f"[HERMES_BRIDGE] Converted history: {len(hermes_history)} messages")

    # ── Quick Ollama availability check ─────────────────────────────────────────
    # This avoids the ~50-second retry loop in Hermes when Ollama is down.
    ollama_ok = _quick_ollama_check()
    _dbg(f"[HERMES_BRIDGE] Ollama availability check: {ollama_ok}")

    # ── Priority 1: Ollama available → try Hermes ──────────────────────────────
    # Try Hermes in-process. If Ollama rejects structured content (HTTP 400),
    # _is_hermes_provider_error() catches it and we fall back to main_brain.
    _skip_hermes_in_process = False

    if ollama_ok and _AIAgent is not None and not _skip_hermes_in_process:
        try:
            _dbg(f"[HERMES_BRIDGE] Ollama available — attempting Hermes ({_resolve_fairy_model()})")
            _LAST_PROVIDER = "hermes"
            return _run_turn_in_process(user_text, hermes_history)
        except HermesBridgeError as exc:
            # Hermes call failed — check if it's a provider/Ollama error
            if _is_hermes_provider_error(exc):
                _dbg(f"[HERMES_BRIDGE] Hermes failed with provider error: {exc}. Falling back to main_brain.")
                # Fall through to main_brain fallback below
            else:
                # Genuine (non-provider) application error in-process — this
                # is not "Ollama/provider is unhappy", it's "Hermes itself
                # broke". Retry exactly once, isolated in a subprocess, in
                # case the crash was caused by in-process state (a bad
                # import, a corrupted module-level cache, a crashed native
                # extension, etc.) that a fresh interpreter wouldn't hit.
                # This does NOT fall through to the OpenRouter fallback —
                # a genuine Hermes bug isn't a provider outage, and we don't
                # want a single turn silently spending three providers'
                # worth of calls (in-process Hermes + subprocess Hermes +
                # OpenRouter) without that being a deliberate decision.
                _dbg(f"[HERMES_BRIDGE] In-process Hermes crashed (non-provider): {exc}. Retrying once via subprocess.")
                try:
                    _LAST_PROVIDER = "subprocess_retry"
                    return _run_turn_subprocess(user_text, hermes_history)
                except HermesBridgeError as subprocess_exc:
                    _dbg(f"[HERMES_BRIDGE] Subprocess retry also failed: {subprocess_exc}. Surfacing original error.")
                    raise exc

    # ── Ollama unavailable OR Hermes call failed OR gemma4 model ─────────────────
    # Use main_brain.chat() which handles Ollama→OpenRouter fallback.
    # gemma4 always routes here (Hermes in-process doesn't support Ollama's format).
    _dbg("[HERMES_BRIDGE] Routing to main_brain fallback (direct Ollama → OpenRouter free)")
    _LAST_PROVIDER = "main_brain"
    return _run_turn_via_main_brain(user_text, hermes_history)


def run_turn_safe(
    user_text: str,
    history: List[Any],
    on_status=None,
) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Graceful wrapper. Returns (success, reply_text, updated_history).
    """
    _dbg(f"[HERMES_BRIDGE] run_turn_safe called")
    try:
        reply_text, updated_history = run_turn(user_text, history, on_status)
        _dbg(f"[HERMES_BRIDGE] run_turn_safe returning SUCCESS")
        return True, reply_text, updated_history
    except HermesBridgeError as exc:
        # Check if this is a main_brain fallback that succeeded (text-only).
        # The _FALLBACK_RESPONSE sentinel holds the full reply — if set, return
        # it as success=True so the caller can show the answer + a fallback notice.
        global _FALLBACK_RESPONSE
        if _FALLBACK_RESPONSE:
            full_text = _FALLBACK_RESPONSE
            _FALLBACK_RESPONSE = ""  # consume so next call doesn't re-use it
            _dbg("[HERMES_BRIDGE] run_turn_safe returning SUCCESS from main_brain fallback")
            return True, full_text, history
        _dbg(f"[HERMES_BRIDGE] run_turn_safe returning FAIL (HermesBridgeError): {exc}")
        return False, f"[Hermes Error] {exc}", history
    except Exception as exc:
        tb = traceback.format_exc()
        _dbg(f"[HERMES_BRIDGE] run_turn_safe returning FAIL (Unexpected): {exc}")
        return False, f"[Hermes Unexpected Error] {exc}\n{tb}", history