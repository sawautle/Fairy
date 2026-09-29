#!/usr/bin/env python3
"""
OpenRouter resilience layer for main_brain.py.

Provides:
  - Retry loop with exponential backoff (1s, 2s, ... capped 60s, max 10 attempts)
    for transient errors (429, 5xx, network).
  - 401/403 short-circuit → key-recovery mode.
  - Key-recovery: terminal prompt or Discord DM, validate + persist new key.
  - Ollama fallback with "[Ollama fallback]" prefix.

Design principles:
  - Pure decision functions are fully deterministic and tested with fake transports.
  - Sleep is injected as `sleep_fn` so tests never wait.
  - No new dependencies; uses stdlib urllib only.
  - Timeouts on all network calls.

Public API (re-exports from main_brain for convenience):
    chat_with_resilience, recover_openrouter_key, try_ollama_fallback,
    classify_http_error, is_openrouter_available
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path as _Path
from typing import Any, Callable

# ─── urllib access via sys.modules (enables test-patching) ─────────────────────
# Python caches imported modules in sys.modules. Accessing urlopen via
# sys.modules['urllib.request'].urlopen means we get the CURRENT value of
# that attribute at call time — including any patched binding from tests.
# A top-level `import urllib.request; x = urllib.request.urlopen` would capture
# the pre-patch reference and never see the mock. Using sys.modules fixes it.


_DEFAULT_REQUEST_TIMEOUT = 60.0
"""Default per-request timeout. Never call urlopen without a timeout — a missing
timeout means the underlying SSL socket can hang indefinitely on read()."""


def _urllib_urlopen(req, timeout=None):
    """Proxy to urllib.request.urlopen that is patchable via sys.modules.

    Always passes an explicit timeout (default 60s) to urlopen so the underlying
    socket can never block forever. This is belt-and-suspenders: even when a
    caller forgets to pass a timeout, we don't open an unbounded read on a
    network socket.

    Also sets req.timeout so transports that read the timeout from the request
    object (rather than from the urlopen kwarg) still get a bound.
    """
    _timeout = timeout if timeout is not None else _DEFAULT_REQUEST_TIMEOUT
    try:
        if getattr(req, "timeout", None) is None:
            req.timeout = _timeout
    except Exception:
        # Some Request subclasses may not allow attribute assignment; safe to ignore.
        pass
    return sys.modules["urllib.request"].urlopen(req, timeout=_timeout)


def _urllib_Request(url, data=None, headers=None, method=None):
    """Proxy to urllib.request.Request that is patchable via sys.modules."""
    return sys.modules["urllib.request"].Request(url, data=data, headers=headers, method=method)


# ─── Project root for config path ────────────────────────────────────────────
_SCRIPT_DIR = _Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# ─── Config values ───────────────────────────────────────────────────────────
# Lazy-load so the module can be imported before config.py is available.
def _get_openrouter_key() -> str:
    try:
        from config import OPENROUTER_KEY
        return OPENROUTER_KEY or os.environ.get("OPENROUTER_API_KEY", "")
    except Exception:
        return os.environ.get("OPENROUTER_API_KEY", "")

def _get_ollama_url() -> str:
    try:
        from config import OLLAMA_URL
        return OLLAMA_URL or "http://localhost:11434"
    except Exception:
        return "http://localhost:11434"

def _get_model_brain() -> str:
    try:
        from config import MODEL_BRAIN
        return MODEL_BRAIN or "gemma4"
    except Exception:
        return "gemma4"

# ─── Error classification ─────────────────────────────────────────────────────
# Pure: no I/O, no side-effects, fully deterministic.

TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
"""Status codes that indicate a temporary condition — retry is appropriate."""

AUTH_STATUS_CODES = frozenset({401, 403})
"""Status codes that indicate an auth problem — never retry, enter key recovery."""


def classify_http_error(status_code: int | None) -> str:
    """
    Classify an HTTP error status code.

    Returns one of:
        "auth"    - 401 or 403 — invalid/missing key; do not retry.
        "transient" - 429 or 5xx — retryable with backoff.
        "client"  - 400 or other 4xx — do not retry.
        "none"    - no HTTP error (network error, etc. — treated as transient).

    Pure function.
    """
    if status_code is None:
        return "none"  # Network error is treated as transient
    if status_code in AUTH_STATUS_CODES:
        return "auth"
    if status_code in TRANSIENT_STATUS_CODES:
        return "transient"
    if 400 <= status_code < 500:
        return "client"
    return "none"


def should_retry(error_class: str) -> bool:
    """
    Return True when a retry is appropriate for the given error class.

    Pure function.

    Args:
        error_class: one of "auth", "transient", "client", "none"
    """
    return error_class in ("transient", "none")


# ─── Backoff ──────────────────────────────────────────────────────────────────

def compute_backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """
    Exponential backoff: min(base * 2**attempt, cap).

    Pure function.

    Args:
        attempt: 0-based attempt number (0 = first backoff).
        base:    initial delay in seconds (default 1.0).
        cap:     maximum delay in seconds (default 60.0).

    Returns:
        Delay in seconds to sleep before the next retry.
    """
    return min(base * (2 ** attempt), cap)


def parse_retry_after(headers: dict | None) -> float | None:
    """
    Parse the Retry-After header from an HTTP response.

    Accepts both:
      - Integer seconds: "120"
      - HTTP-date: "Wed, 21 Oct 2015 07:28:00 GMT"

    Pure function (no I/O).

    Args:
        headers: dict of lowercase header name -> value, or None.

    Returns:
        Delay in seconds, or None if the header is absent or unparseable.
    """
    if not headers:
        return None
    # http.client.HTTPMessage stores keys title-cased ("Retry-After"); tests
    # and ad-hoc dicts often use lowercase. Try both.
    raw = (
        headers.get("retry-after")
        or headers.get("Retry-After")
        or None
    )
    if raw is None:
        return None
    raw = raw.strip()
    # Integer form: seconds
    try:
        return float(raw)
    except ValueError:
        pass
    # HTTP-date form: try to compute delta from now
    try:
        from email.utils import parsedate_to_datetime
        deadline = parsedate_to_datetime(raw)
        delta = (deadline - datetime_now_utc()).total_seconds()
        return max(0.0, delta)
    except Exception:
        return None


def datetime_now_utc():
    """Return the current UTC datetime. Extracted so it can be patched in tests."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


# ─── OpenRouter availability probe ──────────────────────────────────────────
# 5-minute TTL cache, thread-safe.

_openrouter_health_cache: dict[str, Any] = {
    "available": False,
    "ts": 0.0,
}
_health_lock = threading.Lock()
_OR_HEALTH_TTL = 300.0  # 5 minutes


def is_openrouter_available(sleep_fn: Callable[[float], None] | None = None) -> bool:
    """
    Lightweight probe of OpenRouter availability.

    Makes a single cheap GET request to /api/v1/models with a short timeout.
    Result is cached for 5 minutes (same pattern as is_vision_capable).

    This is used to recover back to OpenRouter after an Ollama-fallback turn.

    Args:
        sleep_fn: only used if a previous result was stale and we re-check.

    Returns:
        True if OpenRouter responds without auth errors.
    """
    key = _get_openrouter_key()
    if not key:
        return False

    now = time.time()
    with _health_lock:
        stale = (now - _openrouter_health_cache["ts"]) >= _OR_HEALTH_TTL

    if not stale:
        return _openrouter_health_cache["available"]

    # Actual probe (outside lock to avoid holding lock during I/O)
    available = _probe_openrouter(key)

    with _health_lock:
        _openrouter_health_cache["available"] = available
        _openrouter_health_cache["ts"] = now

    return available


def _probe_openrouter(key: str, timeout: float = 10.0) -> bool:
    """
    Perform the actual HTTP probe of OpenRouter /api/v1/models.

    Returns True on any non-auth error (transient/unavailable are "still reachable").
    Returns False only on 401/403 (key definitely bad) or network error.

    Pure in the sense that it only touches the network.
    """
    import urllib.error

    url = "https://openrouter.ai/api/v1/models"
    req = _urllib_Request(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="GET",
    )
    try:
        with _urllib_urlopen(req, timeout=timeout) as resp:
            resp.read()
            return True
    except urllib.error.HTTPError as e:
        if e.code in AUTH_STATUS_CODES:
            return False  # Key is bad
        return True      # Other errors mean server is reachable
    except Exception:
        return False     # Network error — not reachable


def invalidate_openrouter_health() -> None:
    """Clear the OpenRouter health cache so the next turn re-probes."""
    with _health_lock:
        _openrouter_health_cache["ts"] = 0.0


# ─── Key persistence ──────────────────────────────────────────────────────────

def _config_path() -> _Path:
    return _PROJECT_ROOT / "config" / "api_keys.json"


def persist_openrouter_key(key: str) -> bool:
    """
    Persist an OpenRouter API key to config/api_keys.json.

    Merges with the existing file (preserves other keys like gemini_api_key).
    Falls back to setting the env var if the file is not writable.

    Returns True on success, False on failure.
    """
    path = _config_path()
    try:
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        updated = {**existing, "openrouter_api_key": key}
        path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        return True
    except OSError:
        # File not writable — fall back to env var for this process
        os.environ["OPENROUTER_API_KEY"] = key
        return True  # Treat as success; env var will be used


# ─── Key validation ──────────────────────────────────────────────────────────

def validate_openrouter_key(key: str, timeout: float = 10.0) -> bool:
    """
    Validate an OpenRouter API key with a single cheap GET request.

    Returns True if the key is accepted (any non-401/403 response or
    network success). Returns False on 401/403 or network failure.

    Pure in the sense that it only touches the network.

    Args:
        key:     the API key to validate.
        timeout: request timeout in seconds.

    Returns:
        True if the key is valid, False otherwise.
    """
    return _probe_openrouter(key, timeout)


# ─── Key recovery ───────────────────────────────────────────────────────────

def prompt_for_new_openrouter_key(
    input_fn: Callable[[str], str] | None = None,
) -> str | None:
    """
    Prompt the user for a new OpenRouter API key.

    Terminal use: reads from stdin.
    The caller should pass `input` for testability.

    Args:
        input_fn: callable that takes a prompt string and returns user input.
                  Defaults to Python's built-in `input`.

    Returns:
        The new key (stripped), or None if the user pressed Enter
        (signals Ollama fallback).
    """
    if input_fn is None:
        input_fn = input
    try:
        raw = input_fn(
            "Master, OpenRouter isn't responding (auth error). "
            "Paste a new key (or press Enter to use Ollama fallback): "
        )
        key = raw.strip()
        return key if key else None
    except (EOFError, OSError):
        return None


def recover_openrouter_key(
    input_fn: Callable[[str], str] | None = None,
    validate_fn: Callable[[str], bool] | None = None,
    persist_fn: Callable[[str], bool] | None = None,
) -> str | None:
    """
    Full key-recovery flow:

      1. Prompt the user for a new key.
      2. If they press Enter → return None (Ollama fallback).
      3. Validate the key with a cheap probe request.
      4. If invalid → print error and return None.
      5. If valid → persist the key and return it.

    All three callbacks default to the real implementations so the function
    works without arguments in production.

    Args:
        input_fn:   input() replacement for the prompt.
        validate_fn: validate_openrouter_key replacement.
        persist_fn: persist_openrouter_key replacement.

    Returns:
        A valid, persisted key, or None if recovery failed/was declined.
    """
    if validate_fn is None:
        validate_fn = validate_openrouter_key
    if persist_fn is None:
        persist_fn = persist_openrouter_key

    key = prompt_for_new_openrouter_key(input_fn)
    if not key:
        return None

    if not validate_fn(key):
        print("That key doesn't work either. Falling back to Ollama.")
        return None

    if not persist_fn(key):
        print("Couldn't save the new key, but using it for this session.")

    # Invalidate the OR health cache so the next probe uses the new key.
    invalidate_openrouter_health()
    return key


# ─── Ollama fallback ──────────────────────────────────────────────────────────

def try_ollama_fallback(
    model: str | None = None,
    messages: list | None = None,
    tools: list | None = None,
    options: dict | None = None,
) -> tuple[dict, str]:
    """
    Route a turn to Ollama as a fallback when OpenRouter is unavailable.

    Marks the response content with "[Ollama fallback]" so the user knows
    which brain answered.

    The Ollama model is taken from the FAIRY_OLLAMA_MODEL env var,
    defaulting to "llama3.1:8b".

    Args:
        model:    requested model name (informational only).
        messages: conversation history.
        tools:    tool definitions.
        options:  inference options.

    Returns:
        (response_dict, "ollama_fallback") where response_dict is in
        Ollama ChatResponse shape and content is prefixed with
        "[Ollama fallback] ".
    """
    from controller import ollama_client

    fallback_model = os.environ.get("FAIRY_OLLAMA_MODEL", "llama3.1:8b")
    requested = model or _get_model_brain()

    try:
        resp = ollama_client.chat(
            model=fallback_model,
            messages=messages or [],
            tools=tools,
            options=options,
            keep_alive=0,
        )
    except Exception as exc:
        # Last resort: any Ollama error is surfaced as a RuntimeError so
        # the caller can decide how to handle total failure.
        raise RuntimeError(
            f"OpenRouter unavailable and Ollama fallback also failed: {exc}"
        ) from exc

    # Mark the response so the user knows this came from Ollama.
    msg = resp.get("message") or {}
    content = msg.get("content") or ""
    if content and not content.startswith("[Ollama fallback]"):
        msg["content"] = f"[Ollama fallback] {content}"
    # Always ensure a 'message' key is present (defensive: some Ollama error
    # bodies omit it entirely; downstream code assumes Ollama ChatResponse shape).
    if "message" not in resp or resp["message"] is None:
        resp["message"] = {"role": "assistant", "content": content}
    else:
        resp["message"] = msg

    return resp, "ollama_fallback"


# ─── OpenRouter chat with resilience ─────────────────────────────────────────

def _make_openrouter_request(
    messages: list,
    tools: list | None,
    options: dict | None,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> tuple[Any, dict]:
    """
    Build the urllib Request and payload for an OpenRouter /chat/completions call.

    Pure in the sense that it only constructs data structures.

    Returns (req, payload).
    """
    key = _get_openrouter_key()
    num_predict = (options or {}).get("num_predict")

    payload: dict[str, Any] = {
        "model": "openrouter/free",
        "messages": messages,
        "max_tokens": num_predict or max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://fairy.local",
        "X-Title": "Fairy AI",
    }
    url = "https://openrouter.ai/api/v1/chat/completions"
    req = _urllib_Request(url, data=json.dumps(payload).encode("utf-8"),
                          headers=headers, method="POST")
    return req, payload


def _openrouter_request_with_resilience(
    messages: list,
    tools: list | None,
    options: dict | None,
    max_retries: int = 10,
    timeout: float = 60.0,
    sleep_fn: Callable[[float], None] | None = None,
    on_auth_failure: Callable[[], str | None] | None = None,
) -> str:
    """
    Execute an OpenRouter /chat/completions request with retry and backoff.

    Retry logic:
      - Max 10 retries for transient errors (429, 5xx, network).
      - Never retries on 401/403 — calls on_auth_failure() instead.
      - Never retries on 400 (client error).
      - Exponential backoff: 1s, 2s, 4s ... capped at 60s.
      - Honors Retry-After header when present.
      - Each attempt has a timeout of `timeout` seconds.

    Args:
        messages:      conversation history.
        tools:         tool definitions.
        options:       inference options (num_predict used for max_tokens).
        max_retries:   maximum retry attempts (default 10).
        timeout:       per-attempt timeout in seconds.
        sleep_fn:      injected sleep function (tests pass a no-op).
        on_auth_failure: called when 401/403 is encountered; should return
                        a new valid key or None. If it returns a new key,
                        retries once with that key. If None, raises
                        AuthError.

    Returns:
        The raw JSON response string from OpenRouter.

    Raises:
        AuthError:     on 401/403 when key recovery fails or is not configured.
        RuntimeError:  on client errors (400) or after all retries exhausted.
    """
    # Import locally so that tests patching urllib.request.urlopen work correctly
    import urllib.error

    if sleep_fn is None:
        sleep_fn = time.sleep

    last_err: Exception | None = None
    current_key = _get_openrouter_key()
    recovery_attempted = False

    for attempt in range(max_retries + 1):
        req, payload = _make_openrouter_request(messages, tools, options)
        retry_after_override: float | None = None

        try:
            with _urllib_urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                # Extract Retry-After if present (guard against fakes/missing attrs).
                # Even on 2xx, some rate-limited responses include it for the next call.
                try:
                    resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                except Exception:
                    resp_headers = {}
                parsed_retry_after = parse_retry_after(resp_headers)
                if parsed_retry_after is not None:
                    retry_after_override = parsed_retry_after

                # Check for JSON-embedded error (HTTP 200 with error object)
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and "error" in parsed:
                        err = parsed["error"]
                        err_code = err.get("code", "?")
                        err_msg = err.get("message", str(err))
                        # Check for auth error embedded in 200 response
                        code = int(err_code) if str(err_code).isdigit() else None
                        cls = classify_http_error(code)
                        if cls == "auth":
                            raise AuthError(f"OpenRouter JSON auth error: {err_msg[:200]}")
                        raise RuntimeError(f"OpenRouter JSON error {err_code}: {err_msg[:300]}")
                except json.JSONDecodeError:
                    pass

                return raw

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            cls = classify_http_error(e.code)
            last_err = RuntimeError(f"OpenRouter HTTP {e.code}: {body[:300]}")
            # Extract Retry-After from the error response headers (for 503/429 retries)
            try:
                e_headers = {k.lower(): v for k, v in e.headers.items()}
            except Exception:
                e_headers = {}
            retry_after_override = parse_retry_after(e_headers)

            if cls == "auth":
                if on_auth_failure is not None and not recovery_attempted:
                    recovery_attempted = True
                    new_key = on_auth_failure()
                    if new_key:
                        # Replace the key in the module-level config
                        os.environ["OPENROUTER_API_KEY"] = new_key
                        # Also patch the config CFG dict directly if available
                        try:
                            import config as _cfg
                            _cfg.OPENROUTER_KEY = new_key
                            _cfg.CFG["openrouter_api_key"] = new_key
                        except Exception:
                            pass
                        current_key = new_key
                        # Retry with the new key (do NOT count as a retry attempt)
                        continue
                    # Recovery declined or failed → raise AuthError
                raise AuthError(
                    f"OpenRouter auth error ({e.code}). "
                    f"Key recovery failed or was declined."
                ) from last_err

            if cls == "client":
                # 400 and other 4xx — never retry
                raise last_err

            # Transient: retry with backoff
            last_err = RuntimeError(f"OpenRouter HTTP {e.code} (attempt {attempt + 1}/{max_retries + 1}): {body[:200]}")

        except urllib.error.URLError as e:
            last_err = RuntimeError(
                f"OpenRouter request failed (network/timeout): {e.reason}"
            )

        except TimeoutError:
            last_err = RuntimeError(f"OpenRouter request timed out after {timeout}s")

        except AuthError:
            raise  # Already handled above

        except Exception as e:
            last_err = RuntimeError(f"OpenRouter unexpected error: {e}")

        # Transient or network error: sleep and retry
        if attempt < max_retries:
            # Honor Retry-After if present, otherwise compute backoff
            delay = retry_after_override if retry_after_override is not None else compute_backoff(attempt, base=1.0, cap=60.0)
            sleep_fn(delay)

    # All retries exhausted
    raise RuntimeError(
        f"OpenRouter unavailable after {max_retries + 1} attempts. "
        f"Last error: {last_err}"
    ) from last_err


class AuthError(Exception):
    """Raised when OpenRouter returns a 401/403 and key recovery fails or is declined."""
    pass


def chat_with_resilience(
    model: str,
    messages: list,
    tools: list | None = None,
    options: dict | None = None,
    max_retries: int = 10,
    timeout: float = 60.0,
    sleep_fn: Callable[[float], None] | None = None,
    key_recovery_fn: Callable[[], str | None] | None = None,
) -> tuple[dict, str]:
    """
    OpenRouter chat with full resilience: retries, backoff, key recovery, Ollama fallback.

    This is the drop-in replacement for the old `_openrouter_chat()`.

    Args:
        model:           requested model name (informational; uses openrouter/free internally).
        messages:        conversation history.
        tools:           tool definitions.
        options:         inference options.
        max_retries:     maximum retry attempts (default 10).
        timeout:         per-attempt timeout in seconds.
        sleep_fn:        injected sleep (tests pass no-op).
        key_recovery_fn: called on 401/403; should prompt user and return new key or None.
                         Defaults to recover_openrouter_key().

    Returns:
        (response_dict, "openrouter") in Ollama ChatResponse shape.

    Raises:
        AuthError:  on 401/403 when key recovery fails.
        RuntimeError: on total failure after all retries.
    """
    # Resolve key_recovery_fn: explicit arg > ctx > default.
    if key_recovery_fn is None:
        # Check the module-level resilience context set by configure_resilience.
        # This lets Discord wire in a DM-based recovery without patching globals.
        try:
            from controller import main_brain as _mb
            ctx = getattr(_mb, "_resilience_ctx", None)
            if ctx and ctx.get("key_recovery_fn"):
                key_recovery_fn = ctx["key_recovery_fn"]
            else:
                key_recovery_fn = recover_openrouter_key
        except Exception:
            key_recovery_fn = recover_openrouter_key

    # Resolve sleep_fn: explicit arg > ctx > time.sleep.
    if sleep_fn is None:
        try:
            from controller import main_brain as _mb
            ctx = getattr(_mb, "_resilience_ctx", None)
            if ctx and ctx.get("sleep_fn"):
                sleep_fn = ctx["sleep_fn"]
        except Exception:
            pass

    try:
        raw = _openrouter_request_with_resilience(
            messages=messages,
            tools=tools,
            options=options,
            max_retries=max_retries,
            timeout=timeout,
            sleep_fn=sleep_fn,
            on_auth_failure=key_recovery_fn,
        )
    except AuthError:
        raise  # Key recovery failed; let caller decide (Ollama fallback)
    except RuntimeError:
        raise  # All retries exhausted; let caller decide (Ollama fallback)

    try:
        resp_json = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenRouter returned malformed JSON: {raw[:300]}") from e

    # Convert to Ollama shape
    converted = _openrouter_to_ollama_shape(resp_json, "openrouter/free")
    return converted, "openrouter"


def _openrouter_to_ollama_shape(resp_json: dict, model_id: str) -> dict:
    """
    Convert OpenAI /chat/completions response to Ollama ChatResponse shape.

    Handles: assistant content, tool_calls (OpenAI → Ollama format), role.
    """
    msg = resp_json.get("choices", [{}])[0].get("message", {})
    raw_content = msg.get("content") or ""
    raw_tool_calls = msg.get("tool_calls") or []

    ollama_tool_calls = []
    for tc in raw_tool_calls:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments")
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
