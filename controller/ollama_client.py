"""
Optimized thin wrapper around the `ollama` Python package.
VRAM-lean: keep_alive=0 unloads after response; num_ctx=2048 shrinks KV-cache.

DIAGNOSABILITY (FIX: Ollama hang → honest error):
  The ollama Python package uses httpx internally with timeout=None
  (unbounded). If Ollama stalls, the call hangs forever. We bypass
  the module-level ollama.chat() (which creates an unbounded client)
  and call ollama.Client(..., timeout=httpx.Timeout(...)) directly
  with explicit connection + read timeouts.

  On any httpx/connection/timeout error, we:
    1. Log the request payload (truncated) and the failure type to
       fairy_debug.log via the agent_controller._debug hook.
    2. Raise a typed RuntimeError subclass so callers (main_brain,
       resilience layer, delegate) can distinguish timeout from
       connection-failed from other failures and surface an honest
       error to the user instead of a spinner that never ends.
"""

import json
import time
import logging
from typing import Any

import ollama
import httpx

# ── Debug logging (mirrors agent_controller._debug pattern) ───────────────────
_logger = logging.getLogger("fairy.ollama_client")

# Default timeouts: 10s to connect, 120s to read a response.
# With higher context (8192 tokens), Gemma4 needs more time to process.
DEFAULT_CONNECT_TIMEOUT = float(10.0)
DEFAULT_READ_TIMEOUT = float(120.0)


class OllamaConnectionError(RuntimeError):
    """Ollama is unreachable (server down, port closed, network error)."""
    pass


class OllamaTimeoutError(RuntimeError):
    """Ollama accepted the connection but did not respond within the read timeout."""
    pass


class OllamaError(RuntimeError):
    """Ollama returned a non-success status (4xx/5xx) or another HTTP error."""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _debug(label: str, data: Any = None) -> None:
    """Append a structured OLLAMA_CLIENT line to the debug log.
    Tries agent_controller._debug first; falls back to a logger line.
    """
    try:
        from controller.agent_controller import _debug as _ac_debug
        _ac_debug(label, data)
        return
    except (ImportError, TypeError):
        pass
    if data is None:
        _logger.info(label)
    else:
        try:
            payload = json.dumps(data, ensure_ascii=False, default=str)
        except Exception:
            payload = str(data)
        _logger.info(f"{label} | {payload}")


def _make_client(connect_timeout: float, read_timeout: float) -> ollama.Client:
    """Create an ollama.Client with explicit httpx timeouts.

    Bypasses the module-level ollama._client (which uses timeout=None)
    so we can guarantee a bounded call. The host/port come from the
    OLLAMA_HOST env var, defaulting to localhost:11434.
    """
    timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout)
    return ollama.Client(timeout=timeout)


def _truncate_for_log(value: Any, limit: int = 400) -> str:
    """Truncate a value (dict, list, str) for debug-log readability."""
    try:
        s = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        s = str(value)
    if len(s) > limit:
        return s[:limit] + f"...<truncated {len(s) - limit} chars>"
    return s


def unload(model: str) -> None:
    """
    Force Ollama to unload `model` from VRAM immediately.
    Call this manually before heavy VRAM consumers (e.g., TTS) need the memory.
    """
    try:
        ollama.generate(model=model, prompt="", keep_alive=0)
    except Exception:
        pass


def chat(
    model: str,
    messages: list,
    tools: list | None = None,
    keep_alive: int | str = 0,
    num_ctx: int = 32768,
    options: dict | None = None,
    retries: int = 1,
    retry_delay: float = 3.0,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
):
    """
    Send a chat request to a local Ollama model with aggressive VRAM optimizations.

    keep_alive: 0 = unload from VRAM immediately after this response.
                -1 = keep loaded indefinitely (much faster follow-ups, but
                competes with TTS for VRAM).
    num_ctx: 65536 tokens default. Ollama's own default is 2048, but that's
             far too small for the system prompt + tool schemas + history; 2048
             is the value that makes "23 tokens" exceed the limit and trips a
             spurious "Context length exceeded" on trivial messages.  Hermes
             requires at least 65536 for reliable tool use with Gemma4; 32768
             was insufficient because the system prompt alone exceeds that.
    connect_timeout: seconds to wait for TCP connect (default 10s).
    read_timeout:    seconds to wait for response payload (default 120s).
    retries:         how many retries on OOM (default 1).

    Raises:
      OllamaConnectionError — server unreachable, DNS failure, port closed.
      OllamaTimeoutError    — connected but no response within read_timeout.
      OllamaError           — HTTP non-success status.
    """
    merged_options = {"num_ctx": num_ctx}
    if options:
        merged_options.update(options)

    kwargs = {
        "model": model,
        "messages": messages,
        "keep_alive": keep_alive,
        "options": merged_options,
    }
    if tools:
        kwargs["tools"] = tools

    # ── Log request payload (truncated) ───────────────────────────────
    last_user_text = ""
    try:
        if messages:
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user":
                    last_user_text = str(m.get("content") or "")[:120]
                    break
    except Exception:
        last_user_text = "<unreadable>"
    _debug("OLLAMA_REQUEST", {
        "model": model,
        "num_ctx": num_ctx,
        "keep_alive": keep_alive,
        "messages_count": len(messages) if messages else 0,
        "last_user_text_preview": last_user_text,
        "tools_count": len(tools) if tools else 0,
        "connect_timeout": connect_timeout,
        "read_timeout": read_timeout,
    })

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        client = _make_client(connect_timeout, read_timeout)
        try:
            try:
                response = client.chat(**kwargs)
            finally:
                # Always close the httpx transport so sockets are released.
                try:
                    client.close()
                except Exception:
                    pass

            # ── Log success summary ──────────────────────────────────
            try:
                msg = response.get("message", {}) if isinstance(response, dict) else {}
                _debug("OLLAMA_RESPONSE", {
                    "model": model,
                    "status": "ok",
                    "done_reason": response.get("done_reason") if isinstance(response, dict) else None,
                    "content_chars": len(str(msg.get("content") or "")),
                    "tool_calls": len(msg.get("tool_calls") or []),
                })
            except Exception:
                pass
            return response

        except httpx.ConnectError as exc:
            last_error = OllamaConnectionError(
                f"Could not connect to Ollama at the configured host. "
                f"Verify Ollama is running (try: ollama serve) and that "
                f"OLLAMA_HOST points to the right address. Underlying: {exc}"
            )
            _debug("OLLAMA_CONNECT_FAIL", {
                "model": model,
                "error": str(exc),
                "attempt": attempt,
                "retries_left": retries - attempt,
            })
            # Connection errors are not retried here (Ollama being down won't
            # resolve in 3s); caller decides whether to fall back.
            break

        except httpx.TimeoutException as exc:
            last_error = OllamaTimeoutError(
                f"Ollama did not respond within {read_timeout}s for model "
                f"{model!r}. The model may be loading, busy, or the request "
                f"may be too large. Underlying: {exc}"
            )
            _debug("OLLAMA_TIMEOUT", {
                "model": model,
                "read_timeout": read_timeout,
                "connect_timeout": connect_timeout,
                "attempt": attempt,
                "retries_left": retries - attempt,
                "error": str(exc),
            })
            break

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            last_error = OllamaError(
                f"Ollama returned HTTP {status}: {exc.response.text[:200] if exc.response is not None else exc}",
                status_code=status,
            )
            _debug("OLLAMA_HTTP_ERROR", {
                "model": model,
                "status_code": status,
                "response_preview": (exc.response.text[:200] if exc.response is not None else None),
            })
            break

        except Exception as exc:
            err_text = str(exc).lower()
            oom_indicators = (
                "out of memory", "cuda", "ggml_assert",
                "cudamalloc", "failed to allocate",
            )
            if any(ind in err_text for ind in oom_indicators) and attempt < retries:
                _debug("OLLAMA_OOM_RETRY", {
                    "model": model,
                    "attempt": attempt,
                    "error": str(exc),
                })
                unload(model)
                time.sleep(retry_delay)
                last_error = exc
                continue
            _debug("OLLAMA_UNEXPECTED_ERROR", {
                "model": model,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            last_error = exc
            break

    if last_error is None:
        last_error = RuntimeError("ollama_client.chat failed without a recorded error")
    raise last_error
