#!/usr/bin/env python3
"""
controller/vision.py — Shared image-intake pipeline for Fairy.

Public API
----------
describe_image(image_bytes, mime, question="") -> str
    Sends the image to the brain via OpenRouter's vision content format.
    Returns a clear error string on failure (never raises into chat path).

is_vision_capable() -> bool
    Cheap cached check of whether the primary brain model supports image input.
    Callers should degrade gracefully when this returns False.

Private helpers (also exported for testability)
-----------------------------------------------
validate_image_size(image_bytes: bytes) -> tuple[bool, str]
    Returns (ok, reason). ok=False means the image is too large; reason
    describes the limit that was exceeded.

_openrouter_vision(messages, model, timeout) -> str
    Low-level OpenRouter /chat/completions call for vision messages.
    Reuses the same transport/auth pattern as main_brain.py without
    duplicating API-key handling.

_is_vision_capable_sync() -> bool
    Uncached capability check — runs at startup and is memoised into
    _vision_capable_cache.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

# ─── Config ───────────────────────────────────────────────────────────────────
# Reuse OPENROUTER_KEY from the same config path that main_brain.py uses,
# avoiding any duplication of key-loading logic.
try:
    from config import OPENROUTER_KEY
except ImportError:
    try:
        import os as _os
        import json as _json
        _cfg = {}
        try:
            _base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            _cfg = _json.loads(
                open(_os.path.join(_base, "config", "api_keys.json"), encoding="utf-8").read()
            )
        except Exception:
            pass
        OPENROUTER_KEY = _cfg.get("openrouter_api_key", "") or _os.environ.get("OPENROUTER_API_KEY", "")
    except Exception:
        OPENROUTER_KEY = ""

_OPENROUTER_CONFIGURED = bool(OPENROUTER_KEY.strip())

# ─── Constants ────────────────────────────────────────────────────────────────
# Discord's bot-file size limit (8 MB); we enforce it before base64 encoding.
MAX_IMAGE_BYTES = 8 * 1024 * 1024

# Vision-capable models known to support the image_url content type in
# OpenRouter chat completions.  This list is intentionally conservative and
# covers the models most likely to be used as the primary brain.  OpenRouter
# also has a /models endpoint, but the lookup is network-dependent, so we
# fall back to this list rather than making a blocking call at startup.
_VISION_CAPABLE_MODELS: frozenset[str] = frozenset({
    # OpenAI
    "openai/chatgpt-4o",
    "openai/chatgpt-4o-latest",
    "openai/chatgpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    # Anthropic
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-sonnet-latest",
    "anthropic/claude-3.5-sonnet-20241022",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3-haiku",
    # Google
    "google/gemini-2.0-flash",
    "google/gemini-2.0-flash-exp",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-latest",
    "google/gemini-2.5-pro",
    "google/gemini-3.0-flash",
    "google/gemini-3.0-flash-lite",
    "google/gemini-3.5-flash",
    "google/gemini-3.5-flash-latest",
    "google/gemini-3.5-pro",
    "google/gemini-3.5-pro-latest",
    "google/gemini-3.6-flash",
    "google/gemini-3.6-flash-latest",
    "google/gemini-3.6-pro",
    # Mistral
    "mistralai/pixtral-12b",
    # Qwen
    "qwen/qwen2-vl-72b",
    "qwen/qwen2.5-vl-72b-instruct",
    "qwen/qwen2.5-coder-32b-instruct",
    "qwen/qwen-vl-plus",
    # Meta
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout",
    "meta-llama/llama-3.2-11b-vision",
    "meta-llama/llama-3.2-11b-vision-uncensored",
    "meta-llama/llama-3.2-90b-vision",
    "meta-llama/llama-3.2-90b-vision-uncensored",
    "llama-3.2-11b-vision-uncensored",
    "llama-3.2-11b-vision",
    "llama-3.2-90b-vision",
    # Deepseek
    "deepseek/deepseekvl2",
    # InternVL
    "deepseek/deepseek-coder-v2",
    "openrouter/auto",
})

# Well-known text-only models — if the primary model matches these, vision is off.
# Note: Gemma 4 is the BRAIN model (no vision), Llama Vision is the EYES model.
_TEXT_ONLY_MODELS: frozenset[str] = frozenset({
    "gemma4",
    "gemma:4b",
    "gemma3:4b",
    "gemma-4-26b",
    "gemma-4-31b",
    "llama4",
    "llama-4",
    "llama3.3-70b",
    "llama-3.3-70b-instruct",
    "qwen2.5-coder:7b",
    "qwen2.5-coder-7b",
    "qwen2.5-7b-instruct",
    "qwen3:8b",
    "qwen3-8b",
    "codellama",
    "codellama:13b",
    "mistral-nemo",
    "mistral-large",
    "deepseek-chat",
    "deepseek-coder",
})

# ─── Vision-capable cache ────────────────────────────────────────────────────
_vision_capable_cache: dict[str, bool | None] = {
    "ok": None,
    "ts": 0.0,
}
_vision_cache_lock = threading.Lock()
_VISION_CACHE_TTL = 300.0  # 5 minutes


def _get_primary_model() -> str:
    """Return the vision model name from config.

    NOTE: The brain model (Gemma 4) does NOT support vision input.
    This module sends images to the dedicated VISION_MODEL (Llama Vision).
    """
    try:
        from config import VISION_MODEL
        return VISION_MODEL or "llama-3.2-11b-vision-uncensored"
    except ImportError:
        try:
            # Fallback: try to get from models.json
            import json as _json
            import os as _os
            from pathlib import Path
            _base = Path(__file__).resolve().parent.parent
            try:
                _cfg = _json.loads((_base / "config" / "models.json").read_text(encoding="utf-8"))
                return _cfg.get("vision", "llama-3.2-11b-vision-uncensored")
            except Exception:
                return "llama-3.2-11b-vision-uncensored"
        except Exception:
            return "llama-3.2-11b-vision-uncensored"


def _is_vision_capable_sync() -> bool:
    """
    Uncached capability check.

    Checks the primary model against known text-only and vision-capable
    lists first (no network required).  If the model is not recognised,
    attempts a lightweight OpenRouter /models lookup and caches the result.
    """
    if not _OPENROUTER_CONFIGURED:
        return False

    model = _get_primary_model().lower().strip()

    # Text-only denylist — these definitely do NOT support vision.
    for text_model in _TEXT_ONLY_MODELS:
        if text_model in model:
            return False

    # Vision-capable allowlist — these definitely DO support vision.
    for vision_model in _VISION_CAPABLE_MODELS:
        if model.endswith(vision_model) or vision_model in model:
            return True

    # Unknown model — try the OpenRouter models endpoint.
    # This is the only network-dependent check; we do it only once per TTL.
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models_list = data.get("data", []) if isinstance(data, dict) else []
        vision_ids: set[str] = set()
        for m in models_list:
            if not isinstance(m, dict):
                continue
            mid = (m.get("id") or "").lower()
            arch = m.get("architecture") or m.get("architectures") or {}
            if isinstance(arch, dict) and arch.get("modality") in (
                "text+image->text", "text+image->text+code",
            ):
                vision_ids.add(mid)
            elif any(token in mid for token in ("vision", "vl", "pixtral", "image")):
                vision_ids.add(mid)
        for vid in vision_ids:
            _VISION_CAPABLE_MODELS.add(vid)
        return any(vid in model for vid in vision_ids)
    except Exception:
        # Network error or malformed response — be conservative.
        return False


def is_vision_capable() -> bool:
    """
    Return True if the primary brain model supports image input.

    Result is cached for 5 minutes to avoid repeated network lookups.
    Safe to call from hot paths.
    """
    if not _OPENROUTER_CONFIGURED:
        return False

    with _vision_cache_lock:
        cached = _vision_capable_cache
        if cached["ok"] is not None and (time.time() - cached["ts"]) < _VISION_CACHE_TTL:
            return bool(cached["ok"])

    try:
        result = _is_vision_capable_sync()
    except Exception:
        result = False

    with _vision_cache_lock:
        _vision_capable_cache["ok"] = result
        _vision_capable_cache["ts"] = time.time()

    return result


# ─── Size guard ──────────────────────────────────────────────────────────────

def validate_image_size(image_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> tuple[bool, str]:
    """
    Validate image size.

    Returns (True, "") if the image is within the limit.
    Returns (False, reason) with a human-readable explanation if too large.

    Pure function — no I/O, no network, fully deterministic.
    """
    size = len(image_bytes)
    if size <= max_bytes:
        return True, ""
    mb = max_bytes / (1024 * 1024)
    actual_mb = size / (1024 * 1024)
    if mb < 1:
        return False, (
            f"Image is {size} bytes — exceeds the {max_bytes}-byte limit. "
            "Please use a smaller image."
        )
    return False, (
        f"Image is {actual_mb:.1f} MB — exceeds the {mb:.0f} MB limit. "
        "Please use a smaller image."
    )


# ─── OpenRouter vision transport ──────────────────────────────────────────────

def _openrouter_vision(
    messages: list,
    model: str = "llama-3.2-11b-vision-uncensored",
    timeout: float = 60.0,
) -> str:
    """
    Call OpenRouter /chat/completions with a vision-capable message payload.

    Reuses the same auth/transport pattern as main_brain.py without
    duplicating API-key handling.

    Args:
        messages:  OpenAI-format message list, at least one of which
                   contains an ``image_url`` content part.
        model:     Model ID to request.  Defaults to openrouter/free.
        timeout:   Request timeout in seconds (default 60; images are slower).

    Returns:
        The assistant's text response content.

    Raises:
        RuntimeError: on any transport or API error.
    """
    if not _OPENROUTER_CONFIGURED:
        raise RuntimeError("OpenRouter is not configured (no API key).")

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
    }

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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body[:300]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenRouter request failed (network/timeout): {e.reason}") from e

    try:
        resp_json = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenRouter returned malformed JSON: {raw[:300]}") from e

    if isinstance(resp_json, dict) and "error" in resp_json:
        err_obj = resp_json["error"]
        err_msg = err_obj.get("message", str(err_obj))[:300]
        err_code = err_obj.get("code", "?")
        raise RuntimeError(f"OpenRouter JSON error {err_code}: {err_msg}")

    msg = (resp_json.get("choices") or [{}])[0].get("message", {})
    content = msg.get("content") or ""
    return str(content)


# ─── Public API ──────────────────────────────────────────────────────────────

def describe_image(
    image_bytes: bytes,
    mime: str,
    question: str = "",
) -> str:
    """
    Send an image to the primary brain and return a text description / answer.

    Uses OpenRouter's vision content format (base64 data-URL in a user message).
    Gracefully degrades when vision is unavailable or an error occurs —
    always returns a string, never raises into the chat path.

    Args:
        image_bytes:  Raw image bytes (PNG, JPEG, WebP, GIF, etc.).
        mime:         MIME type of the image, e.g. ``"image/png"``.
        question:     Optional question or instruction about the image.
                     If empty, a default question is used.

    Returns:
        - On success: the model's description/answer.
        - On size error: a user-facing error string.
        - On any other error: a descriptive error string.
    """
    # ── Size guard ──────────────────────────────────────────────────────────
    ok, reason = validate_image_size(image_bytes)
    if not ok:
        return f"[Vision] {reason}"

    # ── Capability check ────────────────────────────────────────────────────
    if not is_vision_capable():
        return (
            "[Vision] The current brain model doesn't support image input. "
            "Try switching to a vision-capable model."
        )

    # ── Build message ───────────────────────────────────────────────────────
    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")
    except Exception as exc:
        return f"[Vision] Failed to encode image: {exc}"

    data_url = f"data:{mime};base64,{b64}"

    if question:
        text_prompt = question.strip()
    else:
        text_prompt = (
            "Describe this image in detail, then answer any question the user "
            "is asking about it. Be specific and observant."
        )

    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    # ── Call ───────────────────────────────────────────────────────────────
    model = _get_primary_model()

    try:
        result = _openrouter_vision(messages, model=model, timeout=60.0)
        return result.strip() if result else "[Vision] Got an empty response from the model."
    except RuntimeError as exc:
        return f"[Vision] {exc}"
    except Exception as exc:
        return f"[Vision] Unexpected error: {exc}"
