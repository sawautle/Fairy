#!/usr/bin/env python3
"""
vision/vision_client.py — Fairy's Eyes: OpenRouter Llama Vision API client.

Provides a clean interface for the brain (Gemma 4) to request image analysis
from the vision module (Llama 3.2 Vision).

The brain never talks directly to the vision model — it calls describe_image()
or capture_and_describe() from this module, and the vision module handles
all vision model communication independently.

API Design:
    VisionResult — dataclass with structured fields
    describe_image(image_bytes, mime, question) -> VisionResult
        Send any image bytes to Llama Vision via OpenRouter.
        Returns structured analysis.
    is_vision_available() -> bool
        Check if OpenRouter key is configured and service is reachable.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ─── Config ───────────────────────────────────────────────────────────────────
def _find_base() -> Path:
    """Locate fairy's root directory."""
    for p in __import__("sys").path:
        if p and Path(p).joinpath("config").is_dir():
            base = Path(p)
            if (base / "config" / "api_keys.json").exists():
                return base
    # Fallback: relative to this file
    return Path(__file__).resolve().parent.parent


def _load_keys() -> dict:
    """Load API keys from config/api_keys.json."""
    base = _find_base()
    try:
        return json.loads((base / "config" / "api_keys.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_openrouter_key() -> str:
    cfg = _load_keys()
    return cfg.get("openrouter_api_key", "") or os.environ.get("OPENROUTER_API_KEY", "")


def _get_vision_model() -> str:
    """Get the vision model from config."""
    cfg = _load_keys()
    # Try vision_model from models.json first, then api_keys.json
    model = cfg.get("vision_model", "llama-3.2-11b-vision-uncensored")
    if not model or model == "llama-3.2-11b-vision-uncensored":
        # Check models.json
        base = _find_base()
        try:
            models_cfg = json.loads((base / "config" / "models.json").read_text(encoding="utf-8"))
            model = models_cfg.get("vision", "llama-3.2-11b-vision-uncensored")
        except Exception:
            pass
    return model or "llama-3.2-11b-vision-uncensored"


# ─── Constants ────────────────────────────────────────────────────────────────
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB Discord limit
DEFAULT_VISION_MODEL = "llama-3.2-11b-vision-uncensored"
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"

# System prompt for vision analysis
VISION_SYSTEM_PROMPT = """You are Fairy's vision module. Your job is to describe what you see in images clearly and accurately.

Fairy is a female AI companion. She addresses the user as "Master".
You provide factual, detailed descriptions of images for Fairy to use.
Never make up details. If you're unsure, say so.

Response format: Give a clear, detailed description covering:
1. What is shown in the image
2. Any text visible
3. Key objects and their positions
4. Any notable patterns, colors, or elements
5. Overall context and purpose

Be observant and thorough but concise."""

# ─── Vision Result ────────────────────────────────────────────────────────────
@dataclass
class VisionResult:
    """
    Structured result from vision analysis.

    Fields:
        success:     True if analysis succeeded
        description: Clear description of the image
        objects:     List of detected objects
        text:        List of detected text strings
        raw:         Raw response text from model
        error:       Error message if success=False
        model:       Model used for analysis
        latency_ms:  Time taken for analysis in ms
    """
    success: bool
    description: str = ""
    objects: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    raw: str = ""
    error: str = ""
    model: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Capability check ────────────────────────────────────────────────────────
_is_configured_cache: dict[str, bool | float] = {"ok": None, "ts": 0.0}
_cache_lock = threading.Lock()
_CACHE_TTL = 300.0  # 5 minutes


def is_vision_available() -> bool:
    """Check if vision service is configured and available."""
    with _cache_lock:
        now = time.time()
        if _is_configured_cache["ok"] is not None:
            if now - _is_configured_cache["ts"] < _CACHE_TTL:
                return bool(_is_configured_cache["ok"])
        _is_configured_cache["ts"] = now

    # Actual check
    key = _get_openrouter_key()
    result = bool(key.strip())
    with _cache_lock:
        _is_configured_cache["ok"] = result
    return result


# ─── Image validation ─────────────────────────────────────────────────────────
def validate_image(image_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> tuple[bool, str]:
    """
    Validate image size for API submission.
    Returns (True, "") if OK, (False, reason) if too large.
    """
    size = len(image_bytes)
    if size <= max_bytes:
        return True, ""
    mb_actual = size / (1024 * 1024)
    mb_limit = max_bytes / (1024 * 1024)
    return False, f"Image is {mb_actual:.1f} MB — max is {mb_limit:.0f} MB"


# ─── Core API ─────────────────────────────────────────────────────────────────
def describe_image(
    image_bytes: bytes,
    mime: str,
    question: str = "",
    model: str | None = None,
    timeout: float = 60.0,
) -> VisionResult:
    """
    Analyze an image and return structured description.

    Args:
        image_bytes: Raw image data (PNG, JPEG, WebP, etc.)
        mime:        MIME type, e.g. "image/png"
        question:    Optional question about the image. If empty, uses default.
        model:       Vision model name. Defaults to configured model.
        timeout:     Request timeout in seconds.

    Returns:
        VisionResult with analysis. Always returns a result (never raises).
        Check result.success before using fields.
    """
    t0 = time.time()
    model = model or _get_vision_model()

    # ── Validate ─────────────────────────────────────────────────────────────
    ok, reason = validate_image(image_bytes)
    if not ok:
        return VisionResult(success=False, error=reason, model=model, latency_ms=0.0)

    # ── Capability check ────────────────────────────────────────────────────
    if not is_vision_available():
        return VisionResult(
            success=False,
            error="Vision not available: OpenRouter API key not configured",
            model=model,
        )

    # ── Build message ───────────────────────────────────────────────────────
    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")
    except Exception as exc:
        return VisionResult(success=False, error=f"Base64 encode failed: {exc}", model=model)

    data_url = f"data:{mime};base64,{b64}"

    if question:
        text_prompt = question.strip()
    else:
        text_prompt = (
            "Describe this image in detail. Cover: what is shown, any visible text, "
            "key objects and their positions, notable patterns or elements, and overall context."
        )

    messages = [
        {
            "role": "system",
            "content": VISION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    # ── Call OpenRouter ─────────────────────────────────────────────────────
    key = _get_openrouter_key()
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://fairy.local",
        "X-Title": "Fairy Vision",
    }

    req = urllib.request.Request(
        OPENROUTER_API,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return VisionResult(
            success=False,
            error=f"HTTP {e.code}: {body[:200]}",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
        )
    except urllib.error.URLError as e:
        return VisionResult(
            success=False,
            error=f"Network error: {e.reason}",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
        )
    except TimeoutError:
        return VisionResult(
            success=False,
            error=f"Timeout after {timeout}s",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
        )

    # ── Parse response ──────────────────────────────────────────────────────
    try:
        resp_json = json.loads(raw)
    except json.JSONDecodeError:
        return VisionResult(
            success=False,
            error=f"Malformed response: {raw[:200]}",
            model=model,
            latency_ms=(time.time() - t0) * 1000,
        )

    if isinstance(resp_json, dict) and "error" in resp_json:
        err = resp_json["error"]
        return VisionResult(
            success=False,
            error=err.get("message", str(err))[:200],
            model=model,
            latency_ms=(time.time() - t0) * 1000,
        )

    msg = (resp_json.get("choices") or [{}])[0].get("message", {})
    content = msg.get("content") or ""
    latency = (time.time() - t0) * 1000

    if not content:
        return VisionResult(
            success=False,
            error="Empty response from vision model",
            model=model,
            latency_ms=latency,
        )

    # ── Parse structured output ─────────────────────────────────────────────
    description = content.strip()
    objects = _extract_objects(content)
    text_lines = _extract_text(content)

    return VisionResult(
        success=True,
        description=description,
        objects=objects,
        text=text_lines,
        raw=content,
        model=model,
        latency_ms=latency,
    )


def _extract_objects(content: str) -> list[str]:
    """Extract object list from response if present."""
    # Look for bullet lists or "Objects:" sections
    objects = []
    lines = content.split("\n")
    in_objects = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("objects:") or stripped.lower().startswith("detected objects"):
            in_objects = True
            continue
        if in_objects and stripped.startswith(("-", "•", "*", "1.", "2.", "3.")):
            obj = stripped.lstrip("-•*123456789. ").strip()
            if obj:
                objects.append(obj)
        elif in_objects and stripped and not stripped.startswith("-"):
            # End of objects section
            break
    return objects


def _extract_text(content: str) -> list[str]:
    """Extract text strings from response if present."""
    text = []
    lines = content.split("\n")
    in_text = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("text:") or stripped.lower().startswith("visible text"):
            in_text = True
            continue
        if in_text and (stripped.startswith(("-", "•", "*")) or stripped.startswith('"')):
            txt = stripped.lstrip("-•*\"' ").strip().rstrip('",')
            if txt:
                text.append(txt)
        elif in_text and stripped and not stripped.startswith(("-", "•", "*", '"')):
            break
    return text


# ─── Synchronous convenience wrapper ─────────────────────────────────────────
def analyze_image_sync(
    image_bytes: bytes,
    mime: str,
    question: str = "",
) -> str:
    """
    Simple synchronous wrapper: returns description string.
    For use by the brain when it just needs a description.

    Returns description on success, error string on failure.
    """
    result = describe_image(image_bytes, mime, question)
    if result.success:
        return result.description
    return f"[Vision] {result.error}"


# ─── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Vision Client Self-Test")
    print("=" * 52)
    print(f"Available: {is_vision_available()}")
    print(f"Model: {_get_vision_model()}")
    print(f"OpenRouter key: {'set' if _get_openrouter_key() else 'NOT SET'}")
