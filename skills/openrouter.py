"""OpenRouter API wrapper for Fairy 2.0 escalation.

Free-tier models on OpenRouter change frequently — providers add and drop
":free" variants without notice. Instead of trusting a hardcoded list of
model IDs, this module asks OpenRouter's own /api/v1/models endpoint what
is *currently* free and picks the strongest available model per category.

Safety rule: Fairy never spends money automatically. Every model this
module calls is verified free (zero prompt/completion/request/image cost)
against live (or cached) OpenRouter metadata before it is used. Calling an
explicit paid model requires the caller to pass allow_paid=True.
"""
import json
import logging
import re
import time
import urllib.request
import urllib.error
from pathlib import Path as _Path

try:
    from ..config import OPENROUTER_KEY
except ImportError:
    try:
        from config import OPENROUTER_KEY
    except ImportError:
        import sys as _sys
        _project_root = str(_Path(__file__).resolve().parent.parent)
        if _project_root not in _sys.path:
            _sys.path.insert(0, _project_root)
        try:
            from config import OPENROUTER_KEY
        except ImportError:
            OPENROUTER_KEY = ""

_AVAILABLE = bool(OPENROUTER_KEY)

log = logging.getLogger("fairy.openrouter")

# Call-counting for quota auditing
_call_counter = 0  # total POST /chat/completions calls made by this module in this process


def get_call_count() -> int:
    """Return the number of OpenRouter /chat/completions calls made in this process."""
    return _call_counter

MODELS_URL = "https://openrouter.ai/api/v1/models"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

_CACHE_TTL_SECONDS = 30 * 60

_ZERO_PRICING = {"prompt": "0", "completion": "0", "request": "0", "image": "0"}
_STATIC_FALLBACK_FREE_MODELS = [
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3 70B (fallback)",
     "context_length": 65536, "pricing": _ZERO_PRICING, "architecture": {"input_modalities": ["text"]}},
    {"id": "google/gemma-3-27b-it:free", "name": "Gemma 3 27B (fallback)",
     "context_length": 96000, "pricing": _ZERO_PRICING, "architecture": {"input_modalities": ["text", "image"]}},
    {"id": "mistralai/mistral-small-3.1-24b-instruct:free", "name": "Mistral Small 3.1 (fallback)",
     "context_length": 96000, "pricing": _ZERO_PRICING, "architecture": {"input_modalities": ["text"]}},
]

_LEGACY_NICKNAME_TO_CATEGORY = {
    "qwen3-coder": "coder", "qwen3": "coder", "deepseek-v3": "reasoning",
    "deepseek-chat": "reasoning", "llama4-maverick": "general", "llama4": "general",
    "nemotron-70b": "general", "nemotron": "general", "nemotron-free": "general",
    "gemma3-27b": "fast", "gemma3": "fast", "gemma-27b": "fast",
    "gemma4-9b": "fast", "gemma4-27b": "fast", "gemma4-31b": "fast",
    "phi-4": "fast", "mistral-small": "fast", "mistral": "fast",
    "gemini-flash": "research", "gemini-2.5-flash": "research",
    "claude-sonnet": "reasoning", "claude-opus": "reasoning",
    "gpt5": "reasoning", "o3-mini": "reasoning", "gpt-oss": "general",
    "grok2": "general", "kimi-k2": "reasoning", "liquid": "fast",
    "liquid-free": "fast", "poolside-s": "coder", "poolside-xs": "coder",
    "cohere-code": "coder", "auto-free": "general",
}

_CATEGORY_ALIASES = {
    "coder": "coder", "code": "coder",
    "reasoning": "reasoning", "reason": "reasoning", "smart": "reasoning",
    "research": "long_context", "long-context": "long_context", "long_context": "long_context",
    "fast": "fast", "cheap": "fast", "quick": "fast",
    "general": "general", "chat": "general",
    "vision": "vision", "image": "vision",
}

_VALID_CATEGORIES = {"coder", "reasoning", "long_context", "fast", "general", "vision"}


def _normalize_category(name: str):
    if not name:
        return None
    key = name.strip().lower().replace("_", "-")
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    if key in _LEGACY_NICKNAME_TO_CATEGORY:
        return _LEGACY_NICKNAME_TO_CATEGORY[key]
    return None


_cache = {"ts": 0.0, "raw": None, "source": None}

# Models that recently failed with an availability-type error (403/404/429/etc.)
# get soft-demoted to the bottom of the ranking for a cooldown period, so we
# don't keep spending a request + 1s sleep on a model we already know is down
# right now. They are NOT excluded outright — if every other candidate is
# also down, we still fall through to them eventually.
#
# Persisted to a small JSON file next to this script so the cooldown also
# applies across separate CLI invocations, not just within one long-running
# process. Corrupt/missing/unwritable files are treated as an empty
# blacklist — this is a best-effort optimization, never load-bearing.
_BLACKLIST_TTL_SECONDS = 10 * 60
_BLACKLIST_FILE = _Path(__file__).resolve().parent / ".openrouter_blacklist.json"


def _load_blacklist():
    try:
        raw = json.loads(_BLACKLIST_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    now = time.time()
    return {
        model_id: expiry for model_id, expiry in raw.items()
        if isinstance(expiry, (int, float)) and expiry > now
    }


def _save_blacklist():
    try:
        tmp = _BLACKLIST_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_failure_blacklist), encoding="utf-8")
        tmp.replace(_BLACKLIST_FILE)
    except OSError as e:
        log.warning("Could not persist model blacklist to %s: %s", _BLACKLIST_FILE, e)


_failure_blacklist = _load_blacklist()  # model_id -> expiry timestamp (time.time())


def _blacklist_model(model_id: str, ttl: float = _BLACKLIST_TTL_SECONDS):
    _failure_blacklist[model_id] = time.time() + ttl
    _save_blacklist()


def _is_blacklisted(model_id: str) -> bool:
    expiry = _failure_blacklist.get(model_id)
    if expiry is None:
        return False
    if time.time() >= expiry:
        del _failure_blacklist[model_id]
        _save_blacklist()
        return False
    return True


def _blacklist_remaining(model_id: str):
    expiry = _failure_blacklist.get(model_id)
    if expiry is None:
        return None
    remaining = expiry - time.time()
    return max(0, int(remaining)) if remaining > 0 else None


def _http_get_models(timeout=15):
    req = urllib.request.Request(
        MODELS_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")[:300]
        if e.code == 401:
            raise RuntimeError("OpenRouter 401: API key missing or invalid.") from e
        if e.code == 429:
            raise RuntimeError("OpenRouter 429: rate limited while listing models.") from e
        if e.code == 404:
            raise RuntimeError(f"OpenRouter 404 on models endpoint: {body}") from e
        raise RuntimeError(f"OpenRouter HTTP {e.code} listing models: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenRouter models endpoint unreachable (network/timeout): {e.reason}") from e
    except TimeoutError as e:
        raise RuntimeError("OpenRouter models endpoint timed out.") from e

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError("OpenRouter returned malformed JSON for /models.") from e

    data = parsed.get("data") if isinstance(parsed, dict) else None
    if not isinstance(data, list):
        raise RuntimeError("OpenRouter /models response missing expected 'data' list.")
    return data


def fetch_models(force: bool = False):
    if not _AVAILABLE:
        raise RuntimeError("OpenRouter API key not configured")

    now = time.time()
    if not force and _cache["raw"] is not None and (now - _cache["ts"]) < _CACHE_TTL_SECONDS:
        return _cache["raw"]

    try:
        data = _http_get_models()
        _cache.update(ts=now, raw=data, source="live")
        return data
    except RuntimeError as e:
        log.warning("OpenRouter model discovery failed: %s", e)
        if _cache["raw"] is not None:
            log.warning("Falling back to previously cached model list (stale).")
            _cache["source"] = "stale-cache"
            return _cache["raw"]
        log.warning("No cache available either — using static fallback registry.")
        _cache.update(ts=now, raw=_STATIC_FALLBACK_FREE_MODELS, source="static-fallback")
        return _STATIC_FALLBACK_FREE_MODELS


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _is_free(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    cost_fields = ("prompt", "completion", "request", "image", "web_search", "internal_reasoning")
    if not pricing:
        return False
    for field in cost_fields:
        if field in pricing and _num(pricing[field]) != 0.0:
            return False
    if model.get("id", "").startswith("openrouter/"):
        return False
    return True


def _free_models(force: bool = False):
    raw = fetch_models(force=force)
    return [m for m in raw if _is_free(m)]


_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b")


def _extract_size_b(text: str) -> float:
    matches = _SIZE_RE.findall(text.lower())
    if not matches:
        return 7.0
    return max(float(m) for m in matches)


def _is_vision_capable(model: dict) -> bool:
    arch = model.get("architecture") or {}
    modalities = arch.get("input_modalities") or []
    if "image" in modalities:
        return True
    legacy_modality = arch.get("modality") or model.get("modality") or ""
    return "image" in legacy_modality.lower() and "->" in legacy_modality


def _is_chat_capable(model: dict) -> bool:
    arch = model.get("architecture") or {}
    outputs = arch.get("output_modalities")
    if outputs:
        return "text" in outputs
    legacy_modality = arch.get("modality") or model.get("modality") or ""
    if "->" in legacy_modality:
        return "text" in legacy_modality.split("->", 1)[1]
    return True


def _score(model: dict, category: str) -> float:
    id_ = (model.get("id") or "").lower()
    name = (model.get("name") or "").lower()
    desc = (model.get("description") or "").lower()
    text = f"{id_} {name} {desc}"
    size = _extract_size_b(text)
    ctx = _num(model.get("context_length"), 4096)

    if category == "long_context":
        return ctx + size * 10
    if category == "fast":
        score = -size
        for kw in ("flash", "mini", "instant", "fast", "turbo", "small", "nano", "xs"):
            if kw in text:
                score += 50
        return score
    if category == "coder":
        score = size
        for kw in ("code", "coder", "codex"):
            if kw in text:
                score += 100
        return score
    if category == "reasoning":
        score = size
        for kw in ("reason", "think", "r1", "qwq", "o1"):
            if kw in text:
                score += 100
        return score
    if category == "vision":
        return size
    return size


def _ranked_candidates(category: str, force: bool = False):
    models = _free_models(force=force)
    models = [m for m in models if _is_chat_capable(m)]
    if category == "vision":
        models = [m for m in models if _is_vision_capable(m)]
    # Sort by score first, then push recently-failed models to the bottom
    # (stable sort preserves score order within each group).
    scored = sorted(models, key=lambda m: _score(m, category), reverse=True)
    scored = sorted(scored, key=lambda m: _is_blacklisted(m.get("id", "")))
    return scored


# ========== FIX 2: get_best_free_model - now skips blacklisted, deduplicates ==========
def get_best_free_model(category: str, exclude=None, force: bool = False):
    cat = _CATEGORY_ALIASES.get(category.lower().replace("_", "-"), category)
    if cat not in _VALID_CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Valid: {sorted(_VALID_CATEGORIES)}")
    exclude = exclude or set()
    seen = set()
    for model in _ranked_candidates(cat, force=force):
        mid = model.get("id", "")
        if not mid or mid in exclude or mid in seen or _is_blacklisted(mid):
            continue
        seen.add(mid)
        return mid
    return None


def list_free_models(force_refresh: bool = False):
    models = _free_models(force=force_refresh)
    out = []
    for m in models:
        out.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "context_length": m.get("context_length"),
            "vision": _is_vision_capable(m),
            "chat_capable": _is_chat_capable(m),
            "blacklist_remaining": _blacklist_remaining(m.get("id", "")),
        })
    out.sort(key=lambda m: m["context_length"] or 0, reverse=True)
    return out


def refresh_free_models():
    before = {m["id"] for m in _free_models()} if _cache["raw"] is not None else set()
    models = _free_models(force=True)
    after = {m["id"] for m in models}
    return {
        "source": _cache["source"],
        "free_model_count": len(models),
        "added": sorted(after - before),
        "removed": sorted(before - after),
    }


def _is_model_id_free(model_id: str, force: bool = False) -> bool:
    ids = {m["id"] for m in _free_models(force=force)}
    return model_id in ids


def _call(model: str, messages: list, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    if not _AVAILABLE:
        raise RuntimeError("OpenRouter API key not configured")
    data = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        CHAT_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://fairy.local",
            "X-Title": "Fairy AI",
        },
        method="POST",
    )
    try:
        global _call_counter
        _call_counter += 1
        call_num = _call_counter
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body[:300]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenRouter request failed (network/timeout): {e.reason}") from e

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenRouter returned malformed JSON: {raw[:300]}") from e

    # OpenRouter sometimes returns HTTP 200 with an error object inside JSON
    if isinstance(result, dict) and "error" in result:
        err = result["error"]
        code = err.get("code", "unknown")
        msg = err.get("message", str(err))
        raise RuntimeError(f"OpenRouter JSON error {code}: {msg[:300]}")

    try:
        content = result["choices"][0]["message"]["content"]
        log.info("OpenRouter call #%d succeeded: model=%s, content_len=%d", call_num, model, len(content))
        return content
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"OpenRouter returned an unexpected response shape: {raw[:300]}") from e


def _looks_like_availability_error(err: Exception) -> bool:
    msg = str(err)
    # HTTP-level errors: 404/400/403 = model gone/gated; 429 = provider rate-limited;
    # 502/503/504 = provider temporarily down. All of these mean "try another model".
    if any(f"HTTP {code}" in msg for code in ("404", "400", "403", "429", "502", "503", "504")):
        return True
    # JSON-wrapped errors where OpenRouter returns 200 OK but embeds an error object
    if "JSON error" in msg:
        match = re.search(r"JSON error (\d+)", msg)
        if match:
            code = int(match.group(1))
            if code in (502, 503, 504, 429, 404, 400, 403):
                return True
    return "unavailable" in msg.lower()


def _call_category_with_fallback(category: str, messages: list, temperature: float, max_tokens: int, max_attempts: int = 5):
    tried = set()
    last_err = None
    for attempt in range(max_attempts):
        model_id = get_best_free_model(category, exclude=tried, force=(attempt > 0))
        if not model_id:
            break
        tried.add(model_id)
        try:
            return _call(model_id, messages, temperature, max_tokens), model_id
        except RuntimeError as e:
            last_err = e
            if _looks_like_availability_error(e):
                _blacklist_model(model_id)
                log.warning("Free model '%s' unavailable (%s) — trying next candidate for '%s'.", model_id, e, category)
                if attempt < max_attempts - 1:
                    # ========== FIX 1: faster retry ==========
                    time.sleep(0.3)
                continue
            raise
    log.warning(
        "All %d free model(s) failed for category '%s': %s. Exhausted %d OpenRouter calls.",
        len(tried), category, last_err, len(tried),
    )
    raise RuntimeError(
        f"No working free model found for category '{category}' after {len(tried)} attempt(s). "
        f"Last error: {last_err}"
    )


def ask_openrouter(prompt: str, model: str = "coder", system_prompt=None, allow_paid: bool = False) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    category = _normalize_category(model)
    if category:
        text, used_model = _call_category_with_fallback(category, messages, 0.7, 2048)
        log.info("ask_openrouter category=%s -> %s", category, used_model)
        return text

    if not allow_paid and not _is_model_id_free(model):
        raise RuntimeError(
            f"Refusing to call '{model}': not confirmed free on OpenRouter right now. "
            f"Pass allow_paid=True to override (this WILL cost money), or use a category "
            f"like 'coder'/'reasoning'/'fast'/'general'/'research'/'vision' instead."
        )
    return _call(model, messages)


def ask_or_coder(prompt: str, system: str = "") -> str:
    return ask_openrouter(prompt, model="coder", system_prompt=system)


def ask_or_smart(prompt: str, system: str = "") -> str:
    return ask_openrouter(prompt, model="reasoning", system_prompt=system)


def ask_or_cheap(prompt: str, system: str = "") -> str:
    return ask_openrouter(prompt, model="fast", system_prompt=system)


def ask_or_chat(prompt: str, system: str = "") -> str:
    return ask_openrouter(prompt, model="general", system_prompt=system)


def ask_or_research(prompt: str, system: str = "") -> str:
    return ask_openrouter(prompt, model="research", system_prompt=system)


ask_general = ask_or_chat


# ---------------------------------------------------------------------------
# Backward-compat shim: some older callers (e.g. Fairy's agent_controller.py)
# do `from skills.openrouter import MODEL_REGISTRY`, a leftover from the old
# hardcoded nickname->slug dict this module used to export. That dict is
# gone by design (it's what caused the original stale-model bug), so instead
# we expose MODEL_REGISTRY as a live snapshot: {category: best_free_model_id}
# computed on first access via module __getattr__ (PEP 562), not at import
# time. This never raises — if discovery fails for any reason, categories
# that couldn't be resolved just map to None, so a broken/offline OpenRouter
# key degrades to an empty-ish registry instead of breaking the import (and
# silently disabling the entire integration, which is what actually happened
# here).
def _build_model_registry():
    registry = {}
    for cat in sorted(_VALID_CATEGORIES):
        try:
            registry[cat] = get_best_free_model(cat)
        except Exception as e:
            log.warning("MODEL_REGISTRY: could not resolve category '%s': %s", cat, e)
            registry[cat] = None
    return registry


def __getattr__(name):
    if name == "MODEL_REGISTRY":
        return _build_model_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: openrouter.py [list|best <category>|ask <category> <prompt>|refresh]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "list":
        for m in list_free_models(force_refresh=True):
            tags = []
            if m["vision"]:
                tags.append("vision")
            if not m["chat_capable"]:
                tags.append("NOT CHAT-CAPABLE - excluded from selection")
            if m["blacklist_remaining"] is not None:
                tags.append(f"recently failed, retry in {m['blacklist_remaining']}s")
            tag = f" [{', '.join(tags)}]" if tags else ""
            print(f"{m['id']:60s} ctx={m['context_length']}{tag}")
        print(f"\nsource: {_cache['source']}  total_free: {len(list_free_models())}")
    elif cmd == "best" and len(sys.argv) >= 3:
        model_id = get_best_free_model(sys.argv[2])
        print(model_id or f"No free model currently available for '{sys.argv[2]}'")
    elif cmd == "ask" and len(sys.argv) >= 4:
        print(ask_openrouter(" ".join(sys.argv[3:]), model=sys.argv[2]))
    elif cmd == "refresh":
        print(json.dumps(refresh_free_models(), indent=2))
    elif cmd == "blacklist":
        if len(sys.argv) >= 3 and sys.argv[2] == "clear":
            _failure_blacklist.clear()
            _save_blacklist()
            print("Blacklist cleared.")
        elif not _failure_blacklist:
            print("Blacklist empty.")
        else:
            for model_id in list(_failure_blacklist):
                remaining = _blacklist_remaining(model_id)
                if remaining is not None:
                    print(f"{model_id:60s} retry in {remaining}s")
    else:
        print("Usage: openrouter.py [list|best <category>|ask <category> <prompt>|refresh|blacklist [clear]]")
        sys.exit(1)