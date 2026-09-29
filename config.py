"""Fairy 2.0 configuration."""
import json
import os
from pathlib import Path

_BASE = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(_BASE / "controller" / ".env", override=False)
    load_dotenv(_BASE / ".env", override=False)
except ImportError:
    pass

# Default OpenRouter model (used when Ollama is unavailable)
DEFAULT_OPENROUTER_MODEL = "google/gemma-3-27b-it:free"
# Default Ollama model
DEFAULT_OLLAMA_MODEL = "gemma4"

def _load_cfg() -> dict:
    """Load configuration. Model names come from models.json; keys come from api_keys.json."""
    # Primary: models.json (no keys, safe to read) — source of truth for model names
    models_cfg = {}
    try:
        models_cfg = json.loads((_BASE / "config" / "models.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    # Secondary: api_keys.json (API keys only, may contain legacy model_brain)
    keys_cfg = {}
    try:
        keys_cfg = json.loads((_BASE / "config" / "api_keys.json").read_text(encoding="utf-8"))
    except Exception:
        pass

    # Combined: models.json wins for model fields; api_keys.json wins for API keys
    combined = dict(models_cfg)
    # Copy API keys from keys_cfg
    for key in ("openrouter_api_key", "gemini_api_key"):
        if key in keys_cfg:
            combined[key] = keys_cfg[key]

    # Use runtime environment if set by model selector, otherwise fall back to config
    runtime_brain = os.environ.get("FAIRY_BRAIN_MODEL")
    if runtime_brain:
        combined["model_brain"] = runtime_brain
    else:
        combined["model_brain"] = models_cfg.get("brain", DEFAULT_OPENROUTER_MODEL)

    combined["model_coder"] = models_cfg.get("coder", "qwen2.5-coder:7b")
    combined["vision_model"] = models_cfg.get("vision", "llama-3.2-11b-vision-uncensored")
    combined["vision_provider"] = models_cfg.get("vision_provider", "openrouter")
    combined["ollama_url"] = models_cfg.get("ollama_url", "http://localhost:11434")

    # Export API keys as environment variables so downstream libraries (e.g. Hermes's
    # auxiliary_client which reads os.getenv("OPENROUTER_API_KEY")) can use them.
    # This makes Hermes's vision_analyze, image_generate, and other OpenRouter-dependent
    # tools work via the same API key that fairy's vision module uses.
    if combined.get("openrouter_api_key"):
        os.environ["OPENROUTER_API_KEY"] = combined["openrouter_api_key"]
    if combined.get("gemini_api_key"):
        os.environ["GEMINI_API_KEY"] = combined["gemini_api_key"]

    # Hermes uses "openai-api" as provider to connect to local Ollama.
    # The openai-api provider reads OPENAI_API_KEY and OPENAI_BASE_URL.
    # Set these so Hermes routes to local Ollama instead of OpenRouter.
    # NOTE: OPENAI_BASE_URL must include /v1 for Ollama's OpenAI-compatible endpoint.
    if combined.get("ollama_url"):
        base = combined["ollama_url"].rstrip("/")
        os.environ["OPENAI_BASE_URL"] = f"{base}/v1"
    # Any non-empty key works — Ollama doesn't validate it, but Hermes requires it.
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "ollama"

    return combined

CFG = _load_cfg()


def _normalize_model_name(value: str | None, default: str | None = None) -> str:
    """Normalize model names for compatibility.

    Short aliases map to full model names.
    The full model path is preserved as-is.
    """
    if value is None:
        return default or DEFAULT_OPENROUTER_MODEL
    cleaned = str(value).strip()
    if not cleaned:
        return default or DEFAULT_OPENROUTER_MODEL
    norm = cleaned.lower().replace(" ", "")

    # Ollama aliases
    ollama_aliases = {
        "gemma4": DEFAULT_OLLAMA_MODEL,
        "gemma4:latest": DEFAULT_OLLAMA_MODEL,
        "gemma-4": DEFAULT_OLLAMA_MODEL,
    }

    # OpenRouter aliases
    or_aliases = {
        "gemma4-openrouter": DEFAULT_OPENROUTER_MODEL,
        "gemma3-27b": DEFAULT_OPENROUTER_MODEL,
    }

    all_aliases = {**ollama_aliases, **or_aliases}

    if norm in all_aliases:
        return all_aliases[norm]

    # If it's already a full model name, return as-is
    if "/" in cleaned:
        return cleaned

    return cleaned


MODEL_BRAIN = _normalize_model_name(
    CFG.get("model_brain") or os.environ.get("FAIRY_MODEL_BRAIN"),
    DEFAULT_OPENROUTER_MODEL
)
MODEL_CODER        = CFG.get("model_coder", "qwen2.5-coder:7b")

# Vision model — separate from main brain. Uses Llama 3.2 Vision (uncensored)
# for image/screen understanding. Brain calls this via vision_client.
VISION_MODEL       = CFG.get("vision_model", "llama-3.2-11b-vision-uncensored")
VISION_PROVIDER    = CFG.get("vision_provider", "openrouter")  # 'openrouter' or 'ollama'
MAX_FIX_ATTEMPTS   = CFG.get("max_fix_attempts", 5)
OLLAMA_URL         = CFG.get("ollama_url", "http://localhost:11434")
OPENROUTER_KEY     = CFG.get("openrouter_api_key", "") or os.environ.get("OPENROUTER_API_KEY", "")
GEMINI_KEY         = CFG.get("gemini_api_key", "") or os.environ.get("GEMINI_API_KEY", "")
OS_SYSTEM          = CFG.get("os_system", "windows").lower()
DEBUG              = os.environ.get("FAIRY_DEBUG", "0") == "1"
# Path that Claude Code may access when delegating repository tasks
CLAUDE_CODE_PROJECT_ROOT = os.environ.get("FAIRY_CLAUDE_CODE_PROJECT_ROOT",
                                         str(_BASE))
DEBUG_FILE         = os.environ.get("FAIRY_DEBUG_FILE", str(_BASE / "fairy_debug.log"))

def log(msg: str) -> None:
    if not DEBUG:
        return
    try:
        with open(DEBUG_FILE, "a", encoding="utf-8") as f:
            from datetime import datetime
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass