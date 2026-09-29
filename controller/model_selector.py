"""
Model selector at startup: choose between Ollama Gemma4 or OpenRouter.

Auto-fallback: if user picks Ollama but it's offline, switches to OpenRouter.
"""
import json
import os
import socket
import time
from pathlib import Path

OLLAMA_GEMMA4 = "gemma4"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma4"

# Default brain model. Resolved by startup_model_check() to either Ollama
# gemma4 or an OpenRouter :free SKU. Read at Hermes agent-construction time
# (not at import time) so the value reflects the user's menu choice.
MODEL_BRAIN = OLLAMA_GEMMA4


def prewarm_ollama_cache() -> None:
    """
    Warm the Hermes context-length cache by probing Ollama's /api/show.

    Hermes's compressor resolves context_length via a cached probe that has a
    3-second timeout (now bumped to 8s). On cold start, /api/show can take
    6-10 seconds for a 16GB GGUF model. If the probe times out, Hermes falls
    back to num_ctx=2048 and trips "Context length exceeded" on trivial
    messages.

    By pre-warming the cache here (before the boot sequence / Hermes startup),
    we fill Hermes's disk cache with the real context_length so the first
    Hermes turn doesn't need to probe at all.

    The cache lives at E:\\Hermes\\data\\context_length_cache.yaml.
    We also pre-warm the in-process L1 TTL cache by warming Ollama's own
    /api/tags endpoint first (fast, <1s) so the slower /api/show has
    a better chance of succeeding inside the 8s window.
    """
    import urllib.request

    HERMES_HOME = Path(os.environ.get(
        "HERMES_HOME",
        r"E:\Hermes\data"
    ))
    cache_path = HERMES_HOME / "context_length_cache.yaml"

    base_url = OLLAMA_URL.rstrip("/")
    model = OLLAMA_MODEL

    def _write_cache(num_ctx_val: int) -> None:
        """Write a context_length entry to Hermes's disk cache."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            else:
                data = {}
        except Exception:
            data = {}
        ctx_dict = data.get("context_lengths", {})
        for key in [
            f"{model}@{base_url}",
            f"{model}@{base_url}/v1",
            f"{model}:latest@{base_url}",
            f"{model}:latest@{base_url}/v1",
        ]:
            ctx_dict[key] = num_ctx_val
        data["context_lengths"] = ctx_dict
        try:
            tmp = cache_path.with_suffix(".yaml.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                yaml.dump(data, fh)
            tmp.replace(cache_path)
            print(f"  [Model Selector] Pre-warmed Hermes context cache: {model} → {num_ctx_val:,} tokens")
        except Exception as exc:
            print(f"  [Model Selector] Cache warm failed (non-fatal): {exc}")

    try:
        import yaml
    except ImportError:
        yaml = None
        print("  [Model Selector] PyYAML not available; skipping cache pre-warm")
        return

    # Step 1: fast /api/tags probe — confirms model is loaded and warm
    try:
        req = urllib.request.Request(
            f"{base_url}/api/tags",
            data=json.dumps({"name": model}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            tags_data = json.load(resp)
        # Extract context_length from loaded model details
        ctx_from_tags = None
        for m in tags_data.get("models", []):
            if m.get("name", "").startswith(model):
                details = m.get("details", {})
                ctx_from_tags = details.get("context_length")
                break
        if ctx_from_tags:
            print(f"  [Model Selector] Ollama /api/tags: {model} context_length={ctx_from_tags:,}")
    except Exception as exc:
        print(f"  [Model Selector] /api/tags probe failed (non-fatal): {exc}")
        ctx_from_tags = None

    # Step 2: /api/show probe — gets explicit num_ctx from Modelfile params
    # This is the slow one (6-10s on cold GGUF introspect). After the
    # /api/tags warm above, Ollama has likely cached its GGUF metadata.
    try:
        req = urllib.request.Request(
            f"{base_url}/api/show",
            data=json.dumps({"name": model}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            show_data = json.load(resp)

        # Check for explicit num_ctx in Modelfile parameters
        num_ctx = None
        params = show_data.get("parameters", "")
        if "num_ctx" in params:
            for line in params.split("\n"):
                if "num_ctx" in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            num_ctx = int(parts[-1])
                            break
                        except ValueError:
                            pass

        # Fall back to GGUF model_info context_length
        if num_ctx is None:
            model_info = show_data.get("model_info", {})
            for key, value in model_info.items():
                if "context_length" in key and isinstance(value, (int, float)) and value >= 1024:
                    num_ctx = int(value)
                    break

        if num_ctx:
            _write_cache(num_ctx)
            print(f"  [Model Selector] Ollama /api/show: {model} num_ctx={num_ctx:,} (from {'Modelfile params' if num_ctx != ctx_from_tags else 'GGUF metadata'})")
        elif ctx_from_tags:
            _write_cache(ctx_from_tags)
            print(f"  [Model Selector] Ollama /api/show: num_ctx not in Modelfile; using GGUF context_length={ctx_from_tags:,}")
    except Exception as exc:
        print(f"  [Model Selector] /api/show probe failed: {exc}")
        if ctx_from_tags:
            _write_cache(ctx_from_tags)
            print(f"  [Model Selector] Falling back to /api/tags context_length={ctx_from_tags:,}")


def is_ollama_running(timeout: float = 2.0) -> bool:
    """Quick socket check: is Ollama server running?"""
    try:
        with socket.create_connection(("localhost", 11434), timeout=timeout):
            return True
    except OSError:
        return False


def is_gemma4_available(timeout: float = 5.0) -> bool:
    """Check if Gemma4 model is pulled in Ollama."""
    if not is_ollama_running(timeout=timeout):
        return False
    try:
        import ollama
        models = ollama.list()
        for m in models.get("models", []):
            name = (m.model or "").lower()
            if "gemma4" in name or ("gemma" in name and "4" in name):
                return True
        return False
    except Exception:
        return False


def _probe_ollama_gemma4(timeout: float = 10.0) -> bool:
    """Probe Ollama's /api/show for the Gemma4 model.

    Returns True if Ollama responds with a valid model descriptor (so we
    know the model is loaded and metadata is readable). Returns False if
    the probe times out, fails, or returns malformed data. Without a
    successful probe, the compressor's context_length resolution can fall
    back to 2048 and Hermes will trip the "Context length exceeded" error
    on trivial messages.

    Tries the bare model name first, then common tagged variants
    (e.g. "gemma4:latest"), and finally any name from ollama.list()
    that contains "gemma4".  Ollama's /api/show requires the full tag
    name when a model is tagged; probing with just "gemma4" silently
    fails in that case.
    """
    import urllib.request

    # Collect candidate names: bare name first, then tagged variants,
    # then any name from the live model list that matches gemma4.
    candidates = [OLLAMA_MODEL]
    # Common tag patterns Ollama users typically have.
    for tag in ("latest", ):
        candidates.append(f"{OLLAMA_MODEL}:{tag}")
    try:
        import ollama
        models = ollama.list()
        for m in models.get("models", []):
            name = (m.model or "")
            if "gemma4" in name.lower() and name not in candidates:
                candidates.append(name)
    except Exception:
        pass  # fall back to bare-name probe only

    for model_name in candidates:
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/show",
                data=json.dumps({"name": model_name}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    continue
                data = json.loads(resp.read())
            # Valid if we got either explicit num_ctx or GGUF model_info
            params = data.get("parameters", "") or ""
            if "num_ctx" in params:
                return True
            model_info = data.get("model_info", {}) or {}
            for key, val in model_info.items():
                if "context_length" in key and isinstance(val, (int, float)) and val >= 1024:
                    return True
        except Exception:
            continue
    return False


def check_openrouter() -> bool:
    """Check if OpenRouter is configured and available."""
    try:
        import urllib.request
        from config import OPENROUTER_KEY
        if not OPENROUTER_KEY:
            return False
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def startup_model_check() -> dict | None:
    """
    Auto-detect the best available brain at startup. Returns a dict with
    keys ``provider``, ``brain``, ``status`` ('ok' | 'fail'), ``detail``
    (human-readable banner text) so the boot banner can show what was
    selected and why. Returns None on a fatal error.

    Default: Ollama Gemma4 (local). Falls back silently to OpenRouter if:
      - Ollama server is not running
      - Gemma4 model is not pulled
      - /api/show probe fails (model can't be introspected for num_ctx)

    The selected brain is written to config/models.json so subsequent
    turns (and the next session) read it consistently. The boot banner
    shows which brain was chosen and why. The module-level ``MODEL_BRAIN``
    constant is also rebound so that ``_resolve_fairy_model()`` in
    hermes_bridge sees the right value at import time.
    """
    global MODEL_BRAIN
    ollama_running = is_ollama_running()
    gemma4_available = is_gemma4_available() if ollama_running else False
    ollama_probe_ok = _probe_ollama_gemma4() if gemma4_available else False
    openrouter_ok = check_openrouter()

    config_dir = Path(__file__).parent.parent / "config"
    models_json = config_dir / "models.json"

    # Load existing config
    try:
        cfg = json.loads(models_json.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}

    # Decision tree: try Gemma4 first, fall back to OpenRouter.
    if gemma4_available and ollama_probe_ok:
        provider = "ollama"
        brain = OLLAMA_GEMMA4
        # Use Hermes (in-process) for Ollama now that the num_ctx bug is fixed
        os.environ["FAIRY_USE_HERMES"] = "1"
        brain_status = "ok"
        brain_detail = "Ollama Gemma4 (local)"
        # Pre-warm Hermes's context-length cache so the first turn doesn't
        # fall back to num_ctx=2048 (root cause of the "Context length
        # exceeded (23 tokens)" error on trivial messages — Hermes's
        # /api/show probe has an 8s timeout and GGUF introspection can
        # exceed it on cold start).
        try:
            prewarm_ollama_cache()
        except Exception as exc:
            print(f"  [Model Selector] Cache pre-warm failed (non-fatal): {exc}")
    elif openrouter_ok:
        provider = "openrouter"
        brain = "google/gemma-3-27b-it:free"
        os.environ["FAIRY_USE_HERMES"] = "1"
        brain_status = "ok"
        if not ollama_running:
            brain_detail = "OpenRouter (Ollama not running)"
        elif not gemma4_available:
            brain_detail = "OpenRouter (Gemma4 not pulled)"
        else:
            brain_detail = "OpenRouter (Gemma4 /api/show probe failed)"
    else:
        # Neither is available — the boot will fail downstream anyway.
        provider = "openrouter"
        brain = "google/gemma-3-27b-it:free"
        os.environ["FAIRY_USE_HERMES"] = "1"
        brain_status = "fail"
        brain_detail = "no brain available (Ollama down, OpenRouter not configured)"

    # Save to config
    cfg["provider"] = provider
    cfg["brain"] = brain
    try:
        models_json.write_text(json.dumps(cfg, indent=4, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    # Set environment for runtime
    os.environ["FAIRY_BRAIN_PROVIDER"] = provider
    os.environ["FAIRY_BRAIN_MODEL"] = brain
    # Also set FAIRY_HERMES_MODEL so hermes_bridge._resolve_fairy_model()
    # resolves to the correct brain at construction time (not the stale
    # config.py default of google/gemma-3-27b-it:free).
    os.environ["FAIRY_HERMES_MODEL"] = brain

    # Rebind module-level MODEL_BRAIN so _resolve_fairy_model() in
    # hermes_bridge.py sees the user's menu choice at construction time
    # (not the hardcoded OLLAMA_GEMMA4 default). Without this, the agent
    # would be constructed pointing at the Ollama gemma4 alias even when
    # the user has no Ollama server, and the first turn would crash.
    MODEL_BRAIN = brain

    # Show what was selected
    print()
    if provider == "ollama":
        print(f"  * Brain: {brain_detail}")
    else:
        print(f"  * Brain: {brain_detail}")
