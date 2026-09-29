# Known Issues

## Open (Non-Blocking)

---

### O2 — `prewarm_ollama_cache()` crashes on cp1252 stdout with `→` glyph

**File:** `controller/model_selector.py`  
**Severity:** Non-blocking (non-fatal, caught by `except Exception`)  
**Date opened:** 2026-09-07

`prewarm_ollama_cache()` prints a diagnostic line containing the Unicode right-arrow character `→` (U+2192). On Windows terminals using the cp1252 character encoding, this raises:

```
'charmap' codec can't encode character '\u2192' in position N
```

Same class of bug as the `✦` glyph crash in `startup_model_check()` that was fixed alongside the context-window transport fix. The same fix pattern (replace `→` with `->`) should be applied here.

**Broader note:** Sweep all `print()` calls in `model_selector.py` and `hermes_bridge.py` for non-ASCII glyphs. The `✦`, `→`, and similar symbols should be replaced with ASCII equivalents (`*`, `->`, `|`, etc.) to ensure cp1252 compatibility across all Windows terminal configurations.

---

### O3 — Ollama 0.33.1 `ollama create` rejects any model name with HTTP 400: "invalid model name"

**File:** N/A (Ollama server-side bug, no fairy/hermes code change can fix it)  
**Severity:** Non-blocking (workaround exists — see below)  
**Date opened:** 2026-09-07

Attempting to bake `num_ctx` into a Modelfile using:

```bash
ollama create gemma4 --file gemma4.Modelfile
```

fails with:

```
Error: HTTP 400: Bad Request (invalid model name for any input)
```

This blocks the intended approach of embedding `PARAMETER num_ctx 65536` into a Modelfile to pre-configure the context window without depending on the Chat Completions API `options` parameter.

**Workaround:** The context-window transport fix uses the Chat Completions API `options.num_ctx` parameter (via `agent/transports/chat_completions.py`) to set the context window at request time. This works correctly. The Modelfile approach is an optimization that would let Ollama pre-allocate the KV cache on model load, but it is blocked by this Ollama bug.

**Status:** Upstream issue in Ollama. Track at https://github.com/ollama/ollama/issues.

---

### O4 — Hardcoded Windows path in `hermes_bridge.py`'s `_resolve_fairy_context_length()`

**File:** `fairy.py` → `hermes_bridge.py`  
**Severity:** Portability note only  
**Date opened:** 2026-09-07

```python
sys.path.insert(0, r"E:\Hermes\hermes-agent")
from hermes_cli.config import load_config_readonly, cfg_get
```

This path is hardcoded as a Windows-style absolute path. If Fairy is ever run from WSL, a container, or a non-Windows host, this import will fail and `_resolve_fairy_context_length()` returns `None` silently.

**Impact:** Low. The function is only used at agent construction time to set `_config_context_length` on the compressor. When `None` is returned, Hermes falls back to its own probe, which typically resolves to the correct value (65,536 from `/api/show` on this setup). The fallback works; the hardcoded path is just a missed optimization on non-Windows hosts.

**Fix:** Derive the path from the location of `hermes_bridge.py`'s own file:

```python
hermes_root = Path(__file__).parent  # would be E:\fairy\
# Walk up to find E:\Hermes\hermes-agent\ relative to hermes_bridge.py
```

Or use an environment variable (e.g. `HERMES_ROOT`) as the canonical source of truth.

---

## Fixed

### F1 — Gemma4/Ollama context-window transport bug: "Context length exceeded (23 tokens)" on trivial messages

**Commits:** `fairy: 355ad40`, `hermes-agent: 940a05b`  
**Date fixed:** 2026-09-07  
**Verified:** End-to-end with `python fairy.py --probe-turn "wassup"` — Ollama Gemma4 responded correctly with 65,536-token context window, no errors.

**Root cause:** Three compounding issues:

1. Hermes's context compressor resolved `num_ctx` from Ollama's `/api/show` probe, which could time out on cold start, falling back to 2,048 — the exact value that makes 23 tokens exceed the limit.
2. The `ChatCompletions` transport (`agent/transports/chat_completions.py`) didn't pass `ollama_num_ctx` through `extra_body.options`, so Ollama's runtime KV cache used its hard default (16,384), insufficient for Hermes's system prompt + tool schemas.
3. Fairy's `hermes_bridge._resolve_fairy_model()` resolved to the Ollama `"gemma4"` alias even when OpenRouter was chosen as the provider, causing the wrong model to be passed to Hermes.

**Fixes applied:**

- `controller/model_selector.py`: `_probe_ollama_gemma4()` calls `/api/show` (POST, 10s timeout) and checks for `num_ctx` in `parameters` or `context_length` in `model_info`; `startup_model_check()` now runs automatically (no prompt) and rebinds the `MODEL_BRAIN` global so Hermes sees the user's actual choice.
- `hermes_bridge.py`: `_resolve_fairy_model()` reads from `model_selector.MODEL_BRAIN` after `startup_model_check()` has run; `_resolve_fairy_context_length()` reads `data/config.yaml` and applies the override to both the agent and its compressor, invalidating the compressor's `_resolved_context_length` cache.
- `controller/main_brain.py`: Bump `num_ctx` thresholds (simple 16K, medium 32K, complex 64K).
- `controller/intent_resolver.py`: Detect context-length error patterns in LLM candidate calls, fall back to web search on context errors.
- `fairy.py`: Add `--probe-context` and `--probe-turn` CLI flags for non-interactive verification.
- `agent/transports/chat_completions.py` (hermes-agent): Pass `ollama_num_ctx` via `extra_body.options.num_ctx` in the Chat Completions legacy path.
- `agent/process_bootstrap.py` (hermes-agent): `_SafeWriter.fileno()` returns `-1` on StringIO streams to avoid `io.UnsupportedOperation` errors in captured stdout.
- `hermes_state.py` (hermes-agent): WAL-reset bug warning downgraded to DEBUG when `HERMES_QUIET_WAL_BUG=1` for embedded hosts.
