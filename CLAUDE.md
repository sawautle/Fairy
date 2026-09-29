# CLAUDE.md — Claude Code configuration for this repo

## Free-model-only account
**Never use background sub-agents or opus-class models.**
This is a free-tier OpenRouter account. Any request for a paid model (e.g.
`claude-opus-5`, `claude-sonnet-5`, `gpt-4o`, etc.) or a background
sub-agent that requests a paid model will fail with HTTP 402. Use only:
- `openrouter/free` (the universal free router on OpenRouter)
- `gemma4:latest` or whatever `MODEL_BRAIN` is set to in `config.py`
- Ollama local models if available

## Project context
- Entry point: `fairy.py` (TUI + voice) or `main.py` (text-only REPL)
- Controller: `controller/agent_controller.py` — handles all routing
- Intent resolver: `controller/intent_resolver.py` — pre-processes user input
  with context-based disambiguation before routing
- Voice: `controller/voice_listener.py` — PTT via Right-Ctrl + Whisper STT
- Web research: `controller/web_research.py` — crawler + site resolution
- Webhook endpoint: `main.py` exposes a Flask webhook at `/whisper-webhook`

## Running Fairy
```
python fairy.py          # TUI + voice
python main.py          # text-only REPL
python -m pytest        # full test suite
fairycode.bat           # launch Claude Code at this repo (double-click)
```

## Constraints
- Do NOT commit (user tests live with voice first)
- Do NOT touch or print `config/api_keys.json`
- Feature passes live test → commit immediately in same session
