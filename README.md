# Fairy — self-extending local voice/text assistant

Fairy is a local-first assistant built around one idea: a reasoning model
(Gemma, via Ollama) decides what to do, and a coding model (Qwen) can write
brand-new skills for itself on demand. Anything Qwen writes gets
sandbox-tested before it's ever saved or trusted.

The project started as a minimal scaffold and has grown into a full TUI +
voice assistant with Discord integration, screen/vision understanding, web
research, and an optional bridge to [Hermes Agent](https://github.com/NousResearch/hermes-agent)
for heavier delegated tasks. `install.ps1` at the repo root sets up all of
the above (Ollama, the models, Python deps, and Hermes) in one step.

## Setup

**Quick path (Windows):** run `install.ps1` — see [Installing everything](#installing-everything-windows) below.

**Manual path:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

ollama pull gemma4
ollama pull qwen2.5-coder:7b
```

Then copy the config templates and fill in your own values (both are
git-ignored, so your real keys never get committed):
```
config\api_keys.example.json  ->  config\api_keys.json
controller\.env.example       ->  controller\.env
```

Edit `config.py` / `config/models.json` if you use different model tags.

## Run

```bash
python main.py     # text-only REPL
python fairy.py     # full TUI, with voice (wake word + push-to-talk)
```

Try:
- `use the calculator to work out 12 * (4 + 3)` — exercises an existing skill
- `download all the images from a webpage` — if this skill doesn't exist yet,
  Gemma calls `create_skill`, which hands off to Qwen, sandbox-tests the
  result, and saves it into `generated_skills/` if it passes.

## How it fits together

- `main.py` / `fairy.py` — entry points (text REPL vs. full TUI + voice)
- `controller/agent_controller.py` — the "boss": routes Gemma's tool calls,
  drives the Qwen generate → sandbox test → fix loop
- `controller/skill_manager.py` — list/load/save/run skills from disk
- `controller/ollama_client.py` — thin wrapper over the `ollama` package
- `controller/intent_resolver.py` — pre-processes input with context-based
  disambiguation before routing
- `controller/voice_listener.py` — push-to-talk + wake word, feeds Whisper STT
- `controller/discord_bot.py` — optional Discord bridge (`FAIRY_USE_HERMES`,
  Discord bot token in `controller/.env`)
- `controller/web_research.py` — crawler + site resolution for research skills
- `hermes_bridge.py` — optional bridge to a local Hermes Agent instance for
  heavier delegated/agentic tasks
- `sandbox/sandbox_runner.py` — static pattern check + isolated subprocess
  smoke test for anything Qwen writes, before it's ever saved or trusted
- `skills/` — built-in skills (calculator, web search/fetch, maps, reminders,
  system monitor, browser automation, computer control, vision, and more)
- `generated_skills/` — where Qwen's self-written skills land once they pass
  sandboxing
- `vision/` — screen capture + vision-model client for "look at my screen"
  style requests
- `memory/` — long-term memory manager (facts persisted across sessions)
- `tests/` — pytest suite; run with `python -m pytest`

## Installing everything (Windows)

```powershell
.\install.ps1
```

This installs/updates Ollama, pulls `gemma4` and `qwen2.5-coder:7b`, installs
[Hermes Agent](https://github.com/NousResearch/hermes-agent) via its official
installer, and sets up Fairy's own Python virtual environment and config
templates. Review the script before running it — it makes system-level
changes (installing software, pulling multi-GB models).

## Known gaps / honest caveats

- **Sandbox is a subprocess, not a real sandbox.** The banned-token check in
  `sandbox_runner.py` is a first filter, not a security boundary — before you
  point this at anything that matters, swap it for a container
  (Docker/gVisor/firejail). See `CONTRIBUTING.md` if you want to help with this.
- **Windows-only today.** Several modules (`pypiwin32`, window/process control
  in `vision/` and `controller/`) assume Windows. Porting to Linux/macOS is
  a welcome contribution.
- **Tool-calling support depends on your Ollama version and Gemma build** —
  if `gemma4` doesn't support tool calls in your local Ollama, check
  `ollama show gemma4`, or swap `MODEL_BRAIN` for a tool-calling-capable model.
- See `docs/known-issues.md` for smaller, currently-tracked bugs.

## Contributing

See `CONTRIBUTING.md` for setup details, test instructions, and ground
rules (mainly: never commit secrets or `.venv`/`browser_data`/generated
model files — they're all already git-ignored).

## License

MIT — see `LICENSE`.
