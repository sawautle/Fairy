# Contributing to Fairy

Thanks for taking a look at Fairy. This is a scaffold project (see
`README.md` for the architecture), so there's plenty of room to extend it.

## Getting set up

1. **Clone and enter the repo:**
   ```bash
   git clone https://github.com/<your-username>/fairy.git
   cd fairy
   ```
2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
3. **Copy the config templates and fill in your own values** (never commit
   the filled-in versions — both are already git-ignored):
   ```bash
   copy config\api_keys.example.json config\api_keys.json
   copy controller\.env.example controller\.env
   ```
4. **Install Ollama and pull the models Fairy expects:**
   ```bash
   ollama pull gemma4
   ollama pull qwen2.5-coder:7b
   ```
   Or run `install.ps1` at the repo root, which does this step plus the
   rest of the one-time setup for you.
5. **Run it:**
   ```bash
   python main.py        # text-only REPL
   python fairy.py       # TUI + voice
   ```

## Running tests

```bash
python -m pytest
```

Root-level `test_*.py` files and everything under `scripts/` are excluded
from collection on purpose (see `pytest.ini`) — they're standalone
diagnostic scripts that hit live services (Ollama, OpenRouter) rather than
unit tests.

## Where things live

See the "How it fits together" section of `README.md` for the module map.
A few pointers if you're adding something new:

- **New built-in skill** → add it under `skills/`, following the shape of
  `skills/calculator.py`.
- **Changing how skills are generated at runtime** → that's
  `controller/agent_controller.py` (the Gemma → Qwen → sandbox loop) and
  `controller/skill_manager.py` (load/save/list).
- **Sandbox hardening** → `sandbox/sandbox_runner.py`. The README is explicit
  that the current sandbox is a subprocess with a banned-token check, not a
  real security boundary — contributions that swap in a container-based
  sandbox (Docker/gVisor/firejail) are especially welcome.
- **Hermes integration** → `hermes_bridge.py`, gated by the `FAIRY_USE_HERMES`
  environment variable.

## Ground rules

- **Never commit secrets.** `config/api_keys.json` and `controller/.env`
  are git-ignored for a reason — use the `.example` files as templates.
- **Never commit `.venv/`, `browser_data/`, or anything under
  `drivers/`, `openwakeword_models/`, `tts_outputs/`, `screenshots/`** —
  these are local/generated and already in `.gitignore`.
- **Keep `requirements.txt` honest.** If your change adds a new import,
  add the package here in the same PR.
- Open an issue or draft PR early for anything that touches
  `controller/agent_controller.py` or `sandbox/` — those are the
  highest-blast-radius files in the codebase.

## Reporting a security issue

If you find something like a credential leak, an unsafe `eval`/`exec` path,
or anything that lets generated code escape the sandbox, please open an
issue (or contact the maintainer privately if it's actively exploitable)
rather than a public PR with a working exploit attached.
