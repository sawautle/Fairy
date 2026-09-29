#!/usr/bin/env python3
"""
Fairy entry point. Text REPL with VRAM-aware orchestration.

Text-only Fairy entry point. The main TUI/voice entry point is fairy.py.
"""
import argparse
import signal
import sys
import time

from controller import agent_controller
from controller.agent_controller import _log_chat_turn
from controller import tts
from controller import approval_gate  # register terminal mode (set_discord_active stays False)
from controller import terminal_input as _terminal_input
from controller.terminal_input import read_line as _read_line

_running = True


def _sigint_handler(signum, frame):
    global _running
    print("\n[Fairy] Interrupted. Cleaning up...")
    _running = False
    tts.stop_server()
    sys.exit(0)


signal.signal(signal.SIGINT, _sigint_handler)


def main():
    parser = argparse.ArgumentParser(description="Fairy AI Assistant")
    parser.add_argument("--no-tts", action="store_true", help="Disable voice output")
    args = parser.parse_args()

    print("=" * 50)
    print("🧚 Fairy is awake.")
    print("   TTS: Edge TTS output (optional)")
    print("   Type 'quit' to exit. Ctrl+C for emergency stop.")
    print("   Multi-line: Alt+Enter / Ctrl+J wraps, Enter sends")
    print("=" * 50 + "\n")

    # Eagerly initialize the prompt_toolkit session (fails silently on
    # no-TTY / missing dep — read_line() will fall back to plain input).
    _terminal_input._initialize()

    # Hermes (Fairy's primary brain) handles Ollama → OpenRouter fallback at turn time.
    # We no longer require Ollama to be running at startup — Hermes will use
    # OpenRouter's free model if Ollama is unavailable.
    try:
        import ollama
        ollama.list()
        print("[OK] Ollama reachable — Hermes will use Gemma 4.\n")
    except Exception as exc:
        print(f"[INFO] Ollama not reachable: {exc}")
        print("       Hermes will use OpenRouter free model as fallback.\n")

    # ------------------------------------------------------------------ #
    # Start TTS once and keep it alive for the whole session
    # ------------------------------------------------------------------ #
    if False:  # Legacy IndexTTS server startup; Edge TTS needs no local server.
        print("[TTS] Starting IndexTTS server (first boot may take 30–60s)...")
        if tts.start_server(wait_sec=90):
            print("[TTS] Server ready.\n")
        else:
            print("[TTS] Failed to start. Continuing without voice.\n")

    history = []

    while _running:
        try:
            user_text = _read_line().strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            break

        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit", "q"):
            break

        # ------------------------------------------------------------------ #
        # PHASE 1: LLM  (Gemma loads, TTS stays warm in remaining VRAM)
        # ------------------------------------------------------------------ #
        _log_chat_turn(user_text, "main_start")
        try:
            reply, history = agent_controller.handle_request(user_text, history)
        except Exception as exc:
            err_text = str(exc).lower()
            if any(x in err_text for x in ("cuda", "out of memory", "cudamalloc", "ggml_assert")):
                print(f"[ERROR] Out of VRAM! Gemma 4 + IndexTTS don't fit together.")
                print("[ERROR] Close other GPU apps, or run: python main.py --no-tts")
            else:
                print(f"[ERROR] LLM inference failed: {exc}\n")
            _log_chat_turn(user_text, "main_end", error=str(exc), outcome="exception")
            continue

        count, names = agent_controller._was_tool_dispatched()
        _log_chat_turn(user_text, "main_end",
                          reply_preview=(reply or "")[:300],
                          tools_dispatched=count,
                          tool_names=names,
                          outcome="success")
        print(f"fairy> {reply}\n")

        # ------------------------------------------------------------------ #
        # PHASE 2: TTS  (Gemma already unloaded by keep_alive=0)
        # ------------------------------------------------------------------ #
        if args.no_tts or not reply:
            continue

        # If server died somehow, try to revive it once
        if False:  # Legacy IndexTTS health check; retained only as historical context.
            print("[TTS] Server not responding. Trying to restart...")
            if not tts.start_server(wait_sec=60):
                print("[TTS] Restart failed. Skipping voice.\n")
                continue

        tts.speak_and_play(reply, block=False)

    print("\n[Fairy] Goodbye, Master.")
    tts.stop_server()


if __name__ == "__main__":
    main()
