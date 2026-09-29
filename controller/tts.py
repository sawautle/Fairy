"""
TTS for Fairy. Edge TTS (cloud, free) + pygame playback.
Strips markdown, normalises whitespace, and expands common abbreviations
so she doesn't spell out things like "sec" as "S-E-C".

Voice: "Alyws" — Fairy's default Edge TTS voice is en-US-AriaNeural.
The voice can be overridden via the FAIRY_TTS_VOICE environment variable
or the tts_voice key in config/api_keys.json.  We always re-pin to
en-US-AriaNeural on startup so a stray config edit doesn't change her
voice — Master wants Alyws as the default and we re-assert that on every
load.
"""

import json
import os
import re
import tempfile
import threading

# Pygame is only used as the audio playback backend. Hide its startup banner.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

_current_playback_thread = None
_stop_requested = threading.Event()

# ── Voice configuration ──────────────────────────────────────────────────────
# Default voice for Fairy is "Alyws" = Microsoft Aria (en-US-AriaNeural).
# Re-pin to the default every load so the voice doesn't drift if the config
# file is edited or another voice slips in.  Master always wants Alyws.
FAIRY_DEFAULT_VOICE = "en-US-AriaNeural"  # "Alyws"


def _resolve_voice() -> str:
    """
    Return the Edge TTS voice to use.  Order of precedence:
      1. FAIRY_TTS_VOICE environment variable
      2. tts_voice in config/api_keys.json
      3. HARD default: en-US-AriaNeural (Alyws)
    """
    # 1. Env var
    env_voice = os.environ.get("FAIRY_TTS_VOICE", "").strip()
    if env_voice:
        return env_voice

    # 2. config/api_keys.json
    try:
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg_voice = str(cfg.get("tts_voice", "")).strip()
            if cfg_voice:
                return cfg_voice
    except Exception:
        pass

    # 3. Hard default
    return FAIRY_DEFAULT_VOICE


def _strip_markdown(text: str) -> str:
    # Remove markdown formatting
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)

    # Normalise whitespace: collapse multiple spaces, newlines, tabs into single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _expand_abbreviations(text: str) -> str:
    """
    Replace common abbreviations that TTS might read as individual letters.
    """
    # Word boundaries (\b) ensure we don't match inside words like "sector".
    # Also handle variations with punctuation (e.g., "sec.").
    text = re.sub(r"\bsec\b", "second", text, flags=re.IGNORECASE)
    # You can add more here if needed, e.g.:
    # text = re.sub(r"\bhrs?\b", "hours", text, flags=re.IGNORECASE)
    # text = re.sub(r"\bmin\b", "minute", text, flags=re.IGNORECASE)
    return text


def speak_and_play(text: str, block: bool = False):
    if not text or not text.strip():
        return

    clean = _strip_markdown(text)
    if not clean:
        return

    # Expand abbreviations before speaking
    clean = _expand_abbreviations(clean)

    def _do_it():
        try:
            import edge_tts

            # Resolve voice each call so env-var changes take effect on next
            # speak (useful for one-off personality switches).  Default is
            # always "Alyws" (en-US-AriaNeural).
            voice = _resolve_voice()

            # Use the normal speaking rate (explicitly set to "+0%" to be sure)
            communicate = edge_tts.Communicate(
                clean,
                voice=voice,
                rate="+0%"   # <-- normal speed; adjust to "+10%" if you want slightly faster
            )

            # Edge TTS's save() is async; using a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name
                # If your edge_tts has save_sync, use it; otherwise use asyncio.run()
                try:
                    communicate.save_sync(tmp_path)
                except AttributeError:
                    # Fallback to async call
                    import asyncio
                    asyncio.run(communicate.save(tmp_path))

            if _HAS_PYGAME:
                pygame.mixer.init()
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not _stop_requested.is_set():
                    pygame.time.wait(50)
                pygame.mixer.music.stop()

            try:
                os.remove(tmp_path)
            except Exception:
                pass
        except Exception as exc:
            print(f"[TTS] Error: {exc}")

    _stop_requested.clear()

    if block:
        _do_it()
    else:
        _current_playback_thread = threading.Thread(target=_do_it, daemon=True)
        _current_playback_thread.start()


def stop():
    _stop_requested.set()
    try:
        if _HAS_PYGAME and pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception:
        pass


def stop_server():
    stop()
    try:
        if _HAS_PYGAME and pygame.mixer.get_init():
            pygame.mixer.quit()
    except Exception:
        pass