"""
Push-to-Talk voice listener for Fairy.

Records microphone audio ONLY while Right Ctrl is physically held.
On release, the clip is sent to the local Whisper STT server.
No wake words. No VAD gating. No energy filtering.
"""

import io
import re
import wave
import queue
import threading
import ctypes
import socket

import requests
import numpy as np


WHISPER_URL = "http://localhost:9000/v1/audio/transcriptions"

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
MIN_UTTERANCE_SEC = 0.3

# Keep the terminal clean during normal use.
# Change to True only when debugging the voice pipeline.
DEBUG_VOICE = False

# Suppress all PTT UI chatter (rec indicator, release message, startup banner).
# When True the transcript line is the only output from a PTT interaction.
PTT_QUIET = True

from typing import Callable, Optional

# Optional callable — if set, used to check STT availability before transcribing.
# Expected signature: () -> bool
# Set by fairy.py to read _whisper_ready without a hard import cycle.
_STT_READY_GETTER: Optional[Callable[[], bool]] = None


def set_stt_ready_getter(fn: Optional[Callable[[], bool]]) -> None:
    """Register a callable that returns True when the STT server is reachable."""
    global _STT_READY_GETTER
    _STT_READY_GETTER = fn


def _is_stt_ready() -> bool:
    """True when the whisper STT server is reachable on port 9000."""
    if _STT_READY_GETTER is not None:
        return _STT_READY_GETTER()
    try:
        # Use 127.0.0.1 (not 'localhost') to avoid IPv6 first-timeout on
        # Windows. Same trick fairy.py uses to keep boot snappy.
        with socket.create_connection(("127.0.0.1", 9000), timeout=0.2):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Push-to-Talk: Right Ctrl (Windows, zero-deps via ctypes)
# ---------------------------------------------------------------------------

def is_ptt_active() -> bool:
    """True while Right Ctrl is physically held. Windows only."""
    try:
        # VK_RCONTROL = 0xA3
        # High bit (0x8000) means key is currently down
        return bool(ctypes.windll.user32.GetAsyncKeyState(0xA3) & 0x8000)
    except AttributeError:
        # Non-Windows fallback — PTT is effectively disabled
        return False


# ---------------------------------------------------------------------------
# TTS coordination
# ---------------------------------------------------------------------------

# When Fairy is speaking, the microphone pipeline must ignore audio so Fairy
# does not transcribe or trigger herself.
_speaking_event = threading.Event()


def set_speaking(speaking: bool):
    """Pause/resume voice processing while Fairy's TTS is playing."""
    if speaking:
        _speaking_event.set()
    else:
        _speaking_event.clear()


def is_speaking() -> bool:
    return _speaking_event.is_set()


# ---------------------------------------------------------------------------
# Whisper correction
# ---------------------------------------------------------------------------

class FairySTTCorrector:
    COMMANDS = {
        "fitgirl": [
            "fit girl",
            "fitgrill",
            "fitgril",
            "streetgirl",
            "street girl",
            "siegel",
            "sigil",
            "fit gir",
            "fit gurl",
            "food girl",
            "foot girl",
            "fiddle",
        ],
        "repacks": [
            "re packs",
            "repax",
            "re-packs",
            "repack",
        ],
        "youtube": [
            "u tube",
            "you tube",
            "yt",
        ],
        "spotify": [
            "spot ify",
            "spotfy",
        ],
        "instagram": [
            "insta gram",
            "instgram",
        ],
        "discord": [
            "this cord",
            "discored",
        ],
        "browser": [
            "brows er",
            "brawser",
        ],
        "amazon": [
            "am zon",
            "amaz on",
        ],
    }

    @classmethod
    def correct(cls, text: str) -> str:
        if not text:
            return text

        lowered = text.lower()

        for correct_word, aliases in cls.COMMANDS.items():
            for alias in aliases:
                pattern = r"\b" + re.escape(alias) + r"\b"
                lowered = re.sub(
                    pattern,
                    correct_word,
                    lowered,
                    flags=re.IGNORECASE,
                )

        return lowered


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _pcm_to_wav_bytes(pcm_frames: bytes) -> bytes:
    buf = io.BytesIO()

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_frames)

    return buf.getvalue()


def _transcribe(pcm_frames: bytes) -> str:
    if not _is_stt_ready():
        if DEBUG_VOICE:
            print("[STT] Server not ready — ears still warming up")
        return ""

    try:
        resp = requests.post(
            WHISPER_URL,
            files={
                "file": (
                    "utterance.wav",
                    _pcm_to_wav_bytes(pcm_frames),
                    "audio/wav",
                )
            },
            data={"model": "whisper-1"},
            timeout=30,
        )

        resp.raise_for_status()

        raw = resp.json().get("text", "").strip()
        corrected = FairySTTCorrector.correct(raw)

        if DEBUG_VOICE:
            print(f"[STT] '{raw}' -> '{corrected}'")

        return corrected

    except Exception as exc:
        if DEBUG_VOICE:
            print(f"[STT] Error: {exc}")

        return ""


# ---------------------------------------------------------------------------
# Main listener
# ---------------------------------------------------------------------------

class VoiceListener:
    def __init__(
        self,
        on_directed,
        on_noteworthy=None,
        classify_ambient=None,
        device=None,
    ):
        self.on_directed = on_directed
        self.on_noteworthy = on_noteworthy
        self.classify_ambient = classify_ambient
        self.device = device

        self._audio_q = queue.Queue()
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._process_thread = None

        try:
            import sounddevice as _sd
        except ImportError as exc:
            raise ImportError(
                "Voice listening requires: sounddevice. "
                "Install with: pip install sounddevice"
            ) from exc

    def start(self):
        self._stop_event.clear()

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )

        self._process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
        )

        self._capture_thread.start()
        self._process_thread.start()

        if not PTT_QUIET:
            print("[Voice] 🎧 Push-to-Talk active. Hold Right Ctrl to speak to Fairy.")

    def stop(self):
        self._stop_event.set()

    def _capture_loop(self):
        import sounddevice as sd

        def _callback(indata, frames, time_info, status):
            self._audio_q.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=_callback,
        ):
            while not self._stop_event.is_set():
                sd.sleep(100)

    def _process_loop(self):
        buffer = bytearray()
        was_ptt = False

        while not self._stop_event.is_set():
            try:
                frame = self._audio_q.get(timeout=0.05)
            except queue.Empty:
                # If PTT was held and is now released, finalize whatever we have
                if was_ptt and not is_ptt_active():
                    self._finalize(buffer)
                    buffer = bytearray()
                    was_ptt = False
                continue

            # Never let Fairy's own TTS become an STT utterance.
            if is_speaking():
                buffer.clear()
                was_ptt = False
                continue

            ptt = is_ptt_active()

            if ptt:
                if not was_ptt:
                    # Rising edge: user just pressed Right Ctrl
                    buffer.clear()
                    was_ptt = True
                    # Indicator is purely cosmetic — never let a print
                    # error break the audio pipeline.
                    if not PTT_QUIET:
                        try:
                            print("● rec")
                        except Exception:
                            pass
                buffer.extend(frame)

            elif was_ptt:
                # Falling edge: user just released Right Ctrl.
                # Include this last frame (captured while held) and finalize.
                buffer.extend(frame)
                self._finalize(buffer)
                buffer = bytearray()
                was_ptt = False
                if not PTT_QUIET:
                    print("[PTT] 🛑 Right Ctrl released — processing...")

    def _finalize(self, buffer: bytearray):
        pcm = bytes(buffer)
        min_bytes = int(SAMPLE_RATE * 2 * MIN_UTTERANCE_SEC)

        if len(pcm) < min_bytes:
            if DEBUG_VOICE:
                print("[PTT] ⚠️  Clip too short, discarded")
            return

        text = _transcribe(pcm)

        if not text:
            return

        print(f"[Voice] Transcript: '{text}'")
        self.on_directed(text)