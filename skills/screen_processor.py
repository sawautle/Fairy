"""
skills/screen_processor.py — Persistent vision session for Fairy.

Adopts the Mark-LII pattern: one Gemini Live session stays connected across
multiple analyze() calls, so follow-up questions are snappy (no reconnect
overhead). Audio response plays directly via sounddevice.

Public API
----------
start_session(player=None, timeout=20.0)
    Start (or reuse) the background vision session. Thread-safe.
    Returns True on success, False on failure.

stop_session()
    Close the vision session cleanly.

is_session_ready() -> bool
    True iff the Gemini live session is active and ready to receive frames.

analyze_now(image_bytes, mime_type, user_text, player=None) -> str
    Capture + compress + send + return text (or error string).
    Works whether session is running or not (auto-starts if needed).

screen_process(parameters, response=None, player=None, session_memory=None)
    Skill-compatible wrapper. Parameters:
        angle: "screen" (default) or "camera"
        text:  question to ask about the captured image

warmup_session(player=None)
    Pre-connect the session so the first real analyze is instant.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import mss
    import mss.tools
    _MSS = True
except ImportError:
    _MSS = False

try:
    import PIL.Image
    _PIL = True
except ImportError:
    _PIL = False

try:
    import sounddevice as sd
    _SD = True
except ImportError:
    _SD = False

from google import genai
from google.genai import types as gtypes

# Fairy's central TTS (fallback when audio playback isn't available)
from controller.tts import speak_and_play


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


_BASE        = _base_dir()
_CONFIG_PATH = _BASE / "config" / "api_keys.json"


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config_key(key: str, value) -> None:
    try:
        cfg = _load_config()
        cfg[key] = value
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
    except Exception as e:
        print(f"[Vision] ⚠️  Could not save config key '{key}': {e}")


def _get_api_key() -> str | None:
    """Return the Gemini API key, or None if not set."""
    cfg = _load_config()
    key = cfg.get("gemini_api_key", "")
    return key if key else None


def _get_os() -> str:
    return _load_config().get("os_system", "windows").lower()


# ── Image compression ────────────────────────────────────────────────────────
_IMG_MAX_W = 1280
_IMG_MAX_H = 720
_JPEG_Q    = 82

# ── Audio playback ───────────────────────────────────────────────────────────
_LIVE_MODEL           = "models/gemini-2.5-flash-native-audio-preview-12-2025"
_CHANNELS            = 1
_RECEIVE_SAMPLE_RATE = 24_000
_CHUNK_SIZE          = 1_024

_SYSTEM_PROMPT = (
    "You are Fairy, Master's personal AI assistant. "
    "You are sarcastic, playful, clever, and slightly mischievous. "
    "You are given an image from either the user's screen or their webcam. "
    "Analyze what you see with detail and intelligence. "
    "Describe objects, text, people, components, and their context clearly. "
    "For technical questions (circuits, code, hardware) give specific, expert answers. "
    "Be concise — 2-4 sentences — unless the question demands more detail. "
    "Speak directly to the user ('I can see...', 'You have...'). "
    "Address the user as Master."
)


def _compress(img_bytes: bytes, source_format: str = "PNG") -> tuple[bytes, str]:
    if not _PIL:
        return img_bytes, f"image/{source_format.lower()}"

    try:
        img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((_IMG_MAX_W, _IMG_MAX_H), PIL.Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_Q, optimize=False)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        print(f"[Vision] ⚠️  Image compress failed: {e}")
        return img_bytes, f"image/{source_format.lower()}"


# ── Screen capture ───────────────────────────────────────────────────────────

def _capture_screen() -> tuple[bytes, str]:
    if not _MSS:
        raise RuntimeError("mss is not installed. Run: pip install mss")

    with mss.mss() as sct:
        monitors = sct.monitors
        target   = monitors[1] if len(monitors) > 1 else monitors[0]
        shot     = sct.grab(target)
        png      = mss.tools.to_png(shot.rgb, shot.size)

    return _compress(png, "PNG")


# ── Camera capture ───────────────────────────────────────────────────────────

def _cv2_backend() -> int:
    if not _CV2:
        return 0
    os_name = _get_os()
    if os_name == "windows":
        return cv2.CAP_DSHOW
    if os_name == "mac":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def _probe_camera(index: int, backend: int, warmup: int = 5) -> bool:
    if not _CV2:
        return False
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return False
    for _ in range(warmup):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return False
    return bool(np.mean(frame) > 8)


def _detect_camera_index() -> int:
    backend = _cv2_backend()
    print("[Vision] 🔍 Auto-detecting camera...")
    for idx in range(6):
        if _probe_camera(idx, backend):
            print(f"[Vision] ✅ Camera found at index {idx}")
            _save_config_key("camera_index", idx)
            return idx
        print(f"[Vision] ⚠️  Camera index {idx}: no usable frame")
    print("[Vision] ⚠️  No camera found — defaulting to index 0")
    _save_config_key("camera_index", 0)
    return 0


def _get_camera_index() -> int:
    cfg = _load_config()
    if "camera_index" in cfg:
        return int(cfg["camera_index"])
    return _detect_camera_index()


def _capture_camera() -> tuple[bytes, str]:
    if not _CV2:
        raise RuntimeError("OpenCV (cv2) is not installed. Run: pip install opencv-python")

    index   = _get_camera_index()
    backend = _cv2_backend()
    cap     = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        raise RuntimeError(f"Camera index {index} could not be opened.")

    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Camera returned no frame.")

    if _PIL:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb)
        img.thumbnail((_IMG_MAX_W, _IMG_MAX_H), PIL.Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_Q)
        return buf.getvalue(), "image/jpeg"

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])
    return buf.tobytes(), "image/jpeg"


# ── Persistent vision session ────────────────────────────────────────────────

class _VisionSession:
    """
    Persistent Gemini Live session that stays connected across multiple analyze()
    calls. One background thread runs an asyncio event loop; analyze() pushes
    to a queue and the loop processes it, sending image+text to Gemini and
    playing back audio responses directly via sounddevice.

    Thread-safe: start/stop/analyze can be called from any thread.
    """

    def __init__(self) -> None:
        self._loop:        Optional[asyncio.AbstractEventLoop] = None
        self._thread:      Optional[threading.Thread]          = None
        self._session:     Optional[object]                   = None   # Gemini live session
        self._out_queue:   Optional[asyncio.Queue]             = None
        self._audio_in:    Optional[asyncio.Queue]            = None
        self._ready_evt:   threading.Event                    = threading.Event()
        self._player:      Optional[object]                    = None
        self._lock:        threading.Lock                      = threading.Lock()
        self._start_called: bool                              = False
        self._transcript_buf: list[str]                       = []     # accumulates text for last reply

    # ── Public API ────────────────────────────────────────────────────────

    def start(self, player=None, timeout: float = 20.0) -> bool:
        """
        Start (or reuse) the background session thread.
        Returns True if session is ready within timeout, False otherwise.
        """
        with self._lock:
            if self._thread and self._thread.is_alive():
                if player is not None:
                    self._player = player
                return self._session is not None

            self._player = player
            self._thread  = threading.Thread(
                target=self._run_event_loop,
                daemon=True,
                name="FairyVisionThread",
            )
            self._thread.start()
            self._start_called = True

        if not self._ready_evt.wait(timeout=timeout):
            print(f"[Vision] ⚠️  Session did not connect within {timeout}s")
            return False

        print("[Vision] ✅ Session ready")
        return True

    def stop(self) -> None:
        """Close the session gracefully from any thread."""
        with self._lock:
            if self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self._session_cancel(),
                    self._loop,
                )
            self._ready_evt.clear()
            self._session  = None
            self._out_queue = None
            self._audio_in  = None

    def is_ready(self) -> bool:
        """True iff the Gemini live session is connected."""
        return self._session is not None and self._ready_evt.is_set()

    def analyze(self, image_bytes: bytes, mime_type: str, user_text: str) -> None:
        """
        Queue an image + question for analysis. Thread-safe.
        Works whether session is running or not (auto-starts if needed).
        """
        # Auto-start if not running
        if not self._start_called or not self._ready_evt.is_set():
            if not self.start(player=self._player):
                print("[Vision] ⚠️  Could not start session — dropping analyze request")
                return

        if not self._out_queue:
            print("[Vision] ⚠️  No output queue — dropping analyze request")
            return

        asyncio.run_coroutine_threadsafe(
            self._out_queue.put((image_bytes, mime_type, user_text)),
            self._loop,
        )

    # ── Internal asyncio methods ───────────────────────────────────────────

    async def _session_cancel(self) -> None:
        """Graceful session cancellation signal."""
        try:
            if self._session is not None:
                await self._session.close()
        except Exception:
            pass

    def _run_event_loop(self) -> None:
        """Background thread: create event loop and run the session."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._session_loop())
        except Exception as e:
            print(f"[Vision] ⚠️  Event loop exited: {e}")
        finally:
            self._loop.close()

    async def _session_loop(self) -> None:
        """Main session loop with auto-reconnect."""
        self._out_queue = asyncio.Queue(maxsize=30)
        self._audio_in  = asyncio.Queue()
        api_key = _get_api_key()

        if not api_key:
            print("[Vision] ⚠️  No Gemini API key found in config/api_keys.json")
            return

        client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta"},
        )

        config = gtypes.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            system_instruction=_SYSTEM_PROMPT,
            speech_config=gtypes.SpeechConfig(
                voice_config=gtypes.VoiceConfig(
                    prebuilt_voice_config=gtypes.PrebuiltVoiceConfig(
                        voice_name="Alyws"  # Fairy's voice personality
                    )
                )
            ),
        )

        backoff = 2.0
        while True:
            try:
                print("[Vision] 🔌 Connecting...")
                async with client.aio.live.connect(
                    model=_LIVE_MODEL, config=config
                ) as session:
                    self._session   = session
                    self._ready_evt.set()
                    backoff        = 2.0
                    print("[Vision] ✅ Connected")

                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(self._send_loop())
                        tg.create_task(self._recv_loop())
                        tg.create_task(self._play_loop())

            except* Exception as eg:
                for exc in eg.exceptions:
                    print(f"[Vision] ⚠️  Session error: {exc}")
            finally:
                self._session = None
                self._ready_evt.clear()

            print(f"[Vision] 🔄 Reconnecting in {backoff:.0f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)
            # Keep _ready_evt cleared so callers know we're reconnecting

    async def _send_loop(self) -> None:
        """Pull (image, mime, text) from queue and send to Gemini."""
        while True:
            image_bytes, mime_type, user_text = await self._out_queue.get()
            if not self._session:
                print("[Vision] ⚠️  No session — dropping image")
                continue

            self._transcript_buf = []  # reset for new response

            try:
                b64 = base64.b64encode(image_bytes).decode("ascii")
                await self._session.send_client_content(
                    turns={
                        "parts": [
                            {"inline_data": {"mime_type": mime_type, "data": b64}},
                            {"text": user_text},
                        ]
                    },
                    turn_complete=True,
                )
                print(f"[Vision] 📤 Sent {len(image_bytes):,} bytes — '{user_text[:60]}'")
            except Exception as e:
                print(f"[Vision] ⚠️  Send error: {e}")
                raise  # propagate → TaskGroup cancels siblings → session reconnect

    async def _recv_loop(self) -> None:
        """Receive audio + text from Gemini."""
        try:
            async for response in self._session.receive():
                # Audio chunks → play queue
                if response.data:
                    await self._audio_in.put(response.data)

                sc = response.server_content
                if not sc:
                    continue

                # Text transcript
                if sc.output_transcription and sc.output_transcription.text:
                    chunk = sc.output_transcription.text.strip()
                    if chunk:
                        self._transcript_buf.append(chunk)

                # Turn complete → speak final transcript + close camera
                if sc.turn_complete:
                    if self._transcript_buf:
                        full = re.sub(r"\s+", " ", " ".join(self._transcript_buf)).strip()
                        if full:
                            print(f"[Vision] 💬 {full}")
                            if self._player and hasattr(self._player, "write_log"):
                                self._player.write_log(f"Fairy: {full}")
                            # Play via TTS as primary fallback
                            speak_and_play(full, block=False)

                    # Auto-close camera ~2s after reply
                    if self._player and hasattr(self._player, "stop_camera_stream"):
                        async def _deferred_close():
                            await asyncio.sleep(2.0)
                            try:
                                self._player.stop_camera_stream()
                            except Exception:
                                pass
                        asyncio.create_task(_deferred_close())

                    self._transcript_buf = []

        except Exception as e:
            print(f"[Vision] ⚠️  Recv error: {e}")
            raise

    async def _play_loop(self) -> None:
        """Play audio chunks from Gemini via sounddevice."""
        if not _SD:
            return  # graceful degradation

        try:
            stream = sd.RawOutputStream(
                samplerate=_RECEIVE_SAMPLE_RATE,
                channels=_CHANNELS,
                dtype="int16",
                blocksize=_CHUNK_SIZE,
            )
            stream.start()
            try:
                while True:
                    chunk = await self._audio_in.get()
                    await asyncio.to_thread(stream.write, chunk)
            except Exception as e:
                print(f"[Vision] ⚠️  Play error: {e}")
            finally:
                stream.stop()
                stream.close()
        except Exception as e:
            print(f"[Vision] ⚠️  Play loop init failed: {e}")


# ── Module-level singleton ──────────────────────────────────────────────────

_session      = _VisionSession()
_session_lock = threading.Lock()


def start_session(player=None, timeout: float = 20.0) -> bool:
    """Start (or reuse) the persistent vision session."""
    return _session.start(player=player, timeout=timeout)


def stop_session() -> None:
    """Close the vision session."""
    _session.stop()


def is_session_ready() -> bool:
    """True iff the session is active."""
    return _session.is_ready()


def analyze_now(
    image_bytes: bytes,
    mime_type: str,
    user_text: str,
    player=None,
) -> str:
    """
    Synchronous capture + send + wait for first transcript chunk.
    Falls back to TTS if session not running (auto-starts).

    Returns the text reply from Gemini, or an error string.
    """
    # Auto-start if needed
    if not _session.is_ready():
        if not _session.start(player=player):
            return "[Vision] Could not start vision session."

    _session.analyze(image_bytes, mime_type, user_text)
    return "Fairy is analyzing. Listen for the audio response."


# ── Skill-compatible wrapper ───────────────────────────────────────────────

def screen_process(
    parameters:     dict,
    response=None,
    player=None,
    session_memory=None,
) -> bool:
    """
    Skill-compatible entry point.

    Parameters (via parameters dict):
        angle: "screen" (default) or "camera"
        text:  question to ask about the captured image

    Returns True on success, False on failure.
    """
    params    = parameters or {}
    user_text = (params.get("text") or params.get("user_text") or "").strip()
    angle     = params.get("angle", "screen").lower().strip()

    if not user_text:
        print("[Vision] ⚠️  No question provided — aborting")
        return False

    print(f"[Vision] ▶ angle={angle!r}  question='{user_text[:80]}'")

    # Capture
    try:
        if angle == "camera":
            image_bytes, mime_type = _capture_camera()
            print(f"[Vision] 📷 Camera: {len(image_bytes):,} bytes")
            if player and hasattr(player, "start_camera_stream"):
                try:
                    player.start_camera_stream()
                except Exception as _e:
                    print(f"[Vision] ⚠️  Camera stream failed: {_e}")
        else:
            image_bytes, mime_type = _capture_screen()
            print(f"[Vision] 🖥️  Screen: {len(image_bytes):,} bytes")
    except Exception as e:
        print(f"[Vision] ❌ Capture error: {e}")
        return False

    # Start session if not already running
    if not _session.is_ready():
        if not _session.start(player=player):
            fallback = "I had trouble starting the vision session, Master."
            speak_and_play(fallback)
            return False

    _session.analyze(image_bytes, mime_type, user_text)
    return True


def warmup_session(player=None) -> None:
    """Pre-connect the session so the first analyze is instant."""
    try:
        if not _session.start(player=player):
            print("[Vision] ⚠️  Warmup failed to connect")
        else:
            print("[Vision] ✅ Warmup complete")
    except Exception as e:
        print(f"[Vision] ⚠️  Warmup error: {e}")


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[TEST] screen_processor.py")
    print("=" * 52)
    mode = input("angle — screen / camera (default: screen): ").strip().lower() or "screen"
    q    = input("Question (Enter = default): ").strip() or "What do you see? Be brief."

    t0 = time.perf_counter()
    warmup_session()
    print(f"Session ready in {time.perf_counter()-t0:.2f}s\n")

    t1 = time.perf_counter()
    ok = screen_process({"angle": mode, "text": q})
    print(f"Queued in {time.perf_counter()-t1:.3f}s — waiting for audio...")
    time.sleep(10)
    print("Done." if ok else "Failed.")
