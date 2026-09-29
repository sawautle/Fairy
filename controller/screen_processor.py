#!/usr/bin/env python3
"""
Fairy Screen Processor — Real-time screen watching with voice feedback.

Features:
- Captures screen with mss
- Sends to OpenRouter vision-capable model
- Speaks back via Edge TTS (Alyws voice)
- Fairy personality (Master, playful, helpful)
- Subject change detection
- Background monitoring loop
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# Optional dependencies
try:
    import mss
    _MSS = True
except ImportError:
    _MSS = False

try:
    import PIL.Image
    _PIL = True
except ImportError:
    _PIL = False

# ─── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _SCRIPT_DIR / "config" / "api_keys.json"

# ─── Config loading ─────────────────────────────────────────────────────────────
def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

# ─── API setup ─────────────────────────────────────────────────────────────────
_OPENROUTER_KEY = ""
_MODEL_BRAIN = "gemma4"

def _init_api():
    global _OPENROUTER_KEY, _MODEL_BRAIN
    cfg = _load_config()
    _OPENROUTER_KEY = cfg.get("openrouter_api_key", "") or os.environ.get("OPENROUTER_API_KEY", "")
    _MODEL_BRAIN = cfg.get("model_brain", "gemma4")

_init_api()

# ─── Image constants ────────────────────────────────────────────────────────────
_IMG_MAX_W = 1280
_IMG_MAX_H = 720
_JPEG_Q = 82

# ─── Fairy's system prompt for screen watching ──────────────────────────────────
_FAIRY_SCREEN_PROMPT = """You are Fairy, a female AI companion. The user is your Master.

Rules:
- Address Master warmly and naturally
- Be observant about what you see on screen
- Be helpful - if Master is doing something confusing, offer guidance
- Be playful and witty, like a clever friend
- Keep responses SHORT (1-3 sentences) for voice feedback
- If Master seems stuck, offer specific help
- Don't be overly formal or robotic
- If you see something that might interest Master, mention it
- Call Master "Master" not "sir" or other formal titles

Current context: Master is watching their screen. Describe what you see and offer helpful insights.
"""

# ─── TTS import (reuses Fairy's Edge TTS) ──────────────────────────────────────
_TTS_AVAILABLE = False
def _init_tts():
    global _TTS_AVAILABLE
    try:
        # Import Fairy's TTS
        sys.path.insert(0, str(_SCRIPT_DIR))
        from controller.tts import speak_and_play
        _speak = speak_and_play
        _TTS_AVAILABLE = True
        return speak_and_play
    except Exception as e:
        print(f"[Screen] TTS not available: {e}")
        return None

_speak_fn = _init_tts()

# ─── Image compression ─────────────────────────────────────────────────────────
def _compress_image(img_bytes: bytes, source_format: str = "PNG") -> tuple[bytes, str]:
    if not _PIL:
        return img_bytes, f"image/{source_format.lower()}"

    try:
        img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((_IMG_MAX_W, _IMG_MAX_H), PIL.Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_Q, optimize=False)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        print(f"[Screen] Image compress failed: {e}")
        return img_bytes, f"image/{source_format.lower()}"

# ─── Screen capture ────────────────────────────────────────────────────────────
def _capture_screen() -> tuple[bytes, str]:
    if not _MSS:
        raise RuntimeError("mss is not installed. Run: pip install mss")

    with mss.mss() as sct:
        monitors = sct.monitors
        # [0] = all combined, [1..n] = real screens
        target = monitors[1] if len(monitors) > 1 else monitors[0]
        shot = sct.grab(target)
        png = mss.tools.to_png(shot.rgb, shot.size)

    return _compress_image(png, "PNG")

# ─── Vision API call ──────────────────────────────────────────────────────────
def _call_vision(image_bytes: bytes, mime_type: str, user_text: str = "") -> str:
    """Send image to OpenRouter vision model and get response."""
    if not _OPENROUTER_KEY:
        return "OpenRouter API key not configured."

    b64 = base64.b64encode(image_bytes).decode("ascii")

    messages = [
        {
            "role": "system",
            "content": _FAIRY_SCREEN_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text or "What do you see on screen? Keep it brief and helpful."},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
            ]
        }
    ]

    # Find a vision-capable model
    model = _find_vision_model()
    if not model:
        return "No vision-capable model available."

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {_OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://fairy.local",
        "X-Title": "Fairy Screen Processor",
    }

    url = "https://openrouter.ai/api/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return f"API error: {e.code} - {body[:200]}"
    except urllib.error.URLError as e:
        return f"Network error: {e.reason}"

    try:
        resp_json = json.loads(raw)
    except json.JSONDecodeError:
        return "Invalid response from API."

    if isinstance(resp_json, dict) and "error" in resp_json:
        err = resp_json["error"]
        return f"API error: {err.get('message', str(err))}"

    msg = (resp_json.get("choices") or [{}])[0].get("message", {})
    content = msg.get("content") or ""
    return str(content).strip()

# ─── Vision model detection ───────────────────────────────────────────────────
_VISION_MODELS = [
    "openrouter/free",  # free tier
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.0-flash",
    "qwen/qwen2-vl-72b",
]

def _find_vision_model() -> str:
    """Return first available vision-capable model."""
    # Try free first
    if _OPENROUTER_KEY:
        return "openrouter/free"
    return ""

# ─── Screen Processor Class ────────────────────────────────────────────────────
class ScreenProcessor:
    """
    Fairy Screen Processor - watches your screen and provides voice feedback.

    Usage:
        processor = ScreenProcessor()
        processor.start(interval=10)  # Check every 10 seconds

        # Or one-time capture:
        result = processor.analyze_screen("What am I looking at?")
        print(result)
    """

    def __init__(self, interval: int = 10, auto_speak: bool = True):
        """
        Args:
            interval: Seconds between automatic screen checks (default 10)
            auto_speak: Whether to speak responses aloud (default True)
        """
        self.interval = max(5, interval)  # Min 5 seconds
        self.auto_speak = auto_speak

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._pause = threading.Event()
        self._pause.set()  # Start paused

        self.last_subject = ""
        self.last_response = ""
        self.capture_count = 0

        # Subject keywords for change detection
        self.subject_keywords = {
            "coding": ["code", "python", "javascript", "function", "class", "import", "def ", "var ", "const ", "=>"],
            "browsing": ["browser", "chrome", "firefox", "tab", "url", "website", "google", "search"],
            "studying": ["pdf", "book", "chapter", "lecture", "note", "study", "exam", "question"],
            "gaming": ["game", "play", "steam", "epic", "score", "level", "mission"],
            "video": ["youtube", "video", "watch", "movie", "stream", "twitch"],
            "music": ["spotify", "music", "audio", "play", "pause", "song"],
            "document": ["word", "document", "text", "write", "essay", "report"],
            "chat": ["discord", "telegram", "message", "chat", "friend", "server"],
        }

    def start(self, background: bool = True) -> None:
        """Start the screen watching loop."""
        if self._thread and self._thread.is_alive():
            print("[Screen] Already running")
            return

        if not _MSS:
            print("[Screen] ERROR: mss not installed. Run: pip install mss")
            return

        self._running.set()
        self._pause.set()

        if background:
            self._thread = threading.Thread(
                target=self._watch_loop,
                daemon=True,
                name="ScreenProcessor"
            )
            self._thread.start()
            print(f"[Screen] Started watching every {self.interval}s")
        else:
            # Run once
            self._capture_and_analyze()

    def stop(self) -> None:
        """Stop the screen watching loop."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2)
        print("[Screen] Stopped")

    def pause(self) -> None:
        """Pause watching."""
        self._pause.clear()

    def resume(self) -> None:
        """Resume watching."""
        self._pause.set()

    def _detect_subject(self, text: str) -> str:
        """Detect what subject/type of content is on screen."""
        text_lower = text.lower()
        scores: dict[str, int] = {}

        for subject, keywords in self.subject_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[subject] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def _check_subject_change(self, new_subject: str) -> Optional[str]:
        """Return a message if subject changed, None otherwise."""
        if self.last_subject and new_subject != self.last_subject and new_subject != "general":
            old = self.last_subject
            self.last_subject = new_subject
            messages = {
                ("general", "coding"): "Switched to coding, Master. Need help with that?",
                ("general", "studying"): "Looks like study time, Master. I'll keep an eye out.",
                ("general", "browsing"): "Ah, browsing the web now? Anything interesting?",
                ("general", "gaming"): "Game time! Let me know if you need anything.",
                ("general", "video"): "Watching something? Enjoy!",
                ("coding", "studying"): "From code to studying? Versatile as always, Master.",
                ("studying", "coding"): "Back to coding! Need me to take a look?",
                ("browsing", "coding"): "Switched from browsing to code. I'm here if you need me.",
                ("coding", "browsing"): "From code to web? Let me know what you find.",
            }
            return messages.get((old, new_subject))
        return None

    def analyze_screen(self, question: str = "") -> str:
        """One-time screen analysis. Returns text response."""
        return self._capture_and_analyze(question)

    def _capture_and_analyze(self, question: str = "") -> str:
        """Capture screen and send to vision API."""
        try:
            image_bytes, mime_type = _capture_screen()
            print(f"[Screen] Captured {len(image_bytes):,} bytes")
        except Exception as e:
            return f"Screen capture failed: {e}"

        try:
            response = _call_vision(image_bytes, mime_type, question)
            self.last_response = response
            self.capture_count += 1

            # Detect subject
            subject = self._detect_subject(response)
            change_msg = self._check_subject_change(subject)

            # Speak if enabled
            if self.auto_speak and _speak_fn and response:
                _speak_fn(response)
                if change_msg:
                    time.sleep(1)  # Small delay between messages
                    _speak_fn(change_msg)

            return response
        except Exception as e:
            return f"Analysis failed: {e}"

    def _watch_loop(self) -> None:
        """Background loop - capture, analyze, speak."""
        print("[Screen] Watching loop started")

        while self._running.is_set():
            # Wait for pause to be set
            self._pause.wait(timeout=1)
            if not self._running.is_set():
                break

            # Capture and analyze
            try:
                self._capture_and_analyze()
            except Exception as e:
                print(f"[Screen] Error: {e}")

            # Wait for interval or until stopped/paused
            for _ in range(self.interval):
                if not self._running.is_set():
                    break
                self._pause.wait(timeout=1)

        print("[Screen] Watch loop ended")

# ─── Singleton instance ────────────────────────────────────────────────────────
_processor: Optional[ScreenProcessor] = None
_processor_lock = threading.Lock()

def get_processor() -> ScreenProcessor:
    """Get or create the global screen processor instance."""
    global _processor
    with _processor_lock:
        if _processor is None:
            _processor = ScreenProcessor()
        return _processor

# ─── CLI interface ─────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  ✦ Fairy Screen Processor")
    print("=" * 50)
    print()

    if not _MSS:
        print("ERROR: mss not installed")
        print("Run: pip install mss")
        return

    processor = get_processor()

    print("Commands:")
    print("  start [interval]  - Start watching (default: 10s)")
    print("  stop              - Stop watching")
    print("  once [question]   - One-time capture")
    print("  pause             - Pause watching")
    print("  resume            - Resume watching")
    print("  quit              - Exit")
    print()

    while True:
        try:
            cmd = input("✦ fairy screen> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if action in ("quit", "exit", "q"):
            processor.stop()
            break

        elif action == "start":
            interval = int(arg) if arg else 10
            processor.interval = interval
            processor.start(background=True)

        elif action == "stop":
            processor.stop()

        elif action == "pause":
            processor.pause()
            print("[Screen] Paused")

        elif action == "resume":
            processor.resume()
            print("[Screen] Resumed")

        elif action in ("once", "capture", "see"):
            question = arg or "What do you see on screen? Be brief and helpful."
            print(f"[Screen] Analyzing: {question}")
            response = processor.analyze_screen(question)
            print(f"\nFairy: {response}\n")

        elif action in ("help", "?"):
            print("Commands:")
            print("  start [interval]  - Start watching (default: 10s)")
            print("  stop              - Stop watching")
            print("  once [question]   - One-time capture")
            print("  pause             - Pause watching")
            print("  resume            - Resume watching")
            print("  quit              - Exit")

        else:
            # Treat as a question
            processor.analyze_screen(cmd)

if __name__ == "__main__":
    main()
