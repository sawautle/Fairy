#!/usr/bin/env python3
"""
skills/vision_skill.py — Fairy's Vision Tool

This skill provides the brain (Gemma 4) with the ability to see.
The brain can call these functions as tools to analyze images.

Tool: vision_analyze
    Analyze an image file or screenshot.
    Args:
        mode: "screen" or "camera" (which to capture)
        question: What to ask about the captured image
        image_path: Path to an image file (optional)

Tool: vision_describe
    Describe a specific image file.
    Args:
        image_path: Path to the image file
        question: What to ask about it

Both tools use Llama Vision via OpenRouter.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from vision import (
    describe_image,
    capture_and_describe,
    is_vision_available,
    VisionResult,
)


def vision_analyze(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
) -> bool:
    """
    Capture screen/camera and analyze with Llama Vision.

    Called by the brain as a tool when it needs to see something.
    The result is returned via the response parameter or spoken via TTS.

    Parameters (via parameters dict):
        mode:     "screen" (default) or "camera"
        question: What to ask about the image
        image_path: Path to an image file (optional, overrides mode if provided)
        camera_index: Camera to use (default: 0)

    Returns True on success, False on failure.
    The actual description is printed/logged for the controller to handle.
    """
    params = parameters or {}
    mode = params.get("mode", "screen").lower().strip()
    question = (params.get("question") or params.get("text") or "").strip()
    image_path = params.get("image_path")
    camera_index = int(params.get("camera_index", 0))

    print(f"[Vision] Analyzing: mode={mode!r} question={question[:80]!r}")

    # Check availability
    if not is_vision_available():
        msg = "[Vision] Not available — OpenRouter API key not configured"
        print(msg)
        return False

    try:
        if image_path:
            # Analyze a specific image file
            path = Path(image_path)
            if not path.exists():
                print(f"[Vision] Image file not found: {image_path}")
                return False

            mime = _get_mime_type(path)
            image_bytes = path.read_bytes()
            result = describe_image(image_bytes, mime, question)
        else:
            # Capture and analyze
            result = capture_and_describe(question, angle=mode, camera_index=camera_index)

        if result.success:
            print(f"[Vision] {result.description[:200]}")
            return True
        else:
            print(f"[Vision] Error: {result.error}")
            return False

    except Exception as exc:
        print(f"[Vision] Exception: {exc}")
        return False


def vision_describe(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
) -> bool:
    """
    Describe a specific image file.

    Parameters (via parameters dict):
        image_path: Path to the image file
        question: What to ask about it (default: "Describe this image in detail")

    Returns True on success, False on failure.
    """
    params = parameters or {}
    image_path = params.get("image_path")
    question = (params.get("question") or params.get("text") or "").strip()

    if not image_path:
        print("[Vision] No image_path provided")
        return False

    path = Path(image_path)
    if not path.exists():
        print(f"[Vision] Image not found: {image_path}")
        return False

    try:
        mime = _get_mime_type(path)
        image_bytes = path.read_bytes()
        result = describe_image(image_bytes, mime, question)

        if result.success:
            print(f"[Vision] {result.description[:200]}")
            return True
        else:
            print(f"[Vision] Error: {result.error}")
            return False

    except Exception as exc:
        print(f"[Vision] Exception: {exc}")
        return False


def _get_mime_type(path: Path) -> str:
    """Determine MIME type from file extension."""
    ext = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return mime_map.get(ext, "image/jpeg")


# ─── Tool definitions for agent_controller ────────────────────────────────────
def get_vision_tools() -> list[dict]:
    """Return tool definitions for the vision skill."""
    return [
        {
            "type": "function",
            "function": {
                "name": "vision_analyze",
                "description": "Capture your screen or camera and analyze the image with Fairy's vision module. Use this when you need to see what's on screen or through the camera. Returns a description of what you see.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["screen", "camera"],
                            "description": "What to capture: 'screen' (default) captures your display, 'camera' captures from webcam",
                        },
                        "question": {
                            "type": "string",
                            "description": "What to ask about the captured image. E.g. 'What is on screen?' or 'Read any visible text'",
                        },
                        "camera_index": {
                            "type": "integer",
                            "description": "Which camera to use if mode='camera' (default: 0)",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "vision_describe",
                "description": "Describe a specific image file. Use this when the user sends an image or references an image file you should analyze.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the image file to analyze",
                        },
                        "question": {
                            "type": "string",
                            "description": "What to ask about the image. E.g. 'What do you see?' or 'Read the text in this image'",
                        },
                    },
                    "required": ["image_path"],
                },
            },
        },
    ]


# ─── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Vision Skill Self-Test")
    print("=" * 52)
    print(f"Vision available: {is_vision_available()}")
    print(f"Tools defined: {len(get_vision_tools())}")
    for tool in get_vision_tools():
        print(f"  - {tool['function']['name']}: {tool['function']['description'][:60]}...")
