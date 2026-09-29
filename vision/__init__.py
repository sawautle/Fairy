#!/usr/bin/env python3
"""
vision/__init__.py — Fairy's Eyes Module

Fairy has two distinct AI components:

    Brain  (Gemma 4 26B) — reasoning, personality, tool calling, memory
    Eyes   (Llama Vision) — screenshots, image understanding, UI reading

The vision module is independent from the brain. It provides:

    describe_image(image_bytes, question) -> dict
        Analyze an image and return structured description.

    capture_and_describe(question, angle="screen") -> dict
        Capture screen/camera and analyze.

    is_available() -> bool
        Check if vision service is configured and working.

Architecture:
    vision/
        __init__.py       — public API (describe_image, capture_and_describe)
        vision_client.py  — OpenRouter API client for Llama Vision
        vision_server.py  — local HTTP server for standalone vision (optional)
        screen_capture.py — screen and camera capture utilities

Usage:
    from vision import describe_image, capture_and_describe

    # Analyze a screenshot
    result = capture_and_describe("What is open on the screen?")
    print(result["description"])

    # Analyze user-provided image
    with open("image.png", "rb") as f:
        result = describe_image(f.read(), "image/png", "What do you see?")
"""
from __future__ import annotations

from vision.vision_client import (
    describe_image,
    is_vision_available,
    VisionResult,
    analyze_image_sync,
)

from vision.screen_capture import (
    capture_screen,
    capture_camera,
    compress_image,
)


def capture_and_describe(
    question: str = "",
    angle: str = "screen",
    camera_index: int = 0,
) -> VisionResult:
    """
    Convenience: capture screen or camera, then describe it.

    This is the main entry point the brain uses to "see" something.
    Gemma 4 (the brain) calls this when it needs to understand
    what's on screen or what the user is looking at.

    Args:
        question:     What to ask about the image
        angle:        "screen" (default) or "camera"
        camera_index: Which camera to use (if angle="camera")

    Returns:
        VisionResult with description.
    """
    try:
        if angle == "camera":
            image_bytes, mime = capture_camera(camera_index, compress=True)
        else:
            image_bytes, mime = capture_screen(compress=True)
    except Exception as exc:
        return VisionResult(
            success=False,
            error=f"Capture failed: {exc}",
        )

    return describe_image(image_bytes, mime, question)


# Re-export for convenience
__all__ = [
    "describe_image",
    "capture_and_describe",
    "is_vision_available",
    "VisionResult",
    "analyze_image_sync",
    "capture_screen",
    "capture_camera",
    "compress_image",
]
