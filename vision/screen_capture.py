#!/usr/bin/env python3
"""
vision/screen_capture.py — Fairy's Eyes: Screen and camera capture utilities.

Provides screen and camera capture with compression for vision analysis.
Used by capture_and_describe() in the vision module.

Dependencies:
    - mss: for screen capture (pip install mss)
    - cv2: for camera capture (pip install opencv-python)
    - PIL: for image compression (pip install Pillow)
All are optional — graceful degradation if not available.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

# ─── Optional dependencies ─────────────────────────────────────────────────────
try:
    import mss
    import mss.tools
    _MSS = True
except ImportError:
    _MSS = False

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    import PIL.Image
    _PIL = True
except ImportError:
    _PIL = False

# ─── Compression settings ──────────────────────────────────────────────────────
IMG_MAX_W = 1280
IMG_MAX_H = 720
JPEG_Q = 82


def compress_image(
    img_bytes: bytes,
    source_format: str = "PNG",
    max_w: int = IMG_MAX_W,
    max_h: int = IMG_MAX_H,
    quality: int = JPEG_Q,
) -> tuple[bytes, str]:
    """
    Compress image bytes to JPEG for efficient API submission.

    Args:
        img_bytes:     Raw image data
        source_format: Original format (PNG, BMP, etc.)
        max_w:         Max width in pixels
        max_h:         Max height in pixels
        quality:       JPEG quality (1-100)

    Returns:
        (compressed_bytes, mime_type)
    """
    if not _PIL:
        return img_bytes, f"image/{source_format.lower()}"

    try:
        img = PIL.Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((max_w, max_h), PIL.Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=False)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        print(f"[ScreenCapture] ⚠️  Compression failed: {e}")
        return img_bytes, f"image/{source_format.lower()}"


def capture_screen(
    monitor_index: int = 1,
    compress: bool = True,
) -> tuple[bytes, str]:
    """
    Capture the primary screen.

    Args:
        monitor_index: Which monitor to capture (1 = primary)
        compress:      Compress to JPEG for API efficiency

    Returns:
        (image_bytes, mime_type)

    Raises:
        RuntimeError if mss is not installed
    """
    if not _MSS:
        raise RuntimeError(
            "mss is not installed. Run: pip install mss. "
            "Screen capture is unavailable."
        )

    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            monitor_index = 1  # Fallback to primary
        target = monitors[monitor_index]
        shot = sct.grab(target)
        png_data = mss.tools.to_png(shot.rgb, shot.size)

    if compress:
        return compress_image(png_data, "PNG")
    return png_data, "image/png"


def capture_camera(
    camera_index: int = 0,
    compress: bool = True,
) -> tuple[bytes, str]:
    """
    Capture a single frame from the specified camera.

    Args:
        camera_index: Which camera to use (0 = default)
        compress:     Compress to JPEG for API efficiency

    Returns:
        (image_bytes, mime_type)

    Raises:
        RuntimeError if OpenCV is not installed or camera unavailable
    """
    if not _CV2:
        raise RuntimeError(
            "OpenCV (cv2) is not installed. Run: pip install opencv-python. "
            "Camera capture is unavailable."
        )

    # Determine backend based on OS
    backend = cv2.CAP_DSHOW  # Windows
    if sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2

    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {camera_index} could not be opened")

    # Warmup: discard first few frames
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Camera index {camera_index} returned no frame")

    # Convert to JPEG
    if _PIL:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb)
        buf = io.BytesIO()
        if compress:
            img.thumbnail((IMG_MAX_W, IMG_MAX_H), PIL.Image.BILINEAR)
            img.save(buf, format="JPEG", quality=JPEG_Q)
        else:
            img.save(buf, format="PNG")
        return buf.getvalue(), "image/jpeg" if compress else "image/png"
    else:
        # Fallback to raw OpenCV encoding
        if compress:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        else:
            _, buf = cv2.imencode(".png", frame)
        return buf.tobytes(), "image/jpeg" if compress else "image/png"


# ─── Simple self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Screen Capture Self-Test")
    print("=" * 52)
    print(f"mss available: {_MSS}")
    print(f"cv2 available: {_CV2}")
    print(f"PIL available: {_PIL}")
    print()

    if _MSS:
        print("Testing screen capture...")
        try:
            data, mime = capture_screen(compress=True)
            print(f"  ✓ Screen captured: {len(data):,} bytes, {mime}")
        except Exception as e:
            print(f"  ✗ Screen capture failed: {e}")
    else:
        print("  ⚠ mss not installed — screen capture disabled")

    if _CV2:
        print("Testing camera capture...")
        try:
            data, mime = capture_camera(compress=True)
            print(f"  ✓ Camera captured: {len(data):,} bytes, {mime}")
        except Exception as e:
            print(f"  ✗ Camera capture failed: {e}")
    else:
        print("  ⚠ OpenCV not installed — camera capture disabled")
