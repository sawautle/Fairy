#!/usr/bin/env python3
"""Download OpenWakeWord model files into openwakeword_models/.

Run from the repo root: python scripts/download_oww.py
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(REPO_ROOT, "openwakeword_models")
os.makedirs(TARGET_DIR, exist_ok=True)

MODELS = {
    "hey_jarvis": "https://github.com/dscspidey/jarviswakeword/releases/download/v1.0.0/hey_jarvis.onnx",
    "hey_fairy":  "https://github.com/dscspidey/jarviswakeword/releases/download/v1.0.0/hey_fairy.onnx",
}


def main() -> None:
    try:
        import urllib.request
    except Exception as exc:
        print(f"[download_oww] urllib unavailable: {exc}", file=sys.stderr)
        sys.exit(1)

    for name, url in MODELS.items():
        dest = os.path.join(TARGET_DIR, f"{name}.onnx")
        if os.path.exists(dest):
            print(f"[download_oww] {name}.onnx already present — skipping")
            continue
        print(f"[download_oww] Downloading {name}.onnx from {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[download_oww] Saved {dest}")
        except Exception as exc:
            print(f"[download_oww] Failed to download {name}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
