#!/usr/bin/env python3
"""Ping the OpenRouter /models endpoint and print available free models.

Requires OPENROUTER_API_KEY in config/api_keys.json or the environment.
Exits 0 if at least one free model is listed; exits 1 otherwise.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH = os.path.join(REPO_ROOT, "config", "api_keys.json")
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
FREE_TAG = "free"


def load_api_key() -> str | None:
    for candidate in [
        os.environ.get("OPENROUTER_API_KEY"),
        os.path.join(REPO_ROOT, "config", "api_keys.json"),
    ]:
        if candidate and os.path.isfile(candidate):
            try:
                with open(candidate) as f:
                    data = json.load(f)
                    return data.get("OPENROUTER_API_KEY")
            except Exception:
                pass
        elif candidate:
            return candidate
    return None


def fetch_models(api_key: str) -> list[dict]:
    req = urllib.request.Request(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("data", [])
    except urllib.error.HTTPError as exc:
        print(f"[ping_openrouter] HTTP error {exc.code}: {exc.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"[ping_openrouter] Network error: {exc.reason}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key and os.path.exists(KEY_PATH):
        with open(KEY_PATH) as f:
            api_key = json.load(f).get("OPENROUTER_API_KEY")

    if not api_key:
        print("[ping_openrouter] No API key found — set OPENROUTER_API_KEY or config/api_keys.json")
        sys.exit(1)

    models = fetch_models(api_key)
    free_models = [m for m in models if FREE_TAG in m.get("tags", [])]

    print(f"=== OpenRouter Models ({len(models)} total, {len(free_models)} free) ===")
    for m in free_models:
        print(f"  {m['id']}")

    if not free_models:
        print("[ping_openrouter] No free models found!", file=sys.stderr)
        sys.exit(1)
    print(f"\n{len(free_models)} free model(s) available.")
    sys.exit(0)


if __name__ == "__main__":
    main()
