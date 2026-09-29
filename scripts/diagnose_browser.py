#!/usr/bin/env python3
"""Diagnose CDP browser connectivity issues for the Hermes browser harness.

Checks:
  1. Chrome remote-debugging port is reachable (localhost:9222).
  2. /json/version returns a valid CDP endpoint.
  3. Existing browser-data profile is usable.
  4. Any stale .lock files.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BROWSER_DATA = os.path.join(REPO_ROOT, "browser_data")
CDP_HOST = "localhost"
CDP_PORT = 9222
TIMEOUT = 5


def check_port(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT):
            return True
    except OSError:
        return False


def check_json_endpoint() -> dict | None:
    try:
        req = urllib.request.urlopen(
            f"http://{CDP_HOST}:{CDP_PORT}/json/version", timeout=TIMEOUT
        )
        return json.loads(req.read())
    except Exception as exc:
        return None


def check_lock_files() -> list[str]:
    locks = []
    for root, _dirs, files in os.walk(BROWSER_DATA):
        for fn in files:
            if fn.endswith(".lock"):
                locks.append(os.path.join(root, fn))
    return locks


def main() -> None:
    print("=== Browser Diagnosis ===")
    port_open = check_port(CDP_HOST, CDP_PORT)
    print(f"CDP port {CDP_HOST}:{CDP_PORT} open: {port_open}")

    if port_open:
        info = check_json_endpoint()
        if info:
            print(f"CDP /json/version: {json.dumps(info, indent=2)}")
        else:
            print("CDP /json/version unreachable (browser may not be in debug mode)")

    locks = check_lock_files()
    if locks:
        print(f"Stale lock files: {locks}")
    else:
        print("No stale lock files found.")

    if not port_open:
        print(
            f"\nNo browser detected on {CDP_HOST}:{CDP_PORT}.\n"
            "Start Chrome with: chrome --remote-debugging-port=9222"
        )
        sys.exit(1)
    print("\nBrowser diagnosis complete.")


if __name__ == "__main__":
    main()
