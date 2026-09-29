#!/usr/bin/env python3
"""
vision/vision_server.py — Optional standalone HTTP server for local vision inference.

NOTE: Running Llama Vision locally requires significant GPU VRAM (~40GB).
This server is OPTIONAL. Most users will use OpenRouter (vision_client.py).

To use this server instead of OpenRouter:
    python -m vision.vision_server --port 8765

Then update config/models.json:
    "vision_provider": "http://localhost:8765"

The server exposes a simple REST API:
    POST /analyze
        Body: {"image": "<base64>", "mime": "image/png", "question": "..."}
        Returns: VisionResult JSON

    GET /health
        Returns: {"status": "ok", "model": "llama-3.2-11b-vision-uncensored"}

Architecture note:
    The vision_server is a separate process that loads the vision model.
    The brain (Gemma 4) never directly calls the vision model —
    it calls vision_client which routes to either OpenRouter or vision_server.

For local vision inference, consider:
    - Ollama with vision model (llama3.2-vision)
    - llama.cpp server
    - vLLM server
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Vision server is optional — graceful exit if not configured
VISION_SERVER_AVAILABLE = False


def run_server(port: int = 8765, host: str = "localhost"):
    """Run the vision HTTP server."""
    global VISION_SERVER_AVAILABLE
    VISION_SERVER_AVAILABLE = True

    print(f"Vision server starting on {host}:{port}")
    print("Note: Local vision inference requires ~40GB GPU VRAM")
    print("For most users, use vision_client.py with OpenRouter instead.")
    print()

    class VisionHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            print(f"[VisionServer] {args[0]}")

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "ok",
                    "note": "Vision server requires local inference setup",
                }).encode())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/analyze":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                try:
                    req = json.loads(body)
                    # TODO: Implement local vision inference
                    # For now, return an error directing to OpenRouter
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": "Local vision inference not implemented. "
                                 "Use OpenRouter via vision_client.py instead.",
                    }).encode())
                except json.JSONDecodeError:
                    self.send_response(400)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

    server = HTTPServer((host, port), VisionHandler)
    print(f"Vision server listening on http://{host}:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nVision server stopped")


def main():
    parser = argparse.ArgumentParser(description="Fairy's Vision HTTP Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
