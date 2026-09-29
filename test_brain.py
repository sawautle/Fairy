"""
test_brain.py
Run from project root: E:\fairy
"""

from pathlib import Path
import os
from dotenv import load_dotenv

dotenv_path = Path(__file__).parent / "controller" / ".env"
load_dotenv(dotenv_path=dotenv_path)

print(f"=== Env File Path: {dotenv_path} (Exists: {dotenv_path.exists()}) ===")
print(f"GEMINI_API_KEY loaded: {bool(os.environ.get('GEMINI_API_KEY'))}")
print(f"GROQ_API_KEY loaded: {bool(os.environ.get('GROQ_API_KEY'))}")
print(f"CHARACTER_AI_TOKEN loaded: {bool(os.environ.get('CHARACTER_AI_TOKEN'))}")

from skills import cloud_brain

test_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello! Reply with 'Systems Online' if you can read this."}
]

print("\n=== Testing Cloud Brain Orchestrator ===")
try:
    response = cloud_brain.ask_cloud_brain(test_messages)
    print("\n[SUCCESS] Response received:")
    print(response)
    print("\nCall Statistics:")
    print(cloud_brain.get_cloud_brain_stats())
except Exception as e:
    print(f"\n[ERROR] Request failed: {e}")