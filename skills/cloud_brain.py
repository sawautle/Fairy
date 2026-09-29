"""
cloud_brain.py

Cloud-only conversational brain for Fairy's free-hosted Discord deployment.

Fallback order:
    1. Gemini        (primary)
    2. Groq          (secondary)
    3. OpenRouter    (tertiary / fallback)
"""

import os
import logging
from pathlib import Path
import requests
from dotenv import load_dotenv

# Force load environment variables from controller/.env
env_path = Path(__file__).parent.parent / "controller" / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

logger = logging.getLogger("fairy.cloud_brain")
if os.environ.get("FAIRY_DEBUG") == "1":
    logger.setLevel(logging.DEBUG)
    log_path = os.environ.get("FAIRY_DEBUG_FILE", "fairy_debug.log")
    _handler = logging.FileHandler(log_path)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_handler)
else:
    logger.addHandler(logging.NullHandler())

_call_counter = {"gemini": 0, "groq": 0, "openrouter": 0, "fallback_events": 0}


def get_cloud_brain_stats():
    return dict(_call_counter)


class QuotaExceeded(Exception):
    """Raised when a provider signals it's out of free quota for now."""


def _sanitize_messages(messages):
    """Ensure message structures strictly satisfy API token validation requirements."""
    clean = []
    for m in messages:
        role = m.get("role", "user")
        content = str(m.get("content", "")).strip()
        if not content:
            content = "Hello"
        clean.append({"role": role, "content": content})
    return clean


# ---------------------------------------------------------------------------
# 1. Gemini
# ---------------------------------------------------------------------------
def ask_gemini(messages, timeout=30):
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key or gemini_key.startswith("your_"):
        raise RuntimeError("GEMINI_API_KEY not set or invalid in .env")

    model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
    
    clean_msgs = _sanitize_messages(messages)
    system_bits = [m["content"] for m in clean_msgs if m["role"] == "system"]
    contents = []
    
    for m in clean_msgs:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    if not contents:
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

    if system_bits:
        prefix = "\n".join(system_bits) + "\n\n"
        contents[0]["parts"][0]["text"] = prefix + contents[0]["parts"][0]["text"]

    resp = requests.post(url, json={"contents": contents}, timeout=timeout)
    if resp.status_code in (429, 403):
        raise QuotaExceeded(f"Gemini limit/auth: {resp.text[:200]}")
    resp.raise_for_status()

    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


# ---------------------------------------------------------------------------
# 2. Groq
# ---------------------------------------------------------------------------
def ask_groq(messages, timeout=30):
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key or groq_key.startswith("your_"):
        raise RuntimeError("GROQ_API_KEY not set or invalid in .env")

    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "messages": _sanitize_messages(messages)}

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code in (401, 403, 429):
        raise QuotaExceeded(f"Groq limit/auth: {resp.text[:200]}")
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def ask_cloud_brain(messages, use_openrouter_fallback=True):
    providers = [
        ("gemini", ask_gemini),
        ("groq", ask_groq),
    ]

    for name, fn in providers:
        try:
            result = fn(messages)
            _call_counter[name] += 1
            logger.debug("CLOUD_BRAIN_OK provider=%s", name)
            return result
        except QuotaExceeded as e:
            _call_counter["fallback_events"] += 1
            print(f"[{name.upper()} QUOTA/AUTH SKIPPED]: {e}")
        except Exception as e:
            _call_counter["fallback_events"] += 1
            print(f"[{name.upper()} FAILED]: {e}")

    if use_openrouter_fallback:
        try:
            from skills.openrouter import ask_or_chat
            result = ask_or_chat(_sanitize_messages(messages))
            _call_counter["openrouter"] += 1
            logger.debug("CLOUD_BRAIN_OK provider=openrouter")
            return result
        except Exception as e:
            print(f"[OPENROUTER FAILED]: {e}")

    raise RuntimeError("All cloud brain providers (Gemini, Groq, OpenRouter) failed.")