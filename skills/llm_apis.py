"""Local Ollama API wrapper for Fairy 2.0."""
import json
import urllib.request
from config import OLLAMA_URL, MODEL_BRAIN, MODEL_CODER, log


def _ollama_chat(model: str, messages: list, stream: bool = False, temperature: float = 0.7) -> str:
    data = json.dumps({
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result.get("message", {}).get("content", "")


def ask_brain(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _ollama_chat(MODEL_BRAIN, messages, temperature=temperature)


def ask_coder(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return _ollama_chat(MODEL_CODER, messages, temperature=temperature)


def generate_code(task: str, error: str = "") -> str:
    from core.prompts import build_code_prompt
    prompt = build_code_prompt(task, error)
    return ask_coder(prompt)