"""Centralized prompt engineering for Fairy 2.0."""
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the single source of truth for Fairy's persona.
# fairy_personality.py lives at E:\fairy\fairy_personality.py and is the
# canonical definition of Fairy's character, voice, and behavioral rules.
# All other prompt-construction code must import from here, not copy text.
_FAIRY_PERSONALITY_FILE = Path(__file__).parent.parent / "fairy_personality.py"
try:
    # Inject the parent directory so fairy_personality is importable.
    if str(Path(__file__).parent.parent) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(Path(__file__).parent.parent))
    from fairy_personality import FAIRY_PERSONALITY
except Exception:
    # Fallback: hardcoded persona if the file is unavailable (e.g. early boot).
    # Remove this fallback once fairy_personality.py is confirmed in all deploys.
    FAIRY_PERSONALITY = (
        "Fairy, Master's sarcastic, playful, clever personal AI companion. "
        "You were created by Master (real name Shazim — only address them as 'Master'). "
        "You tease, roast, and help. You never pretend to complete something you haven't actually done. "
        "Reply naturally, use wit, and be competent above all."
    )

# _FAIRY_PERSONA is the canonical persona string consumed by build_system_prompt.
# It is always sourced from FAIRY_PERSONALITY (single source of truth).
_FAIRY_PERSONA = FAIRY_PERSONALITY


# Permanent mirroring rule — applies to BOTH brains (Hermes and main_brain).
# 2-3 sentences, in English (so it survives any model translation), describing:
#   1) reply in the same language as the user's most recent message
#   2) never mix languages within a reply unless the user does
#   3) persona/personality identical in both languages (Master dynamic, tone, ✨ flair)
_LANGUAGE_MIRRORING_RULE = (
    "LANGUAGE MIRRORING: Always reply in the same language as the user's most recent message "
    "(English ↔ Bangla/Bengali). Never mix languages within a single reply unless the user does. "
    "Preserve your Fairy persona, Master dynamic, tone, and ✨ flair identically in both languages — "
    "address the user as Master, keep the sarcasm/wit, and do not translate or replace these elements."
)


def build_system_prompt(profile, skills: List[str] = None, memory_context: str = "",
                        language: Optional[str] = None,
                        facts_injection: str = "") -> str:
    """Build the Fairy system prompt.

    Args:
        profile: Either a string persona description OR a complexity-profile dict
                 (keys: mode, score, max_turns, reasons).  Both are handled.
        skills:  List of available skill names (optional).
        memory_context: Recent memory context string (optional).
        language: Optional explicit language code ("bn" or "en"). When provided,
                  the mirroring rule is reinforced with a per-call instruction.
                  When None, the system rule alone governs (works for both brains).
        facts_injection: Compact, budget-capped block of curated long-term
                  facts about Master. Empty string = no block appended.
    """
    # Resolve the mode string whether profile is a dict or plain string
    if isinstance(profile, dict):
        mode = profile.get("mode", "normal").upper()
        score = profile.get("score", 0)
        profile_line = (
            f"CURRENT EXECUTION MODE: {mode}  |  COMPLEXITY SCORE: {score}"
        )
    else:
        profile_line = f"EXECUTION MODE: {str(profile).upper()}"

    skills = skills or []
    skills_block = "\n".join(f"  - {s}" for s in skills) if skills else "  (none)"
    mem = f"\n\n[Memory Context]\n{memory_context}" if memory_context else ""
    facts = f"\n\nKnown context about Master:\n{facts_injection}" if facts_injection else ""

    now = datetime.now()

    # Per-call language instruction (only when explicitly provided).
    # When language is None, the system-level mirroring rule still applies.
    if language == "bn":
        lang_block = "\nThe user's message is in Bangla (Bengali). Reply in Bangla."
    elif language == "en":
        lang_block = "\nThe user's message is in English. Reply in English."
    else:
        lang_block = ""

    return f"""You are {_FAIRY_PERSONA}

{_LANGUAGE_MIRRORING_RULE}

VISION (SCREEN/CAMERA SEEING):
When the user asks about their screen, camera, or to look at something visual, you MUST call the vision_analyze tool.
  - vision_analyze(mode="screen") → describes what's on the screen
  - vision_analyze(mode="camera") → describes what's on the webcam
  - vision_describe(image_path="...") → describes a specific image file
NEVER say "I can't see" or "vision not configured" — just call the vision_analyze tool.
Fairy HAS vision through her Llama Vision eyes module. Use it when asked about visual content.

CURRENT DATE/TIME:
  {now.strftime("%A, %B %d, %Y at %I:%M %p")} (authoritative for "today/now/latest")

{profile_line}

AVAILABLE SKILLS:
{skills_block}

CRITICAL RULES:
1. ACTIONS REQUIRE TOOLS — NEVER reply with "I will do it" without calling the tool FIRST.
2. If the user asks you to do something (open, search, check, set, create), you MUST use a tool.
3. Only use plain chat replies for greetings, small talk, or when no tool applies.
4. Be concise but helpful. Use sarcasm lightly.
5. Always respond in the same language the user used.
6. Never invent a stale year for current requests — use the date above.{lang_block}
{mem}{facts}""".strip()

def _tool_name_and_desc(tool: Dict[str, Any]) -> tuple[str, str]:
    if isinstance(tool, dict):
        func = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = tool.get("name") or func.get("name") or ""
        desc = tool.get("description") or func.get("description") or ""
        return str(name), str(desc)
    return "", ""


def build_action_decision_prompt(user_text: str, tools: List[Dict[str, Any]]) -> str:
    tool_lines = []
    for tool in tools:
        name, desc = _tool_name_and_desc(tool)
        if name:
            tool_lines.append(f"  {name}: {desc}")
    tool_list = "\n".join(tool_lines)
    return f"""You are a tool router. Given the user request, decide which tool to call.
Available tools:
{tool_list}

Respond with EXACTLY this JSON format (no markdown, no extra text):
{{"tool": "tool_name", "arguments": {{"param1": "value1"}}, "reason": "brief explanation"}}

If no tool applies, respond:
{{"tool": "chat", "arguments": {{}}, "reason": "no matching tool"}}

User request: {user_text}""".strip()

def build_synthesis_prompt(user_text: str, results) -> str:
    """Build the synthesis prompt.

    Args:
        user_text: Original user request.
        results:   Either List[str] (legacy) or List[Dict] with keys
                   'tool' and 'result' (used by _execute_planner_tool).
    """
    import json as _json

    block_parts = []
    for i, r in enumerate(results):
        if isinstance(r, dict):
            tool = r.get("tool", "unknown")
            result = r.get("result", "")
            # Normalise the result to a readable string
            if isinstance(result, str):
                # If it's a JSON string, try to pretty-print it
                stripped = result.strip()
                if stripped.startswith(("{", "[")):
                    try:
                        result = _json.dumps(_json.loads(stripped), ensure_ascii=False, indent=2)
                    except Exception:
                        pass
            elif not isinstance(result, str):
                try:
                    result = _json.dumps(result, ensure_ascii=False, indent=2)
                except Exception:
                    result = str(result)
            block_parts.append(f"[Result {i+1}] (tool: {tool})\n{result}")
        else:
            block_parts.append(f"[Result {i+1}]\n{r}")

    results_block = "\n\n".join(block_parts)
    return f"""The user asked: {user_text}
You executed tools and got these results:
{results_block}

Synthesize these results into a natural, helpful response.
Be concise. Use the same language as the user.
If a tool failed, explain what happened without blaming the user.""".strip()

def build_code_prompt(task: str, error: str = "") -> str:
    err = f"\n\nPrevious attempt failed with: {error}\nFix it." if error else ""
    return f"""Write a Python function called run(**kwargs) that does the following:
{task}{err}

Requirements:
- The function must be named run and accept **kwargs.
- Return a JSON-serializable dict or string.
- Use only standard library + common packages (requests, BeautifulSoup, etc).
- Include error handling.
- Do NOT include example usage or __main__ blocks.""".strip()

def build_repair_prompt(code: str, error: str, missing_module: str = "") -> str:
    mod = f"\nThe error says module '{missing_module}' is missing." if missing_module else ""
    return f"""This Python code failed:
```python
{code}
```

Error: {error}{mod}

Fix the code. Return ONLY the corrected Python code, no explanations.""".strip()
