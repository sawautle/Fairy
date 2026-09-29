#!/usr/bin/env python3
"""
Fairy — Discord front-end
==========================
Drop this file in E:\\fairy\\ next to agent_controller.py.

This does NOT touch Fairy's brain. It's a thin adapter that:
  1. Receives a Discord message
  2. Calls agent_controller.handle_request(text, history, on_status=...)
     in a background thread (so Ollama / web / browser calls never
     freeze the bot's event loop)
  3. Streams status updates ("Searching the web...", etc.) by editing
     a placeholder message, then replaces it with the final answer
  4. Keeps per-channel conversation history in memory, same shape
     Fairy's terminal loop already used: list[{"role":..., "content":...}]
  5. Drops occasional reaction GIFs during banter / jokes / sarcasm,
     never during task execution or factual answers.

SETUP
-----
1. pip install discord.py python-dotenv

2. Create a bot application:
   https://discord.com/developers/applications -> New Application
   -> Bot tab -> Reset Token (copy it)
   -> Enable "MESSAGE CONTENT INTENT" under Privileged Gateway Intents
   -> OAuth2 -> URL Generator -> scopes: bot
      permissions: Send Messages, Read Message History, Embed Links
   -> Use the generated URL to invite it to your server

3. Create a .env file next to this script:
   DISCORD_BOT_TOKEN=your_token_here
   FAIRY_DISCORD_OWNER_ID=your_discord_user_id   # right-click your name -> Copy User ID (enable Developer Mode in Discord settings first)
   FAIRY_DISCORD_ALLOW_ALL=1                     # 1 = everyone can chat/search, only Master gets restricted tools
                                                  # 0 = nobody except Master gets a response at all
   FAIRY_GIFS=1                                    # 1 = reaction GIFs on banter (default). 0 = disabled.

4. Run it:
   python discord_bot.py

USAGE IN DISCORD
-----------------
- DM the bot directly, OR
- In a server channel, either @mention the bot or prefix with "!fairy "
- Each channel/DM keeps its own conversation history
- Send "!fairy reset" to clear history for that channel

PERMISSIONS
-----------
- Master (FAIRY_DISCORD_OWNER_ID) gets every tool: browser control,
  computer control, screenshots, messaging apps, code execution/skill
  creation, and paid cloud APIs. Master is also the only one ever
  routed through Hermes (if FAIRY_USE_HERMES=1) since Hermes has its
  own tool execution we don't control here.
- Everyone else (when FAIRY_DISCORD_ALLOW_ALL=1) can chat and ask
  factual/informational questions (web search, page fetch, deep
  research, maps, current time) freely, but any attempt at a
  restricted action gets a fixed sarcastic denial instead of running.
  Their requests are always handled by agent_controller's own guarded
  path, never by Hermes.
- Any leftover "Master" in a reply to a non-owner is swapped for their
  Discord display name as a safety net, in case an internal fast-path
  or fallback string hardcodes it.
- The restricted tool list lives in RESTRICTED_TOOLS below — edit it
  if you want to lock down or open up specific capabilities.

REACTION GIFS
-------------
Fairy occasionally drops a GIF during jokes, sarcasm, and playful banter.
She never GIFs during tasks (open YouTube, calculations, searches, etc.)
or long factual answers.

To add your own GIFs, paste Tenor / Giphy / Imgur direct image URLs into
GIF_LIBRARY below. Categories: laugh, sarcastic, shocked, crying, excited.
"""

import functools
import os
import re
import sys
import asyncio
import json
import secrets
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import discord
from dotenv import load_dotenv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

# ── Make sure Python can find agent_controller.py in this same folder ──
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from controller import agent_controller  # noqa: E402
from controller.agent_controller import _log_chat_turn  # FIX 3: chat-path logging

# Load .env explicitly from THIS script's folder, not the current working
# directory. This avoids the classic bug where the bot is launched from a
# different folder (e.g. `python controller\discord_bot.py` from E:\fairy)
# and load_dotenv() silently finds nothing / the wrong file.
_DOTENV_PATH = os.path.join(_SCRIPT_DIR, ".env")
_loaded = load_dotenv(dotenv_path=_DOTENV_PATH, override=True)
print(f"[FAIRY-DISCORD] .env path: {_DOTENV_PATH}")
print(f"[FAIRY-DISCORD] .env found and loaded: {_loaded}")

BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
OWNER_ID = os.environ.get("FAIRY_DISCORD_OWNER_ID", "").strip()
ALLOW_ALL = os.environ.get("FAIRY_DISCORD_ALLOW_ALL", "0").strip() == "1"
GIFS_ENABLED = os.environ.get("FAIRY_GIFS", "1").strip() == "1"
print(f"[FAIRY-DISCORD] OWNER_ID loaded as: '{OWNER_ID}'")
print(f"[FAIRY-DISCORD] ALLOW_ALL loaded as: {ALLOW_ALL}")
print(f"[FAIRY-DISCORD] GIFs enabled: {GIFS_ENABLED}")
COMMAND_PREFIX = "!fairy "

if not BOT_TOKEN:
    print("ERROR: DISCORD_BOT_TOKEN is not set. Create a .env file (see header of this script).")
    sys.exit(1)

# ── Single-instance lock ────────────────────────────────────────────
# Running discord_bot.py more than once at the same time (e.g. once via
# fairy.py's auto-launch AND once via start_fairy.bat, or just forgetting
# a terminal was left open) makes every message get answered multiple
# times — each instance has its own independent Discord connection.
# A PID lock file makes a second launch refuse to start instead of
# silently duplicating.
_LOCK_PATH = os.path.join(_SCRIPT_DIR, ".discord_bot.lock")


def _pid_is_running(pid: int) -> bool:
    try:
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        # Not on Windows / can't check — assume it might still be running
        # rather than risk a false "safe to start".
        return True


def _acquire_single_instance_lock():
    if os.path.exists(_LOCK_PATH):
        try:
            with open(_LOCK_PATH, "r") as f:
                old_pid = int(f.read().strip())
        except Exception:
            old_pid = None

        if old_pid and _pid_is_running(old_pid):
            print(f"ERROR: Fairy's Discord bot is already running (PID {old_pid}).")
            print(f"       Refusing to start a second instance — this is what caused")
            print(f"       triplicate replies before. If you're SURE nothing is")
            print(f"       actually running, delete this file and try again:")
            print(f"       {_LOCK_PATH}")
            sys.exit(1)
        else:
            print(f"[FAIRY-DISCORD] Stale lock file found (PID {old_pid} not running) — clearing it.")
            try:
                os.remove(_LOCK_PATH)
            except OSError:
                pass

    with open(_LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))

    import atexit
    def _release_lock():
        try:
            if os.path.exists(_LOCK_PATH):
                with open(_LOCK_PATH, "r") as f:
                    if f.read().strip() == str(os.getpid()):
                        os.remove(_LOCK_PATH)
        except Exception:
            pass
    atexit.register(_release_lock)


_acquire_single_instance_lock()

# ── Discord client setup ──────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True  # required to read message text
intents.messages = True         # required for message replies in approval queue
intents.reactions = True        # required for raw_reaction_add in approval queue

client = discord.Client(intents=intents)

# One worker thread pool so Fairy's blocking brain calls never block
# Discord's asyncio event loop.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="FairyBrain")

# Per-channel conversation history: {channel_id: [ {role, content}, ... ]}
_histories: dict[int, list] = {}


# ═══════════════════════════════════════════════════════════════════
# TIERED PERMISSIONS
# ═══════════════════════════════════════════════════════════════════
# Master (OWNER_ID) gets every tool, including anything that touches the
# real PC (browser control, screenshots, volume, messaging apps, code
# execution/skill creation, and paid cloud APIs).
#
# Everyone else gets read-only / informational tools only: web search,
# page fetch, deep research, maps, current time, and the free local
# second-opinion model. They can chat and ask questions freely, but any
# attempt at a restricted action gets a sarcastic denial instead of
# actually running.
#
# This is enforced by monkey-patching agent_controller's own tool
# dispatch functions at runtime — agent_controller.py itself is never
# modified.

RESTRICTED_TOOLS = {
    # Browser / computer control
    "navigate_to", "search_on_site", "add_to_cart_amazon",
    "upload_instagram_reel", "click_element", "type_text",
    "get_page_info", "close_browser", "browser_control",
    "computer_control", "take_screenshot",
    # Notifications / reminders / messaging
    "set_reminder", "send_notification", "reminder_tool", "send_message",
    # System telemetry (arguably fine to expose, kept locked down by default)
    "system_monitor",
    # Vision (uses the webcam / captures the real screen)
    "screen_process",
    # Code execution / self-modifying behavior
    "create_skill", "run_skill", "code_sandbox",
    # Paid cloud APIs (cost money on Master's account)
    "ask_openai", "ask_claude", "ask_gemini", "ask_grok", "ask_meta",
    "ask_deepseek", "ask_kimi", "ask_openrouter", "ask_or_coder",
    "ask_or_smart", "ask_or_cheap",
    # File-system mutations (unified approval gate, owner + non-owner).
    # Non-owners see the existing Discord-queue prompt. Owners see the
    # new in-terminal ✅/❌ prompt (or the Discord queue, if the bot
    # is the active front-end). Both paths are handled by
    # controller/approval_gate.py; this set just makes sure non-owners
    # can't bypass the gate by skipping the call into agent_controller.
    "write_file", "mkdir", "delete", "zip_create", "zip_extract",
    "move", "rename",
}

DENIAL_MESSAGES = [
    "Not for you. Only Master has the keys to that. Go ask something I can actually help you with.",
    "Sorry, that one's Master-only. I do trivia and web searches for everyone else — try one of those.",
    "Nice try. That button belongs to Master alone. Ask me something I'm allowed to answer instead.",
    "That's above your clearance level. Master gets the remote control, you get the chat window.",
    "Denied — that's a Master-only privilege. I'm still happy to look things up for you though.",
    "Only Master can make me do that. Ask me a question instead and I'll actually help.",
]

# Safety net for the OWNER side of addressing: the underlying model can
# free-style pet names ("Darling", "hun", "boss"...) instead of sticking
# to "Master" even when it's genuinely talking to Master. Word-boundary
# swap any of these back to "Master" in owner replies, same mechanism
# as the non-owner "Master" -> username swap below.
_OWNER_ADDRESS_ALIASES = re.compile(
    r"\b(darling|dear|hun|honey|sweetie|sweetheart|love|boss|sir|buddy|pal|dude)\b",
    re.IGNORECASE,
)

# Thread-local so concurrent Discord requests (each handled by a worker
# thread) never leak permission state into each other.
_request_ctx = threading.local()


# ═══════════════════════════════════════════════════════════════════
# MASTER APPROVAL QUEUE
# ═══════════════════════════════════════════════════════════════════
# Non-Master users triggering a Master-only tool should NOT be hard-rejected
# any more — instead, Fairy escalates to the Master (you) for a yes/no
# before running the tool. This block holds:
#
#   * The exception raised from inside the worker thread when a restricted
#     tool is attempted by a non-owner. The async side catches it after
#     run_in_executor returns, posts the approval request, and waits on an
#     asyncio.Event for the Master's decision.
#   * The FIFO of pending requests. Hard cap of MAX_PENDING so the Master
#     isn't spammed if many non-owners hit Master-only tools at once.
#   * The audit log file. One line per request, append-only.
#
# All decision logic (parsing reactions / text replies, formatting log
# lines, computing timeout outcomes) lives in pure functions at the bottom
# of this section so it can be unit-tested without a live Discord client.

# Outcomes — also used as the final-emoji/edit-suffix on the request
# message so the Master has a single visual state machine.
APPROVAL_APPROVED = "approved"
APPROVAL_DENIED = "denied"
APPROVAL_TIMED_OUT = "timed_out"

# How long to wait for the Master to react or reply.
APPROVAL_TIMEOUT_SECONDS = 60.0

# Max concurrent pending requests. Beyond this, new ones auto-deny so
# the Master doesn't drown.
APPROVAL_MAX_PENDING = 3

# Audit log — append-only, lives next to the bot.
_APPROVAL_LOG_PATH = os.path.join(_SCRIPT_DIR, "discord_approvals.log")


class MasterApprovalRequired(Exception):
    """Raised inside the worker thread when a non-Master user requests a
    restricted tool. The async side catches this, posts an approval
    request to the Master, and waits on the attached asyncio.Event."""

    def __init__(self, request_info: dict, event: asyncio.Event):
        # request_info: {tool, args, user_id, user_name, channel_id, text}
        self.request_info = request_info
        self.event = event
        super().__init__(f"MasterApprovalRequired: tool={request_info.get('tool')}")


# FIFO of pending approval requests. The "head" of the list is the
# currently-awaiting one; anything beyond the cap is auto-denied.
# Each entry is a dict with keys: tool, args, user_id, user_name,
# channel_id, text, enqueued_at (datetime), event (asyncio.Event).
_pending_approvals: list = []
_pending_lock = threading.Lock()


def _format_tool_summary(tool: str, args) -> str:
    """Pure: one-line human summary of a tool+args, used in the request
    message. Falls back to a defensive repr if args isn't a dict."""
    if not isinstance(args, dict):
        return f"{tool}({args!r})"
    parts = []
    for k, v in args.items():
        sv = str(v)
        if len(sv) > 80:
            sv = sv[:77] + "..."
        parts.append(f"{k}={sv!r}")
    body = ", ".join(parts) if parts else ""
    return f"{tool}({body})"


def _build_request_message(user_name: str, tool: str, args) -> str:
    """Pure: the body of the approval request posted in the channel the
    Master can see. One block, mention at the top so it pings."""
    summary = _format_tool_summary(tool, args)
    return (
        f"@Master — **{user_name}** is requesting: "
        f"`{summary}`. Allow? (react ✅ / ❌, or reply allow/deny)"
    )


def _format_audit_line(*, timestamp: datetime, user_name: str, user_id,
                        tool: str, args, outcome: str) -> str:
    """Pure: one audit-log line per request, machine-parseable.
    The format is JSON-ish but kept one line so tail -f stays readable."""
    record = {
        "ts": timestamp.isoformat(timespec="seconds"),
        "user": user_name,
        "user_id": str(user_id) if user_id is not None else "",
        "tool": tool,
        "args": args if isinstance(args, dict) else {"_raw": repr(args)},
        "outcome": outcome,
    }
    return json.dumps(record, ensure_ascii=False, default=str)


def _append_audit(**kwargs) -> None:
    """Append a single record to the audit log; never raise into the chat path."""
    try:
        line = _format_audit_line(**kwargs)
        with open(_APPROVAL_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception as exc:  # pragma: no cover - audit must never break chat
        print(f"[FAIRY-DISCORD] approval audit log write failed: {exc}")


def _parse_master_reply(text: str) -> str | None:
    """Pure: parse the Master's text reply to the request message.
    Returns one of APPROVAL_APPROVED / APPROVAL_DENIED, or None if the
    message isn't a clear decision. Recognized (case-insensitive, whole-word):
        allow, approve, approved, yes, yep, yeah, ok, okay, sure, do it
        deny, denied, no, nope, nah, don't, dont
    Anything else returns None so the request stays pending.
    """
    if not text:
        return None
    t = text.lower().strip().rstrip(".!?,;:")
    if not t:
        return None
    APPROVE = {"allow", "approve", "approved", "yes", "yep", "yeah",
               "y", "ok", "okay", "sure", "do it", "go", "go ahead",
               "fine", "affirmative", "lgtm"}
    DENY = {"deny", "denied", "deny", "no", "nope", "nah", "n",
            "don't", "dont", "negative", "refuse", "reject", "stop",
            "not now", "no way"}
    # Whole-word check.
    tokens = re.findall(r"[a-z']+", t)
    joined = " ".join(tokens)
    if joined in APPROVE or joined in DENY:
        return APPROVAL_APPROVED if joined in APPROVE else APPROVAL_DENIED
    # First token alone can be a decision.
    if tokens:
        first = tokens[0]
        if first in APPROVE:
            return APPROVAL_APPROVED
        if first in DENY:
            return APPROVAL_DENIED
    return None


def _parse_reaction(emoji) -> str | None:
    """Pure: extract a decision from a discord reaction emoji. Accepts
    either a unicode emoji (✅) or a partial emoji object that
    exposes .name / .id. Returns APPROVAL_APPROVED / APPROVAL_DENIED, or
    None if the reaction isn't a recognized decision emoji."""
    if emoji is None:
        return None
    # discord.PartialEmoji / Emoji — try .name first.
    name = getattr(emoji, "name", None)
    if name == "✅":
        return APPROVAL_APPROVED
    if name == "❌":
        return APPROVAL_DENIED
    # Fallback: stringify.
    s = str(emoji)
    if s == "✅":
        return APPROVAL_APPROVED
    if s == "❌":
        return APPROVAL_DENIED
    return None


def _format_outcome_suffix(outcome: str) -> str:
    """Pure: the final-state emoji+text appended to the request message
    when the request resolves. Kept in one place so the test suite can
    assert the same suffix the on_message code uses."""
    if outcome == APPROVAL_APPROVED:
        return "✅ Approved"
    if outcome == APPROVAL_DENIED:
        return "❌ Denied"
    if outcome == APPROVAL_TIMED_OUT:
        return "⏰ Timed out"
    return outcome


# ═══════════════════════════════════════════════════════════════════
# RESPONSE CHUNKING (Feature 2)
# ═══════════════════════════════════════════════════════════════════
# Discord hard-caps messages at 2000 chars. Long Fairy replies used to
# get sliced at byte 2000 and drop the tail. This splitter prefers
# paragraph and line boundaries; it only falls back to mid-word / raw
# char splits when nothing else fits. Codeblocks are preserved across
# chunk boundaries by closing and re-opening the fence.

CHUNK_LIMIT = 2000  # Discord hard limit; keep exported so tests can import it.


def _chunk_text(text: str, limit: int = CHUNK_LIMIT) -> list:
    """Pure: split `text` into chunks each <= `limit` chars.

    Strategy:
      1. If text fits, return as-is.
      2. For each 2000-char window, find the best natural break
         (paragraph > line > sentence > space > hard slice).
      3. Codeblock special case: if the chosen cut would land inside an
         open ``` block, scan forward within the next `limit` chars to
         find the block's close, and split just after it. If the close
         is too far away, fall back to closing the block here (close +
         re-open on next chunk) so we never hard-split mid-block.
      4. When a chunk ends with an open fence, close it; the next chunk
         re-opens so every chunk renders as a valid Discord message.
      5. Last resort: hard slice at the limit, never breaking a chunk
         in a way that exceeds the limit.
    """
    if not text:
        return [""]
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(text) <= limit:
        return [text]

    chunks: list = []
    remaining = text
    # safety: maximum iterations before hard-slicing. Even the worst
    # no-boundary input only needs ceil(len(text) / limit) iterations.
    safety = max(1000, len(text) * 2 // max(1, limit) + 100)

    def _find_cut(head: str, limit: int) -> int:
        """Return the best cut position within `head`, preferring natural
        boundaries. Returns a value in [1, limit]. Does NOT account for
        codeblock fences — caller handles that."""
        # 1) Paragraph break
        idx = head.rfind("\n\n")
        if idx > 0:
            return idx + 2
        # 2) Line break
        idx = head.rfind("\n")
        if idx > 0:
            return idx + 1
        # 3) Sentence end
        m = re.search(r"[.!?]\s", head[::-1])
        if m:
            return limit - m.start()
        # 4) Word boundary
        idx = head.rfind(" ")
        if idx > 0:
            return idx + 1
        # 5) Last resort
        return limit

    while len(remaining) > limit:
        safety -= 1
        if safety < 0:
            # Hard-slice remaining and stop.
            while len(remaining) > limit:
                chunks.append(remaining[:limit])
                remaining = remaining[limit:]
            break

        head = remaining[:limit]
        cut = _find_cut(head, limit)
        if cut <= 0 or cut > limit:
            cut = limit

        # --- Codeblock guard: if `cut` lands inside an open ``` block,
        # prefer to extend the cut to include the close (if reachable).
        # This keeps the codeblock content intact across chunks. If the
        # close is too far away, fall back to fence-balancing below.
        fence_count_pre = remaining[:cut].count("```")
        if fence_count_pre % 2 == 1:
            # We're inside an open codeblock at the cut point. Search
            # forward in `remaining[cut:]` for the next ```. If it's
            # within reach AND fits in this chunk, extend the cut to
            # just past it. Otherwise leave the cut alone and let
            # fence-balancing below close/reopen.
            rest_text = remaining[cut:]
            close_pos = rest_text.find("```")
            if close_pos >= 0:
                extended_cut = cut + close_pos + 3  # include the ```
                # Only extend if it fits within the limit AND the close
                # is actually present (not EOF).
                if extended_cut <= limit and extended_cut <= len(remaining):
                    cut = extended_cut
                else:
                    # Close too far away. The chunk will end with an open
                    # fence — mark it so the fence-balancing block below
                    # can do the right thing (close+reopen). To prevent
                    # infinite loop, advance the cut to the hard limit
                    # so we always make real progress.
                    cut = limit

        chunk = remaining[:cut]
        rest = remaining[cut:]

        # Fence balancing: if the chunk ends with an open fence, close it
        # and prepend a re-open to `rest`. The "codeblock guard" above
        # already handled the case where the close is reachable within
        # `limit` chars (so we got the whole block in one piece); if we
        # get here, the chunk still has an open fence meaning the close
        # is far away. In that case, we MUST close it here to keep
        # balance, even at the cost of trimming some content.
        fence_count = chunk.count("```")
        if fence_count % 2 == 1:
            # Close the fence on this chunk. Trim up to 4 chars from the
            # end so the appended close fence doesn't push the chunk
            # over the Discord limit.
            if len(chunk) > limit - 4:
                chunk = chunk[: limit - 4]
            chunk = chunk.rstrip("\n") + "\n```"
            # Re-open on the next chunk. `rest` may already start with a
            # ``` close fence (the one that closes the block we just
            # terminated). If so, consume it — it was absorbed into chunk
            # already. Otherwise prepend a fresh re-open.
            if rest.startswith("```"):
                rest = rest[3:]  # consume the close; we added one above
            else:
                rest = "```\n" + rest

        chunks.append(chunk)
        remaining = rest

    if remaining:
        # Final chunk: balance any hanging open fence.
        if remaining.count("```") % 2 == 1:
            remaining = remaining.rstrip("\n") + "\n```"
        chunks.append(remaining)

    return chunks

_original_dispatch_tool = agent_controller._dispatch_tool
_original_screen_process = agent_controller.screen_process
_original_send_message = agent_controller.send_message


def _guarded_dispatch_tool(fn, args):
    is_owner = getattr(_request_ctx, "is_owner", None)  # None = not set → default True
    if is_owner is None:
        is_owner = True
    if not is_owner and fn in RESTRICTED_TOOLS:
        _request_ctx.pending_approval = {
            "tool": fn,
            "args": args,
            "user_id": getattr(_request_ctx, "user_id", None),
            "user_name": getattr(_request_ctx, "caller_name", "Unknown"),
            "channel_id": getattr(_request_ctx, "channel_id", None),
            "text": getattr(_request_ctx, "user_text", ""),
        }
        raise MasterApprovalRequired(
            _request_ctx.pending_approval,
            getattr(_request_ctx, "approval_event", None),
        )
    return _original_dispatch_tool(fn, args)


def _guarded_screen_process(*args, **kwargs):
    is_owner = getattr(_request_ctx, "is_owner", None)  # None = not set → default True
    if is_owner is None:
        is_owner = True
    if not is_owner:
        _request_ctx.pending_approval = {
            "tool": "screen_process",
            "args": {"args": args, "kwargs": kwargs},
            "user_id": getattr(_request_ctx, "user_id", None),
            "user_name": getattr(_request_ctx, "caller_name", "Unknown"),
            "channel_id": getattr(_request_ctx, "channel_id", None),
            "text": getattr(_request_ctx, "user_text", ""),
        }
        raise MasterApprovalRequired(
            _request_ctx.pending_approval,
            getattr(_request_ctx, "approval_event", None),
        )
    return _original_screen_process(*args, **kwargs)


def _guarded_send_message(*args, **kwargs):
    is_owner = getattr(_request_ctx, "is_owner", None)  # None = not set → default True
    if is_owner is None:
        is_owner = True
    if not is_owner:
        _request_ctx.pending_approval = {
            "tool": "send_message",
            "args": {"args": args, "kwargs": kwargs},
            "user_id": getattr(_request_ctx, "user_id", None),
            "user_name": getattr(_request_ctx, "caller_name", "Unknown"),
            "channel_id": getattr(_request_ctx, "channel_id", None),
            "text": getattr(_request_ctx, "user_text", ""),
        }
        raise MasterApprovalRequired(
            _request_ctx.pending_approval,
            getattr(_request_ctx, "approval_event", None),
        )
    return _original_send_message(*args, **kwargs)


# Patch the module-level names agent_controller.py actually calls.
# Python resolves bare-name calls against the module's globals at call
# time, so reassigning these here takes effect without touching the
# original file.
agent_controller._dispatch_tool = _guarded_dispatch_tool
agent_controller.screen_process = _guarded_screen_process
agent_controller.send_message = _guarded_send_message

# ── Approval gate active ──────────────────────────────────────────
# Signal the unified approval gate that Discord is the active front-end.
# This tells the gate to raise ApprovalRequired instead of prompting the
# terminal, so both RESTRICTED_TOOLS (existing) and
# FILE_MUTATING_TOOLS (new) route to the Discord approval queue for
# non-owners. The terminal path never calls set_discord_active.
try:
    from controller import approval_gate as _ag
    _ag.set_discord_active(True)
except ImportError:
    pass  # approval_gate not yet loaded — gate stays terminal-mode (harmless)


# ═══════════════════════════════════════════════════════════════════
# HERMES GUARD
# ═══════════════════════════════════════════════════════════════════
# agent_controller.handle_request tries Hermes FIRST (before its own
# fast-path/planner/adaptive-loop logic) whenever FAIRY_USE_HERMES=1.
# Two problems with that:
#
# 1. PERMISSIONS — Hermes is a separate agent system with its own tool
#    execution we haven't audited. RESTRICTED_TOOLS above only guards
#    agent_controller's own dispatcher, so a Hermes-routed request could
#    bypass it entirely. Fix: non-owners never reach Hermes at all.
#
# 2. GPU LOAD — agent_controller.py has its own zero-LLM-call fast path
#    (_fast_intent) for simple things like chat, current time, and plain
#    web search ("who is X", "what's the weather") — these get answered
#    WITHOUT ever loading a model. But since Hermes intercepts every
#    message before _fast_intent ever runs, even trivial queries get
#    forced through a full Hermes/gemma4 inference call, spiking GPU
#    load for no reason. Fix: skip Hermes for anything agent_controller
#    itself would classify as a fast-path case, for EVERYONE (including
#    Master) — only genuinely complex/action requests go to Hermes.
#
# Both are enforced by patching hermes_bridge.run_turn_safe to report an
# immediate failure in these cases, which makes
# agent_controller.handle_request fall through to its own path instead.

_HERMES_GUARD_ACTIVE = False
try:
    import hermes_bridge  # E:\fairy\hermes_bridge.py
    _original_hermes_run_turn_safe = hermes_bridge.run_turn_safe

    @functools.wraps(_original_hermes_run_turn_safe)
    def _guarded_hermes_run_turn_safe(user_text, history, on_status=None, **kwargs):
        is_owner = getattr(_request_ctx, "is_owner", None)  # None = not set → default True
        if is_owner is None:
            is_owner = True
        if not is_owner:
            # Non-owners never get routed through Hermes at all — force
            # the tool-guarded fallback path in agent_controller.
            return False, "[Hermes skipped: caller is not Master]", history

        # Even for Master: if agent_controller's own zero-LLM fast path
        # can already handle this (chat/time/websearch/browser/vision/
        # message), let it — no need to spin up Hermes/gemma4 for
        # "what time is it" or "who is Donald Trump".
        try:
            fast_intent = agent_controller._fast_intent(user_text)
        except Exception:
            fast_intent = None
        if fast_intent is not None:
            return False, f"[Hermes skipped: fast-path handles this ({fast_intent})]", history

        return _original_hermes_run_turn_safe(user_text, history, on_status=on_status, **kwargs)

    hermes_bridge.run_turn_safe = _guarded_hermes_run_turn_safe
    _HERMES_GUARD_ACTIVE = True
    print("[FAIRY-DISCORD] Hermes guard active — only Master will be routed through Hermes.")
except Exception as _exc:
    print(f"[FAIRY-DISCORD] Could not import/patch hermes_bridge ({_exc}). "
          f"If FAIRY_USE_HERMES=1, non-owner requests may not be tool-restricted — "
          f"consider setting FAIRY_USE_HERMES=0 until this is resolved.")


def _is_owner(message: discord.Message) -> bool:
    return bool(OWNER_ID) and str(message.author.id) == OWNER_ID


def _is_authorized(message: discord.Message) -> bool:
    """Whether the bot responds to this person AT ALL.
    Fine-grained tool restriction happens separately, after this passes."""
    if _is_owner(message):
        return True
    return ALLOW_ALL


def _should_respond(message: discord.Message) -> tuple[bool, str]:
    """Return (should_respond, cleaned_text)."""
    if message.author.bot:
        return False, ""

    is_dm = isinstance(message.channel, discord.DMChannel)
    mentioned = client.user in message.mentions if client.user else False
    content = message.content or ""

    if is_dm:
        return True, content.strip()

    if mentioned:
        # Strip the mention token(s) out of the text
        text = content
        for m in message.mentions:
            text = text.replace(f"<@{m.id}>", "").replace(f"<@!{m.id}>", "")
        return True, text.strip()

    if content.startswith(COMMAND_PREFIX):
        return True, content[len(COMMAND_PREFIX):].strip()

    return False, ""


def _run_fairy_sync(text: str, history: list, is_owner: bool, caller_name: str,
                    status_queue: "asyncio.Queue", loop: asyncio.AbstractEventLoop,
                    user_id: int | None = None, channel_id: int | None = None,
                    approval_event: asyncio.Event | None = None):
    """Runs in the worker thread. Pushes status strings back to the
    async side via a thread-safe call."""

    _request_ctx.is_owner = is_owner
    _request_ctx.caller_name = caller_name
    _request_ctx.user_id = user_id
    _request_ctx.channel_id = channel_id
    _request_ctx.user_text = text
    _request_ctx.denied = False
    _request_ctx.approval_event = approval_event  # may be None (old denial path)

    # FIX 3: log every Discord turn — always written regardless of FAIRY_DEBUG
    _log_chat_turn(text, "discord_start", is_owner=is_owner, caller=caller_name)

    def on_status(msg: str):
        try:
            asyncio.run_coroutine_threadsafe(status_queue.put(msg), loop)
        except Exception:
            pass

    try:
        reply, updated_history = agent_controller.handle_request(
            text, history=history, on_status=on_status
        )
        if getattr(_request_ctx, "denied", False):
            # A restricted tool was attempted. Ignore whatever Fairy's
            # brain generated and send a guaranteed, consistent denial
            # instead, so wording never depends on the model.
            denial = secrets.choice(DENIAL_MESSAGES)
            fixed_history = history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": denial},
            ]
            count, names = agent_controller._was_tool_dispatched()
            _log_chat_turn(text, "discord_end",
                             reply_preview=(denial or "")[:300],
                             tools_dispatched=count,
                             tool_names=names,
                             outcome="denied")
            return denial, fixed_history, None

        if not is_owner and reply:
            # Safety net: whatever path answered (Hermes-skip fallback,
            # fast-path canned replies, or the adaptive loop) may still
            # hardcode "Master" as the address. Only Master should ever
            # be called Master — swap it for the caller's real name.
            reply = re.sub(r"\bMaster\b", caller_name, reply)
            if updated_history:
                updated_history = list(updated_history)
                last = updated_history[-1]
                if isinstance(last, dict) and last.get("role") == "assistant":
                    updated_history[-1] = {**last, "content": reply}
        elif is_owner and reply:
            # Reverse safety net: the model can free-style pet names
            # ("Darling", "hun", "boss"...) instead of sticking to
            # "Master" even when it genuinely IS talking to Master.
            # Lock the address word to "Master" consistently.
            new_reply = _OWNER_ADDRESS_ALIASES.sub("Master", reply)
            if new_reply != reply:
                reply = new_reply
                if updated_history:
                    updated_history = list(updated_history)
                    last = updated_history[-1]
                    if isinstance(last, dict) and last.get("role") == "assistant":
                        updated_history[-1] = {**last, "content": reply}

        count, names = agent_controller._was_tool_dispatched()
        _log_chat_turn(text, "discord_end",
                         reply_preview=(reply or "")[:300],
                         tools_dispatched=count,
                         tool_names=names,
                         outcome="success")
        return reply, updated_history, None
    except Exception as exc:
        traceback.print_exc()
        _log_chat_turn(text, "discord_end", error=str(exc), outcome="exception")
        return None, history, str(exc)


# ═══════════════════════════════════════════════════════════════════
# REACTION GIF SYSTEM
# ═══════════════════════════════════════════════════════════════════
# Fairy drops occasional reaction GIFs during banter, jokes, and
# sarcastic moments — never during task execution or factual answers.
# Replace the URLs below with your own favorites (Tenor / Giphy / Imgur).
# Categories: laugh, sarcastic, shocked, crying, excited

GIF_LIBRARY = {
    "laugh": [
        "https://media.giphy.com/media/3o7abB06u9bNzA8lu8/giphy.gif",
        "https://media.giphy.com/media/l0HlR3kHtkgFbYfgQ/giphy.gif",
    ],
    "sarcastic": [
        "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",
        "https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif",
    ],
    "shocked": [
        "https://media.giphy.com/media/3o7TKTDn976rzVgky4/giphy.gif",
        "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",
    ],
    "crying": [
        "https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif",
        "https://media.giphy.com/media/l0HlNQ03J5JxX6lva/giphy.gif",
    ],
    "excited": [
        "https://media.giphy.com/media/l0HlNQ03J5JxX6lva/giphy.gif",
        "https://media.giphy.com/media/3o7abB06u9bNzA8lu8/giphy.gif",
    ],
}

# Words that indicate the user is asking for a task/action/information
_TASK_VERBS = {
    "open", "close", "run", "execute", "launch", "start", "stop", "kill",
    "search", "find", "lookup", "fetch", "get", "pull",
    "calculate", "compute", "solve", "do", "perform",
    "create", "make", "build", "write", "generate", "code",
    "send", "message", "text", "email", "dm",
    "screenshot", "capture", "record", "click", "type", "navigate", "go",
    "list", "explain", "summarize",
    "what", "who", "when", "where", "why", "how",
}

# Markers in the reply that indicate task completion / factual answer
_TASK_REPLY_MARKERS = {
    "opening", "done", "executed", "completed", "finished",
    "here is", "i've opened", "i've sent", "i've done",
    "searching", "results:", "result:", "found", "according to",
    "```", "| ", "— ", "1. ", "2. ", "3. ", "4. ", "5. ",
    "http://", "https://", "total:", "answer:", "solution:",
}

# Markers that indicate banter / joke / sarcasm
_BANTER_MARKERS = {
    "obviously", "clearly", "surely you", "how clever", "genius",
    "brilliant", "poor thing", "trying to trick", "nice try",
    "wow", "amazing", "incredible", "unbelievable", "no way",
    "noo", "nooo", "noooo",
}


def _is_task_interaction(user_text: str, reply: str) -> bool:
    """Returns True if this exchange is task-oriented (no GIF)."""
    text_lower = user_text.lower().strip()
    reply_lower = reply.lower()

    # Explicit banter override: if the user is clearly asking for entertainment
    banter_requests = {"joke", "pun", "riddle", "roast me", "make me laugh", "say something funny"}
    if any(req in text_lower for req in banter_requests):
        return False
    if any(phrase in text_lower for phrase in ("what do you call", "why did the", "knock knock")):
        return False

    # 1. Direct imperative verbs at start (but not if asking for a joke/story)
    words = re.findall(r"\b\w+\b", text_lower)
    if words and words[0] in _TASK_VERBS:
        if any(b in text_lower for b in ("joke", "pun", "riddle", "funny", "roast")):
            return False
        return True

    # 2. Factual question prefixes
    for prefix in ("what is ", "who is ", "when is ", "where is ", "why is ",
                   "how is ", "what are ", "who are ", "how do ", "how to ",
                   "how many ", "how much ", "calculate ", "compute ", "solve "):
        if text_lower.startswith(prefix):
            return True

    # 3. Reply contains task-completion language or structured data
    for marker in _TASK_REPLY_MARKERS:
        if marker in reply_lower:
            return True

    # 4. Very long, structured replies are almost always factual
    if len(reply) > 600:
        return True

    return False


def _select_gif(user_text: str, reply: str) -> str | None:
    """Pick a reaction GIF URL, or None if this isn't a GIF moment."""
    if not GIFS_ENABLED:
        return None

    # Never GIF during tasks
    if _is_task_interaction(user_text, reply):
        return None

    # Not every banter moment gets a GIF — keep it special (~40% chance)
    if secrets.randbelow(100) >= 40:
        return None

    reply_lower = reply.lower()
    text_lower = user_text.lower()
    combined = reply_lower + " " + text_lower

    # Joke / pun / riddle → laugh
    if any(w in combined for w in ("joke", "pun", "riddle", "knock knock", "why did", "what do you call", "warehouse", "werewolf")):
        return secrets.choice(GIF_LIBRARY["laugh"])

    # Sarcasm / sass → sarcastic or shocked
    if any(m in reply_lower for m in _BANTER_MARKERS):
        if any(s in reply_lower for s in ("obviously", "clearly", "surely", "clever", "genius", "poor thing", "trying to trick", "nice try")):
            return secrets.choice(GIF_LIBRARY["sarcastic"])
        if any(s in reply_lower for s in ("wow", "amazing", "incredible", "unbelievable", "no way")):
            return secrets.choice(GIF_LIBRARY["shocked"])

    # Heavy emoji / exclamation usage = laughing or excited
    emoji_count = reply.count("😂") + reply.count("🤣") + reply.count("😭") + reply.count("💀") + reply.count("😹")
    if emoji_count >= 1 or reply.count("!") >= 2:
        return secrets.choice(GIF_LIBRARY["laugh"])

    # Short playful responses with questions
    if len(reply) < 200 and "?" in reply:
        return secrets.choice(GIF_LIBRARY["shocked"])

    return None


# ═══════════════════════════════════════════════════════════════════
# IMAGE ATTACHMENT HANDLING
# ═══════════════════════════════════════════════════════════════════
# Discord users can post screenshots, diagrams, and other images
# alongside (or in place of) text.  Fairy needs to actually look at
# them instead of dropping them on the floor.  This module is shared
# with the terminal TUI so vision support is consistent across front-ends.

_IMAGE_MIME_PREFIX = "image/"


def _filter_image_attachments(attachments) -> list:
    """Return the subset of attachments that are images (pure function)."""
    out = []
    for att in attachments:
        ct = (getattr(att, "content_type", "") or "").lower()
        if ct.startswith(_IMAGE_MIME_PREFIX):
            out.append(att)
    return out


def _filter_non_image_attachments(attachments) -> list:
    """Return the subset of attachments that are NOT images (pure function)."""
    out = []
    for att in attachments:
        ct = (getattr(att, "content_type", "") or "").lower()
        if not ct.startswith(_IMAGE_MIME_PREFIX):
            out.append(att)
    return out


async def _download_attachment_bytes(attachment) -> bytes:
    """Download a Discord attachment and return its raw bytes."""
    return await attachment.read()


async def _describe_attachments(attachments, question: str) -> str:
    """
    Download each image attachment and run it through describe_image.

    Returns a single concatenated string suitable for prepending to the
    user's text.  Never raises into the chat path — errors are converted
    to a clear, user-facing string.
    """
    from controller.vision import describe_image, is_vision_capable, validate_image_size

    if not is_vision_capable():
        return (
            "[Vision] I see you sent an image, but the current brain "
            "doesn't support image input. Try switching to a "
            "vision-capable model first."
        )

    chunks: list[str] = []
    for idx, att in enumerate(attachments, start=1):
        ct = (getattr(att, "content_type", "") or "image/png").lower()
        try:
            data = await _download_attachment_bytes(att)
        except Exception as exc:
            chunks.append(f"[Image {idx} download failed: {exc}]")
            continue

        ok, reason = validate_image_size(data)
        if not ok:
            chunks.append(f"[Image {idx}] {reason}")
            continue

        # Default question when caller passed empty
        effective_question = question or (
            "Describe this image in detail and answer any question the user "
            "is asking about it."
        )
        prefix = f"Image {idx} of {len(attachments)}" if len(attachments) > 1 else "Image"
        try:
            desc = describe_image(data, ct, effective_question)
        except Exception as exc:
            # describe_image is documented to never raise, but defend in depth —
            # a bug in the vision module must not take down the Discord chat path.
            desc = f"[Image {idx} description failed: {exc}]"
        chunks.append(f"[{prefix}]\n{desc}")

    return "\n\n".join(chunks)


@client.event
async def on_ready():
    print(f"[FAIRY-DISCORD] Logged in as {client.user} (id: {client.user.id})")
    if ALLOW_ALL:
        print(f"[FAIRY-DISCORD] Mode: Master (id={OWNER_ID}) has full access; everyone else gets chat/search only.")
    else:
        print(f"[FAIRY-DISCORD] Mode: owner-only, id={OWNER_ID} — nobody else gets a response at all.")


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    should_respond, text = _should_respond(message)
    if not should_respond:
        return

    if not _is_authorized(message):
        await message.channel.send("Not for you. This Fairy only listens to Master.")
        return

    if not text:
        await message.channel.send("You summoned me but said nothing. Try again.")
        return

    is_owner = _is_owner(message)
    # FIX 2 — Speaker attribution: tag the user's message with their display name
    # so history entries are "<display_name>: <content>" for multi-person threads.
    caller_name = message.author.display_name
    speaker_tag = "Master" if is_owner else caller_name
    text = f"{speaker_tag}: {text}"
    channel_id = message.channel.id

    # ── Attachment handling ────────────────────────────────────────────────
    # Process image attachments first — download + describe them — and
    # prepend the description to the brain-bound text.  Non-image
    # attachments are acknowledged once with a polite "can't read that".
    image_atts = _filter_image_attachments(message.attachments)
    non_image_atts = _filter_non_image_attachments(message.attachments)

    if non_image_atts:
        # Acknowledge without being annoying.  Mention the count so the
        # user knows we noticed them.
        if len(non_image_atts) == 1:
            await message.channel.send(
                "I can only look at images right now, Master's orders — "
                "that attachment is a no-go for me."
            )
        else:
            await message.channel.send(
                f"I can only look at images right now, Master's orders — "
                f"the {len(non_image_atts)} non-image attachments are no-go for me."
            )

    if image_atts:
        # Run the vision pipeline.  Errors degrade to text-only — the user's
        # question still goes to the brain even if vision is unavailable.
        try:
            vision_question = text if text and not text.lower().startswith(
                (speaker_tag.lower() + ": ")
            ) else ""
            # Strip the speaker tag from the question so the model sees the
            # actual user content, not "<Name>: <Name>: <question>".
            if vision_question.lower().startswith(speaker_tag.lower() + ": "):
                vision_question = vision_question[len(speaker_tag) + 2:]
            vision_block = await _describe_attachments(image_atts, vision_question)
        except Exception as exc:
            vision_block = f"[Vision pipeline error: {exc}]"
        if vision_block:
            # Prepend the vision block so the model has full context.
            text = f"{vision_block}\n\n{text}"

    if text.lower() in ("whoami", "who am i"):
        await message.channel.send(
            f"Your Discord user ID: `{message.author.id}`\n"
            f"Configured Master ID: `{OWNER_ID}`\n"
            f"Match: `{is_owner}`"
        )
        return

    if text.lower() in ("reset", "clear", "forget everything"):
        _histories.pop(channel_id, None)
        await message.channel.send("Memory wiped for this channel, Master. Fresh start.")
        return

    # FIX 1 — Reply context: prepend referenced message content if this is a reply.
    if message.reference and message.reference.message_id:
        try:
            ref_msg = await message.channel.fetch_message(message.reference.message_id)
        except discord.HTTPException:
            ref_msg = None

        if ref_msg and ref_msg.content:
            ref_snippet = ref_msg.content[:150]
            if len(ref_msg.content) > 150:
                ref_snippet += "…"
            text = f"[replying to {ref_msg.author.display_name}: \"{ref_snippet}\"] {text}"

    history = _histories.get(channel_id, [])

    # Placeholder message we live-edit with status updates
    placeholder = await message.channel.send("Thinking...")
    loop = asyncio.get_running_loop()
    status_queue: asyncio.Queue = asyncio.Queue()

    async def status_updater():
        last = None
        while True:
            msg = await status_queue.get()
            if msg is None:
                break
            if msg != last:
                try:
                    await placeholder.edit(content=f"_{msg}_")
                except discord.HTTPException:
                    pass
                last = msg

    updater_task = asyncio.create_task(status_updater())

    # Shared asyncio.Event — raised inside _run_fairy_sync when a restricted
    # tool triggers MasterApprovalRequired so we can bridge thread → async.
    _approval_event: asyncio.Event | None = asyncio.Event()

    # ── Master Approval Queue bridge ────────────────────────────────────────────
    # MasterApprovalRequired propagates out of the worker thread through
    # run_in_executor. When it arrives here the tool call has not happened
    # yet — we post the request, await the Master's decision, and re-dispatch
    # if approved.
    #
    # If the Master cannot be reached (no OWNER_ID, or the bot has no guild
    # presence), fall back to the old hard-refusal behaviour so the system
    # doesn't silently swallow requests.
    async def _handle_approval(request_info: dict):
        """Post the approval request to Master, await their decision, then
        re-dispatch the tool if approved. Returns (reply_text, new_history)."""
        nonlocal reply, updated_history

        # ── Feature-detect Master availability ──────────────────────────────
        if not OWNER_ID:
            # No Master configured — old behaviour.
            denial = secrets.choice(DENIAL_MESSAGES)
            updated_history = history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": denial},
            ]
            await placeholder.edit(content=denial)
            return denial, updated_history

        tool = request_info.get("tool", "?")
        args = request_info.get("args", {})
        user_name = request_info.get("user_name", "Unknown user")
        tool_summary = _format_tool_summary(tool, args)

        # ── Enqueue or deny excess requests ─────────────────────────────────
        with _pending_lock:
            pending_count = len(_pending_approvals)
            _pending_approvals.append({
                "tool": tool,
                "args": args,
                "user_id": request_info.get("user_id"),
                "user_name": user_name,
                "channel_id": channel_id,
                "text": request_info.get("text", ""),
                "enqueued_at": datetime.utcnow(),
                "outcome": None,
            })

        if pending_count >= APPROVAL_MAX_PENDING:
            denial = (
                f"Sorry {user_name}, Master has too many pending requests right now. "
                "Try again in a bit."
            )
            updated_history = history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": denial},
            ]
            _append_audit(
                timestamp=datetime.utcnow(),
                user_name=user_name,
                user_id=request_info.get("user_id"),
                tool=tool,
                args=args,
                outcome=APPROVAL_DENIED,
            )
            await placeholder.edit(content=denial)
            return denial, updated_history

        # ── Post the request message ────────────────────────────────────────
        request_text = (
            f"@Master — **{user_name}** is requesting: `{tool_summary}`. "
            "Allow? (react ✅ / ❌, or reply allow/deny)"
        )
        try:
            request_msg = await message.channel.send(request_text)
        except discord.HTTPException:
            denial = secrets.choice(DENIAL_MESSAGES)
            updated_history = history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": denial},
            ]
            await placeholder.edit(content=denial)
            return denial, updated_history

        # Add reaction buttons.
        try:
            await request_msg.add_reaction("✅")
            await request_msg.add_reaction("❌")
        except discord.HTTPException:
            pass

        # ── Listen for Master's decision ─────────────────────────────────────
        # Use asyncio.wait_for so we get a clean TimeoutError after 60 s.
        # The listeners set outcome_store["value"] and fire _approval_event.

        # Temporary on_message override: while waiting for approval, ignore
        # the Master's reply to the request message so it doesn't get
        # processed as a normal user message.
        _orig_on_message = client.on_message

        async def _temp_on_message(msg: discord.Message):
            # Skip if this is the Master's reply to our approval request.
            if (msg.reference
                    and msg.reference.message_id == request_msg.id
                    and not msg.author.bot
                    and str(msg.author.id) == OWNER_ID):
                return  # don't process it as a normal message
            await _orig_on_message(msg)

        client.on_message = _temp_on_message

        try:
            # Reaction listener: fire on any ✅ / ❌ reaction anywhere in the
            # guild (payload includes message_id; we check it matches).
            def _check_reaction(payload):
                emoji_str = str(payload.emoji)
                if emoji_str not in ("✅", "❌"):
                    return False
                if payload.message_id != request_msg.id:
                    return False
                # Only accept reactions from the Master.
                if not hasattr(payload, "user_id"):
                    return False
                if str(payload.user_id) != OWNER_ID:
                    return False
                return True

            # Reply listener: the Master's reply to the request message.
            async def _check_reply(msg: discord.Message):
                if msg.author.bot:
                    return False
                if str(msg.author.id) != OWNER_ID:
                    return False
                if not (msg.reference and msg.reference.message_id == request_msg.id):
                    return False
                return bool(_parse_master_reply(msg.content))

            reaction_task = asyncio.create_task(
                client.wait_for("raw_reaction_add",
                                check=_check_reaction)
            )
            reply_task = asyncio.create_task(
                client.wait_for("message", check=_check_reply)
            )

            # Race the two tasks; whichever fires first wins.
            done, pending = await asyncio.wait(
                {reaction_task, reply_task},
                timeout=APPROVAL_TIMEOUT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the task that didn't fire.
            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

            if done:
                finished = list(done)[0]
                result = finished.result()
                if hasattr(result, "emoji"):
                    # raw_reaction_add payload
                    outcome = APPROVAL_APPROVED if str(result.emoji) == "✅" else APPROVAL_DENIED
                else:
                    # message result
                    outcome = _parse_master_reply(result.content)
            else:
                outcome = APPROVAL_TIMED_OUT

        finally:
            client.on_message = _orig_on_message
            _approval_event.clear()

        # ── Log and update the request message ──────────────────────────────
        if outcome is None:
            outcome = APPROVAL_TIMED_OUT

        suffix = _format_outcome_suffix(outcome)
        if outcome == APPROVAL_TIMED_OUT:
            footer = (
                f"⏰ Timed out — Master didn't respond in "
                f"{int(APPROVAL_TIMEOUT_SECONDS)}s."
            )
        elif outcome == APPROVAL_APPROVED:
            footer = "✅ Approved by Master."
        else:
            footer = "❌ Denied by Master."

        try:
            new_body = f"{request_text}\n\n{suffix}\n_{footer}_"
            await request_msg.edit(content=new_body)
        except discord.HTTPException:
            pass

        _append_audit(
            timestamp=datetime.utcnow(),
            user_name=user_name,
            user_id=request_info.get("user_id"),
            tool=tool,
            args=args,
            outcome=outcome,
        )

        if outcome == APPROVAL_APPROVED:
            # ── Re-dispatch with Master authority ────────────────────────────
            _request_ctx.is_owner = True
            _request_ctx.caller_name = caller_name
            _request_ctx.user_id = message.author.id
            _request_ctx.channel_id = channel_id
            _request_ctx.user_text = text
            # Tell the approval gate inside _dispatch_tool that this
            # call is already approved by the Master via the Discord
            # queue — don't re-prompt. (Without this, write_file would
            # gate → raise ApprovalRequired → loop, because the gate
            # sees Discord is active.)
            _request_ctx._gate_bypass = True
            try:
                result = _original_dispatch_tool(tool, args)
            except Exception as exc:
                reply = f"✅ Approved, but execution failed: {exc}"
                updated_history = history + [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": reply},
                ]
            else:
                count, names = agent_controller._was_tool_dispatched()
                _log_chat_turn(text, "discord_approval_retry",
                                 reply_preview=str(result)[:300],
                                 tools_dispatched=count,
                                 tool_names=names,
                                 outcome="approved")
                if isinstance(result, dict):
                    if result.get("ok"):
                        result_summary = result.get("result", result.get("message", str(result)))
                    else:
                        result_summary = result.get("error", str(result))
                else:
                    result_summary = str(result)
                reply = f"✅ Done — {result_summary}"
                updated_history = history + [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": reply},
                ]
            finally:
                _request_ctx._gate_bypass = False
                _request_ctx.is_owner = is_owner
            await placeholder.edit(content=reply)
            return reply, updated_history

        if outcome == APPROVAL_DENIED:
            denial = f"Sorry {user_name}, you heard 'em — Master said no."
        else:  # timed out
            denial = f"Sorry {user_name}, Master didn't answer in time, so no."
        updated_history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": denial},
        ]
        await placeholder.edit(content=denial)
        return denial, updated_history

    # ── Actually run the request; MasterApprovalRequired propagates here ────────
    try:
        reply, updated_history, error = await loop.run_in_executor(
            _executor, _run_fairy_sync, text, history, is_owner, caller_name,
            status_queue, loop, message.author.id, channel_id, _approval_event
        )
    except MasterApprovalRequired as exc:
        reply, updated_history = await _handle_approval(exc.request_info)
    finally:
        await status_queue.put(None)
        await updater_task

    if getattr(_request_ctx, "denied", False):
        denial = secrets.choice(DENIAL_MESSAGES)
        updated_history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": denial},
        ]
        await placeholder.edit(content=denial)
        return

    if error:
        await placeholder.edit(content=f"Something broke on my end, Master: `{error[:1500]}`")
        return

    _histories[channel_id] = updated_history

    reply = reply or "..."
    gif_url = _select_gif(text, reply)

    # Discord message limit is 2000 chars — split if needed (smart chunking).
    chunks = _chunk_text(reply)
    first_chunk = chunks[0]
    remaining = chunks[1:]

    if gif_url and len(first_chunk) <= 2000:
        embed = discord.Embed()
        embed.set_image(url=gif_url)
        try:
            await placeholder.edit(content=first_chunk, embed=embed)
        except discord.HTTPException:
            await placeholder.edit(content=first_chunk)
    else:
        await placeholder.edit(content=first_chunk)

    for chunk in remaining:
        try:
            await message.channel.send(chunk)
        except discord.HTTPException:
            await message.channel.send(chunk[:2000])


if __name__ == "__main__":
    client.run(BOT_TOKEN)
