"""
Long-term memory v0 — curated, durable facts with a HARD injection budget.

Storage: a single JSON file. Each fact is structured; proposals are gated
behind user confirmation (no silent learning). Stale facts (30+ days unused
with a single use) are excluded from injection but never silently deleted.

The injection budget is the critical safety rule. A previous feature died by
injecting a 30k-char dump into replies; we never repeat that. Selection
priority: ``preference`` first, then highest ``use_count``, then most recent
``last_used_at``. Overflow facts are silently dropped (and logged).

In-process cache: a tiny in-memory dict of the loaded facts list so per-turn
prompt construction does not re-read the file.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────
# Named module constants — the safety contract.
# These are the maximum dimensions of any block we will inject.
# ─────────────────────────────────────────────────────────────────────
INJECTION_BUDGET_MAX_CHARS: int = 2000  # hard cap on injected block size
INJECTION_BUDGET_MAX_FACTS: int = 25    # hard cap on fact count in block

# Staleness: a fact is "stale" if it has not been used in N days and has only
# been used once. Stale facts are excluded from injection (not deleted here —
# a later consolidation slice owns forgetting).
_STALENESS_DAYS: int = 30

# Cap on proposed facts per turn — never interrogate the user with a wall
# of options. 2 is intentional.
_PROPOSALS_PER_TURN_MAX: int = 2

# Valid categories. Unknown values are clamped to "fact_about_master".
VALID_CATEGORIES = {
    "preference",
    "fact_about_master",
    "fact_about_fairy",
    "learned_from_research",
}

_log = logging.getLogger("fairy.long_term_memory")

_BASE = Path(__file__).resolve().parent.parent
_MEM_PATH = _BASE / "memory" / "long_term_facts.json"


# ─────────────────────────────────────────────────────────────────────
# Pattern-based extraction. No LLM call — fast and dependency-free.
# User can always bypass extraction via the explicit "remember:" prefix.
# ─────────────────────────────────────────────────────────────────────
# Capture-group 1 = the candidate fact body.
_EXTRACT_PATTERNS = [
    re.compile(r"^remember:\s*(.+?)\s*\.?\s*$", re.IGNORECASE),
    re.compile(
        r"\bI\s+(?:prefer|only|always|never|hate|like|use)\s+([^.!?\n]{3,200})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy\s+(?:name|pronoun|language|timezone)\s+(?:is|called)\s+([^.!?\n]{2,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcall\s+me\s+([^.!?\n]{2,60})",
        re.IGNORECASE,
    ),
]


def extract_proposals(user_text: str) -> List[str]:
    """Lightweight pattern-based fact extraction. Returns up to 2 candidates.

    No LLM call. Honest about its limits: only catches explicit, common
    patterns. The user can also invoke ``remember: ...`` for an
    unambiguous proposal.
    """
    if not user_text or not isinstance(user_text, str):
        return []
    text = user_text.strip()
    if not text:
        return []

    candidates: List[str] = []
    seen_lower: set[str] = set()

    # Explicit "remember: <fact>" takes priority — extract it, strip it from
    # further processing so the I-prefer/etc patterns don't fire on the same
    # sentence and produce overlapping partial matches.
    _REMEMBER_PAT = re.compile(r"^remember:\s*(.+?)\s*\.?\s*$", re.IGNORECASE)
    m_remember = _REMEMBER_PAT.match(text)
    if m_remember:
        body = m_remember.group(1).strip().strip("'\"").rstrip(".!?,")
        if 4 <= len(body) <= 200:
            candidates.append(body)
            seen_lower.add(body.lower())
        # Strip the matched prefix so the rest of the text doesn't re-fire
        # on the same "I prefer..." phrase.
        rest = text[m_remember.end():].lstrip()
        if rest:
            text = rest

    # General patterns on whatever is left.
    def _is_redundant(candidate: str) -> bool:
        """True if the candidate is contained in (or contains) a higher-ranked
        one. Avoids offering "use free models" when "I prefer use free models"
        is already queued."""
        cl = candidate.lower()
        for existing in seen_lower:
            if cl in existing or existing in cl:
                return True
        return False

    for pat in _EXTRACT_PATTERNS[1:]:   # skip the remember pattern (index 0)
        for m in pat.finditer(text):
            body = (m.group(1) or "").strip().strip("'\"").rstrip(".!?,")
            if 4 <= len(body) <= 200:
                key = body.lower()
                if key not in seen_lower and not _is_redundant(body):
                    seen_lower.add(key)
                    candidates.append(body)
        if len(candidates) >= _PROPOSALS_PER_TURN_MAX:
            break

    return candidates[:_PROPOSALS_PER_TURN_MAX]


# ─────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────
class LongTermMemoryStore:
    """Thread-safe persistent store of curated long-term facts."""

    def __init__(self, path: Optional[Path] = None):
        self.path: Path = path or _MEM_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # In-process cache to avoid re-reading the file per LLM turn.
        # We invalidate it on every mutation. Reads use it to skip disk I/O.
        self._cache: Optional[Dict[str, Any]] = None
        # Eagerly load; corrupt/missing/oversized -> log + start fresh.
        self._data: Dict[str, Any] = self._load()
        self._cache = dict(self._data)

    # ── I/O ─────────────────────────────────────────────────────
    def _load(self) -> Dict[str, Any]:
        """Read and validate the file. Never crash on bad data."""
        if not self.path.exists():
            return {"facts": [], "pending": []}
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            _log.warning("long_term_memory: cannot read %s (%s) — starting fresh", self.path, exc)
            return {"facts": [], "pending": []}
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            _log.warning(
                "long_term_memory: corrupt JSON in %s (%s) — starting fresh",
                self.path,
                exc,
            )
            return {"facts": [], "pending": []}
        if not isinstance(data, dict):
            _log.warning("long_term_memory: top-level not a dict — starting fresh")
            return {"facts": [], "pending": []}
        data.setdefault("facts", [])
        data.setdefault("pending", [])
        # Normalize types — defensive against hand-edited files.
        if not isinstance(data["facts"], list):
            data["facts"] = []
        if not isinstance(data["pending"], list):
            data["pending"] = []
        return data

    def _save(self) -> None:
        """Atomic write via temp file. Caller must hold self._lock."""
        self.data["last_updated"] = datetime.now().isoformat()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except OSError as exc:
            _log.error("long_term_memory: failed to write %s (%s)", self.path, exc)

    # ── accessors ──────────────────────────────────────────────
    @property
    def data(self) -> Dict[str, Any]:
        # Always return a reference to the live dict so mutations land
        # in the same object both _save() and selectors read.
        return self._data

    def _invalidate_cache(self) -> None:
        self._cache = None

    # ── staleness ──────────────────────────────────────────────
    @staticmethod
    def _is_stale(fact: Dict[str, Any]) -> bool:
        if fact.get("use_count", 0) > 1:
            return False
        last = fact.get("last_used_at")
        if not last:
            return False
        try:
            ts = datetime.fromisoformat(last)
        except (TypeError, ValueError):
            return False
        return (datetime.now() - ts) > timedelta(days=_STALENESS_DAYS)

    # ── selection (the budget) ────────────────────────────────
    def _select_facts(self) -> List[Dict[str, Any]]:
        """Pick the facts to inject, ordered by priority.

        Priority: ``preference`` first, then ``use_count`` desc, then
        ``last_used_at`` desc. Excludes stale facts.
        """
        facts = [f for f in self.data.get("facts", []) if not self._is_stale(f)]

        # Use functools.cmp_to_key with a comparator that handles all three
        # priority fields in one pass. Easier to read than a chain of sorts
        # with negate-and-flip tricks.
        from functools import cmp_to_key

        def _cmp(a: Dict[str, Any], b: Dict[str, Any]) -> int:
            # 1) preference first
            a_pref = 1 if a.get("category") == "preference" else 0
            b_pref = 1 if b.get("category") == "preference" else 0
            if a_pref != b_pref:
                # Want preference first, so the preferred item should be "less"
                # (sorted earlier). Higher pref_score wins → return -1.
                return b_pref - a_pref
            # 2) higher use_count first
            a_use = int(a.get("use_count", 0))
            b_use = int(b.get("use_count", 0))
            if a_use != b_use:
                return b_use - a_use  # want bigger first → b - a
            # 3) more recent last_used_at first
            a_ts = a.get("last_used_at") or ""
            b_ts = b.get("last_used_at") or ""
            if a_ts != b_ts:
                # ISO timestamps sort lexicographically = chronologically.
                # Want larger (more recent) first → b - a.
                return -1 if a_ts < b_ts else 1
            return 0  # fully tied

        facts.sort(key=cmp_to_key(_cmp))
        return facts

    def get_injection_block(self) -> str:
        """Build the budget-capped block of facts for prompt injection.

        Updates ``last_used_at`` and ``use_count`` for the selected facts.
        Returns "" if no facts qualify.
        """
        with self._lock:
            selected: List[Dict[str, Any]] = []
            char_budget = INJECTION_BUDGET_MAX_CHARS
            count_budget = INJECTION_BUDGET_MAX_FACTS
            overflow = 0
            for f in self._select_facts():
                if count_budget <= 0 or char_budget <= 0:
                    overflow += 1
                    continue
                text = (f.get("text") or "").strip()
                if not text:
                    continue
                # One fact line = "• <text>" plus the newline.
                line = f"- {text}"
                # Conservative per-line overhead for newline + bullet.
                cost = len(line) + 1
                if cost > char_budget:
                    # The single line alone exceeds the remaining budget.
                    overflow += 1
                    continue
                selected.append(f)
                char_budget -= cost
                count_budget -= 1

            if overflow:
                _log.info(
                    "long_term_memory: budget hit — %d fact(s) dropped from injection",
                    overflow,
                )

            if not selected:
                return ""

            # Mark used and persist.
            now_iso = datetime.now().isoformat()
            ids = {f.get("id") for f in selected}
            for f in self.data.get("facts", []):
                if f.get("id") in ids:
                    f["last_used_at"] = now_iso
                    f["use_count"] = int(f.get("use_count", 0)) + 1
            self._save()
            self._invalidate_cache()

            return "\n".join(f"- {(f.get('text') or '').strip()}" for f in selected)

    # ── facts CRUD ────────────────────────────────────────────
    def _coerce_category(self, category: Optional[str]) -> str:
        if not category:
            return "fact_about_master"
        cat = str(category).strip().lower()
        return cat if cat in VALID_CATEGORIES else "fact_about_master"

    def add_fact(
        self,
        text: str,
        category: Optional[str] = None,
        source: str = "conversation",
    ) -> str:
        """Create and persist a new fact. Returns the new fact's id."""
        if not text or not isinstance(text, str):
            raise ValueError("fact text must be a non-empty string")
        body = text.strip()
        if not body:
            raise ValueError("fact text must be a non-empty string")
        cat = self._coerce_category(category)
        now_iso = datetime.now().isoformat()
        fact = {
            "id": uuid.uuid4().hex,
            "text": body,
            "category": cat,
            "created_at": now_iso,
            "last_used_at": now_iso,
            "use_count": 1,
            "source": source,
        }
        with self._lock:
            self.data.setdefault("facts", []).append(fact)
            self._save()
            self._invalidate_cache()
        return fact["id"]

    def delete_fact(self, fact_id: str) -> bool:
        """Remove a fact by id. Returns True if removed."""
        with self._lock:
            facts = self.data.setdefault("facts", [])
            before = len(facts)
            self.data["facts"] = [f for f in facts if f.get("id") != fact_id]
            removed = len(self.data["facts"]) < before
            if removed:
                self._save()
                self._invalidate_cache()
            return removed

    def list_facts(self) -> List[Dict[str, Any]]:
        """Return all stored facts, newest created first."""
        with self._lock:
            return sorted(
                list(self.data.get("facts", [])),
                key=lambda f: f.get("created_at") or "",
                reverse=True,
            )

    # ── pending proposals (consent-gated learning) ────────────
    def add_pending_proposal(self, text: str) -> bool:
        """Append a proposal. Cap at _PROPOSALS_PER_TURN_MAX active items."""
        if not text or not isinstance(text, str):
            return False
        body = text.strip().strip("'\"`").rstrip(".!?,")
        if not body:
            return False
        with self._lock:
            pending = self.data.setdefault("pending", [])
            # Deduplicate case-insensitively.
            if any(p.lower() == body.lower() for p in pending):
                return False
            if len(pending) >= _PROPOSALS_PER_TURN_MAX:
                return False
            pending.append(body)
            self._save()
            self._invalidate_cache()
        return True

    def get_pending_proposals(self) -> List[str]:
        with self._lock:
            return list(self.data.get("pending", []))

    def clear_pending(self) -> int:
        """Drop all pending proposals. Returns how many were removed."""
        with self._lock:
            removed = len(self.data.get("pending", []))
            self.data["pending"] = []
            if removed:
                self._save()
                self._invalidate_cache()
            return removed

    def confirm_proposal(
        self,
        proposal_text: str,
        category: Optional[str] = None,
    ) -> Optional[str]:
        """Move a proposal from pending to stored facts. Returns the new id, or None."""
        if not proposal_text:
            return None
        target = proposal_text.strip()
        with self._lock:
            pending = self.data.setdefault("pending", [])
            match_idx = None
            for i, p in enumerate(pending):
                if p.strip() == target:
                    match_idx = i
                    break
            if match_idx is None:
                return None
            pending.pop(match_idx)
        # Store outside the lock; add_fact acquires it itself.
        new_id = self.add_fact(target, category=category, source="user_confirmed")
        return new_id

    def reject_proposal(self, proposal_text: str) -> bool:
        if not proposal_text:
            return False
        target = proposal_text.strip()
        with self._lock:
            pending = self.data.setdefault("pending", [])
            before = len(pending)
            self.data["pending"] = [p for p in pending if p.strip() != target]
            removed = len(self.data["pending"]) < before
            if removed:
                self._save()
                self._invalidate_cache()
        return removed

    # ── convenience ───────────────────────────────────────────
    def find_facts_matching(self, needle: str) -> List[Dict[str, Any]]:
        """Return facts whose text contains ``needle`` (case-insensitive)."""
        if not needle:
            return []
        nl = needle.lower()
        return [f for f in self.list_facts() if nl in (f.get("text") or "").lower()]


# ─────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────
_instance: Optional[LongTermMemoryStore] = None
_instance_lock = threading.Lock()


def get_long_term_memory() -> LongTermMemoryStore:
    """Process-wide singleton. Always returns a usable store."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = LongTermMemoryStore()
    return _instance


def reset_long_term_memory() -> None:
    """Drop the in-process singleton so a fresh instance is created next call.

    Tests use this; production code should not.
    """
    global _instance
    with _instance_lock:
        _instance = None
