"""Persistent JSON memory for Fairy 2.0."""
import json
import re
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

_BASE = Path(__file__).resolve().parent.parent
_MEM_PATH = _BASE / "memory" / "long_term.json"

# Any fact matching one of these is treated as system telemetry, not a
# real conversational fact. Extend this if you add more alert types.
_ALERT_PATTERNS = [
    re.compile(r"^system alert:", re.IGNORECASE),
    re.compile(r"\[SYSTEM_ALERT\]", re.IGNORECASE),
]

# Minimum gap between two *similar* alerts before we bother storing another one.
_ALERT_DEDUP_WINDOW = timedelta(minutes=15)
_MAX_ALERTS = 10  # alerts are a rolling status, not history — keep this small


class MemoryManager:
    """Small, process-local persistent memory store."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or _MEM_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                loaded = {}
        else:
            loaded = {}
        # Migrate/ensure all expected keys exist, including the new bucket.
        loaded.setdefault("sessions", [])
        loaded.setdefault("preferences", {})
        loaded.setdefault("projects", {})
        loaded.setdefault("facts", [])
        loaded.setdefault("system_alerts", [])
        loaded.setdefault("last_updated", datetime.now().isoformat())
        return loaded

    def save(self) -> None:
        with self._lock:
            self.data["last_updated"] = datetime.now().isoformat()
            temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            temp_path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(self.path)

    # ── sessions ──────────────────────────────────────────────────
    def add_session_summary(self, summary: str) -> None:
        self.data["sessions"].append({
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        })
        self._trim("sessions", 50)
        self.save()

    def get_recent_context(self, n: int = 3) -> str:
        """Conversational memory only. Alerts never leak in here."""
        recent = self.data.get("sessions", [])[-n:]
        if not recent:
            return ""
        return "\n".join(f"- {s['summary']}" for s in recent)

    # ── preferences / projects ───────────────────────────────────
    def set_preference(self, key: str, value: Any) -> None:
        self.data["preferences"][key] = value
        self.save()

    def get_preference(self, key: str, default=None):
        return self.data["preferences"].get(key, default)

    def add_project(self, name: str, status: str = "active", notes: str = "") -> None:
        self.data["projects"][name] = {
            "status": status,
            "notes": notes,
            "updated": datetime.now().isoformat(),
        }
        self.save()

    def get_project(self, name: str) -> Optional[dict]:
        return self.data["projects"].get(name)

    # ── facts (now filtered) ─────────────────────────────────────
    def add_fact(self, fact: str) -> None:
        """
        Store a real conversational fact. System-alert-shaped text is
        automatically routed to `add_system_alert` instead, so noisy
        telemetry can never flood the fact list Fairy reasons over.
        """
        if any(p.search(fact) for p in _ALERT_PATTERNS):
            self.add_system_alert(fact)
            return
        self.data["facts"].append({"text": fact, "time": datetime.now().isoformat()})
        self._trim("facts", 100)
        self.save()

    def get_facts(self, n: int = 20) -> List[str]:
        return [f["text"] for f in self.data.get("facts", [])[-n:]]

    # ── system alerts: capped, deduped, summarized ───────────────
    def add_system_alert(self, alert: str) -> None:
        """
        Rate-limited, deduplicated store for transient system status
        (GPU/RAM/etc). Keeps only the last _MAX_ALERTS, and skips
        near-duplicate alerts fired within _ALERT_DEDUP_WINDOW so a
        noisy monitor loop can't drown out real memory.
        """
        now = datetime.now()
        key = self._alert_key(alert)
        alerts = self.data["system_alerts"]

        for existing in reversed(alerts):
            if existing.get("key") == key:
                last_time = datetime.fromisoformat(existing["time"])
                if now - last_time < _ALERT_DEDUP_WINDOW:
                    existing["time"] = now.isoformat()
                    existing["text"] = alert
                    existing["count"] = existing.get("count", 1) + 1
                    self.save()
                    return
                break

        alerts.append({"text": alert, "time": now.isoformat(), "key": key, "count": 1})
        self._trim("system_alerts", _MAX_ALERTS)
        self.save()

    @staticmethod
    def _alert_key(alert: str) -> str:
        """Collapse 'GPU load is at 99%' and 'GPU load is at 96%' to the same bucket."""
        return re.sub(r"\d+%?", "N", alert).strip().lower()

    def get_alert_summary(self) -> str:
        """One short line per distinct alert type — safe to drop into a prompt."""
        alerts = self.data.get("system_alerts", [])
        if not alerts:
            return ""
        lines = []
        for a in alerts[-5:]:
            suffix = f" (x{a['count']} recently)" if a.get("count", 1) > 1 else ""
            lines.append(f"- {a['text']}{suffix}")
        return "\n".join(lines)

    # ── structured data → prompt-safe summaries ──────────────────
    @staticmethod
    def summarize_structured(data: Any, max_items: int = 5) -> str:
        """
        Compress a nested dict/list (e.g. a system_monitor telemetry blob)
        into short plain-English lines instead of raw JSON, so it doesn't
        get dumped verbatim into a small local model's context.
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return data[:500]

        if not isinstance(data, dict):
            return str(data)[:500]

        lines = []
        for key, value in data.items():
            if isinstance(value, list):
                items = value[:max_items]
                if items and isinstance(items[0], dict):
                    parts = []
                    for item in items:
                        # pick the 2-3 most informative fields rather than dumping the dict
                        subparts = [f"{k}={v}" for k, v in list(item.items())[:3]]
                        parts.append(", ".join(subparts))
                    lines.append(f"{key}: " + " | ".join(parts))
                else:
                    lines.append(f"{key}: {', '.join(str(i) for i in items)}")
            elif isinstance(value, dict):
                lines.append(f"{key}: " + ", ".join(f"{k}={v}" for k, v in list(value.items())[:5]))
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _trim(self, key: str, max_items: int) -> None:
        if len(self.data.get(key, [])) > max_items:
            self.data[key] = self.data[key][-max_items:]


_instance: Optional[MemoryManager] = None

def get_memory() -> MemoryManager:
    global _instance
    if _instance is None:
        _instance = MemoryManager()
    return _instance