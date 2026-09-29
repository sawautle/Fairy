
"""Reminders and notifications skill for Fairy 2.0."""
import threading
import time
import json
import os
import platform

try:
    from plyer import notification
    _PLYER = True
except Exception:
    _PLYER = False

_scheduled = []
_lock = threading.Lock()


def _notify_fallback(title: str, message: str) -> bool:
    try:
        if platform.system() == "Windows":
            import subprocess
            subprocess.run(["msg", "*", "/TIME:10", title + "\n" + message], capture_output=True, timeout=10)
            return True
    except Exception:
        pass
    return False


def _send_notification(title: str, message: str) -> dict:
    if _PLYER:
        try:
            notification.notify(title=title, message=message, timeout=10)
            return {"status": "sent", "method": "plyer"}
        except Exception:
            pass
    if _notify_fallback(title, message):
        return {"status": "sent", "method": "fallback"}
    return {"status": "error", "message": "No notification method available. Install plyer: pip install plyer"}


def _watcher():
    while True:
        time.sleep(1)
        now = time.time()
        with _lock:
            for item in _scheduled:
                if not item["triggered"] and now >= item["fire_at"]:
                    item["triggered"] = True
                    _send_notification(item["title"], item["message"])


_watcher_thread = threading.Thread(target=_watcher, daemon=True)
_watcher_thread.start()


def send_notification(title: str, message: str) -> str:
    return json.dumps(_send_notification(title, message))


def set_reminder(title: str, message: str, delay_seconds: int) -> str:
    try:
        delay = int(delay_seconds)
        fire_at = time.time() + delay
        with _lock:
            _scheduled.append({"title": title, "message": message, "fire_at": fire_at, "triggered": False})
        return json.dumps({"status": "scheduled", "fires_in_seconds": delay, "title": title})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def list_reminders() -> str:
    with _lock:
        pending = [
            {
                "title": r["title"],
                "message": r["message"],
                "fires_in_seconds": max(0, int(r["fire_at"] - time.time())),
            }
            for r in _scheduled if not r["triggered"]
        ]
    return json.dumps({"pending": pending})


def cancel_reminder(title: str) -> str:
    with _lock:
        global _scheduled
        before = len(_scheduled)
        _scheduled = [r for r in _scheduled if not (r["title"] == title and not r["triggered"])]
        after = len(_scheduled)
    return json.dumps({"cancelled": before - after})


def reminder_tool(action: str, title: str = "", message: str = "", delay_seconds: int = 0) -> str:
    action = str(action).lower().strip()
    if action == "notify":
        return send_notification(title, message)
    if action == "set_reminder":
        return set_reminder(title, message, delay_seconds)
    if action == "list":
        return list_reminders()
    if action == "cancel":
        return cancel_reminder(title)
    return json.dumps({"error": f"Unknown reminder action: {action}"})