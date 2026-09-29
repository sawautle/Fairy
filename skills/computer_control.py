"""Computer control skill for Fairy 2.0."""
import json
import os
import platform
import shutil
import subprocess
from typing import Any

pyautogui: Any | None = None
pyperclip: Any | None = None
AudioUtilities: Any | None = None
IAudioEndpointVolume: Any | None = None
CLSCTX_ALL: Any | None = None
sbc: Any | None = None

try:
    import pyautogui as _pyautogui
    _pyautogui.FAILSAFE = True
    _pyautogui.PAUSE = 0.06
    pyautogui = _pyautogui
    _PYAUTO = True
except Exception:
    _PYAUTO = False


def _pyautogui_safe() -> Any | None:
    return pyautogui if _PYAUTO and pyautogui is not None else None


def _pyperclip_safe() -> Any | None:
    return pyperclip if _CLIP and pyperclip is not None else None


def _screen_brightness_safe() -> Any | None:
    return sbc if _SBC and sbc is not None else None

try:
    import pyperclip as _pyperclip
    pyperclip = _pyperclip
    _CLIP = True
except Exception:
    _CLIP = False

try:
    from pycaw.pycaw import AudioUtilities as _AudioUtilities, IAudioEndpointVolume as _IAudioEndpointVolume
    from comtypes import CLSCTX_ALL as _CLSCTX_ALL
    AudioUtilities = _AudioUtilities
    IAudioEndpointVolume = _IAudioEndpointVolume
    CLSCTX_ALL = _CLSCTX_ALL
    _PYCAW = True
except Exception:
    _PYCAW = False

try:
    import screen_brightness_control as _sbc
    sbc = _sbc
    _SBC = True
except Exception:
    _SBC = False


def _normalize_percent(value) -> int:
    if value is None:
        return 0
    try:
        return max(0, min(100, int(float(value))))
    except Exception:
        return 0


def _volume_key_fallback(percent: int):
    """Use ctypes to simulate volume keys on Windows."""
    try:
        import ctypes
        VK_VOLUME_DOWN = 0xAE
        VK_VOLUME_UP = 0xAF
        for _ in range(50):
            ctypes.windll.user32.keybd_event(VK_VOLUME_DOWN, 0, 0, 0)
        steps = max(0, int(percent // 2))
        for _ in range(steps):
            ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, 0, 0)
        return True
    except Exception as e:
        return str(e)


def set_volume(percent: int | None = None, value: int | None = None, **kwargs) -> str:
    try:
        resolved = kwargs.get("percent")
        if resolved is None:
            resolved = kwargs.get("value")
        if resolved is None:
            resolved = percent
        if resolved is None:
            resolved = value
        p = _normalize_percent(resolved)

        if platform.system() == "Windows":
            if _PYCAW and AudioUtilities is not None and IAudioEndpointVolume is not None and CLSCTX_ALL is not None:
                try:
                    devices = AudioUtilities.GetSpeakers()
                    if not hasattr(devices, "Activate"):
                        raise AttributeError("AudioDevice has no Activate")
                    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                    volume = interface.QueryInterface(IAudioEndpointVolume)
                    volume.SetMasterVolumeLevelScalar(p / 100.0, None)
                    return json.dumps({"status": "ok", "volume": p, "method": "pycaw"})
                except Exception as exc:
                    print(f"[ComputerControl] pycaw volume failed: {exc}; falling back")
            r = _volume_key_fallback(p)
            if r is True:
                return json.dumps({"status": "ok", "volume": p, "method": "key_fallback"})
            return json.dumps({"status": "error", "message": str(r) if isinstance(r, str) else "Volume control unavailable on this machine."})
        if platform.system() == "Darwin":
            subprocess.run(["osascript", "-e", f"set volume output volume {p}"], capture_output=True, timeout=10)
            return json.dumps({"status": "ok", "volume": p, "method": "macos"})
        if platform.system() == "Linux":
            subprocess.run(["amixer", "set", "Master", f"{p}%"], capture_output=True, timeout=10)
            return json.dumps({"status": "ok", "volume": p, "method": "alsa"})
        return json.dumps({"status": "error", "message": "Unsupported platform"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def get_volume() -> str:
    try:
        if _PYCAW and platform.system() == "Windows" and AudioUtilities is not None and IAudioEndpointVolume is not None and CLSCTX_ALL is not None:
            devices = AudioUtilities.GetSpeakers()
            if not hasattr(devices, "Activate"):
                return json.dumps({"status": "error", "message": "Audio device does not support activation on this machine."})
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = interface.QueryInterface(IAudioEndpointVolume)
            cur = int(volume.GetMasterVolumeLevelScalar() * 100)
            return json.dumps({"status": "ok", "volume": cur})
        return json.dumps({"status": "error", "message": "get_volume requires pycaw on Windows. pip install pycaw comtypes"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def set_brightness(percent: int | None) -> str:
    try:
        p = max(0, min(100, int(float(percent or 0))))
        brightness_api = _screen_brightness_safe()
        if brightness_api is not None:
            brightness_api.set_brightness(p)
            return json.dumps({"status": "ok", "brightness": p, "method": "sbc"})
        if platform.system() == "Windows":
            subprocess.run(["powershell", "-Command", f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{p})"], capture_output=True, timeout=10)
            return json.dumps({"status": "ok", "brightness": p, "method": "wmi"})
        if platform.system() == "Darwin":
            subprocess.run(["brightness", "-v", str(p / 100)], capture_output=True, timeout=10)
            return json.dumps({"status": "ok", "brightness": p, "method": "brightness"})
        return json.dumps({"status": "error", "message": "Install screen-brightness-control: pip install screen-brightness-control"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def get_brightness() -> str:
    try:
        brightness_api = _screen_brightness_safe()
        if brightness_api is not None:
            b = brightness_api.get_brightness()
            val = b[0] if isinstance(b, list) else b
            return json.dumps({"status": "ok", "brightness": val, "method": "sbc"})
        return json.dumps({"status": "error", "message": "Install screen-brightness-control: pip install screen-brightness-control"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def take_screenshot(path: str | None = None) -> str:
    try:
        auto = _pyautogui_safe()
        if auto is None:
            return json.dumps({"status": "error", "message": "pyautogui not installed. pip install pyautogui"})
        if path is None:
            path = os.path.join(os.path.expanduser("~"), "fairy_screenshot.png")
        img = auto.screenshot()
        img.save(path)
        return json.dumps({"status": "ok", "path": path})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def type_keys(text: str | None) -> str:
    try:
        auto = _pyautogui_safe()
        if auto is None:
            return json.dumps({"status": "error", "message": "pyautogui not installed"})
        value = str(text or "")
        auto.typewrite(value, interval=0.01)
        return json.dumps({"status": "ok", "typed": value})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def press_key(key: str | None) -> str:
    try:
        auto = _pyautogui_safe()
        if auto is None:
            return json.dumps({"status": "error", "message": "pyautogui not installed"})
        value = str(key or "")
        auto.press(value)
        return json.dumps({"status": "ok", "key": value})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def hotkey(*keys: str | None) -> str:
    try:
        auto = _pyautogui_safe()
        if auto is None:
            return json.dumps({"status": "error", "message": "pyautogui not installed"})
        normalized = tuple(str(k or "") for k in keys)
        auto.hotkey(*normalized)
        return json.dumps({"status": "ok", "keys": normalized})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def open_application(app_or_url: str) -> str:
    """Open a desktop application by friendly name OR a URL.

    Bare names like ``spotify`` or ``steam`` are resolved via three layers in
    order, so a user can say ``open spotify`` without knowing the install
    path:
      1. ``shutil.which(name)`` — picks up anything on PATH (works for
         cmd-line tools and any user whose %PATH% includes the app folder).
      2. ``_LAUNCH_ALIASES[name]`` — known absolute paths for popular apps
         whose installer doesn't put them on PATH (Spotify, Steam, Discord,
         etc.). Avoids ``[WinError 2] The system cannot find the file
         specified`` from a bare ``os.startfile("spotify")`` (ShellExecuteW
         only treats a string as a file/URL if it has a registered handler —
         a bare ``"spotify"`` has none).
      3. ``os.startfile(name)`` — last-resort fallback for legacy URL/file
         protocols (``mailto:``, ``ms-settings:``, document paths).
    """
    try:
        import webbrowser
        if app_or_url.startswith(("http://", "https://", "www.")):
            webbrowser.open(app_or_url)
            return json.dumps({"status": "ok", "opened": app_or_url, "type": "url"})

        if platform.system() == "Windows":
            name_lower = (app_or_url or "").strip().lower()
            # Track the list of paths we searched so we can report a useful
            # error when nothing resolved.
            searched = []
            launched_proc = None
            launched_target = None
            # Layer 1: bare exe on PATH (e.g. `code.exe` from VS Code).
            if name_lower:
                resolved = shutil.which(name_lower) or shutil.which(name_lower + ".exe")
                if resolved:
                    searched.append(resolved)
                    launched_proc = subprocess.Popen(
                        [resolved],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        stdin=subprocess.DEVNULL,
                        close_fds=True,
                    )
                    launched_target = name_lower
                else:
                    # Layer 2: known install paths for apps whose installer does
                    # not put them on PATH. Tried in order; the first hit wins.
                    for alias in _LAUNCH_ALIASES.get(name_lower, ()):
                        searched.append(alias)
                        if os.path.isfile(alias):
                            launched_proc = subprocess.Popen(
                                [alias],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL,
                                close_fds=True,
                            )
                            launched_target = alias
                            break
            if launched_proc is None:
                # Nothing resolved via Layers 1-2. Don't silently fall through
                # to os.startfile (which would either show a Windows error
                # dialog or succeed for a registered protocol even if the
                # intent was a desktop app). Report the searched paths so the
                # caller — and the model — can see why.
                return json.dumps({
                    "status": "error",
                    "opened": name_lower or app_or_url,
                    "type": "app",
                    "error": "executable not found",
                    "searched": searched,
                    "verified": False,
                })
            # Layer 3 + non-Windows fall through to here only on non-Windows,
            # or on Windows if we launched something but still want to try
            # os.startfile for registered protocols. For the normal Windows
            # app-launch path we never reach here (launched_proc is set).
            # We verify the launched process below.
            ok, verify_msg, pid = _verify_popen_launched(launched_proc, name_lower)
            if ok:
                return json.dumps({
                    "status": "ok",
                    "opened": launched_target,
                    "type": "app",
                    "pid": pid,
                    "verified": True,
                })
            return json.dumps({
                "status": "error",
                "opened": launched_target,
                "type": "app",
                "error": verify_msg,
                "verified": False,
            })
        elif platform.system() == "Darwin":
            r = subprocess.run(["open", app_or_url], capture_output=True, timeout=10)
            if r.returncode == 0:
                return json.dumps({"status": "ok", "opened": app_or_url, "type": "app", "verified": True})
            return json.dumps({"status": "error", "opened": app_or_url, "type": "app", "error": r.stderr.decode(errors="ignore") or "launcher returned non-zero", "verified": False})
        else:
            r = subprocess.run(["xdg-open", app_or_url], capture_output=True, timeout=10)
            if r.returncode == 0:
                return json.dumps({"status": "ok", "opened": app_or_url, "type": "app", "verified": True})
            return json.dumps({"status": "error", "opened": app_or_url, "type": "app", "error": r.stderr.decode(errors="ignore") or "launcher returned non-zero", "verified": False})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e), "verified": False})


def _verify_popen_launched(p: subprocess.Popen, name_lower: str) -> tuple[bool, str, int | None]:
    """
    Confirm a Popen-spawned process actually came up. Returns (ok, message, pid).

    Some apps (Discord, Spotify, Steam) spawn a small launcher/updater that
    exits immediately after starting the real process. We handle that by
    also scanning for a process whose image name matches the app name.
    """
    import time as _time
    # Brief wait for the process to spawn (or exit, if it crashed).
    _time.sleep(1.5)
    pid = getattr(p, "pid", None)

    # 1) Did the just-spawned process survive?
    try:
        returncode = p.poll()
    except Exception:
        returncode = None

    alive = False
    if pid is not None:
        # Try psutil first.
        try:
            import psutil
            try:
                proc = psutil.Process(pid)
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    alive = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                alive = False
        except ImportError:
            # Fall back to Windows tasklist.
            if platform.system() == "Windows":
                try:
                    r = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                        capture_output=True, text=True, timeout=5,
                    )
                    out = (r.stdout or "").lower()
                    if "no tasks" not in out and str(pid) in out:
                        alive = True
                except Exception:
                    pass
            else:
                # POSIX: /proc exists.
                try:
                    import os as _os
                    _os.kill(pid, 0)
                    alive = True
                except Exception:
                    alive = False

    if alive:
        return True, "", pid

    # 2) Launcher died — did it spawn the real app? Scan for a process whose
    #    name matches the friendly name. Use the same name alias table that
    #    agent_controller uses for verification.
    matched = _find_process_by_name(name_lower)
    if matched is True:
        return True, "", None
    if matched is False:
        return False, "process exited immediately after launch", None
    # matched is None (couldn't check at all)
    return False, "could not verify launch (no psutil, tasklist failed)", None


def _find_process_by_name(name_lower: str) -> bool | None:
    """
    Return True if a running process name matches the friendly app name,
    False if no match was found, None if we can't check at all.
    """
    candidates = set()
    # Common process-name alias table (subset; mirrors agent_controller).
    _ALIASES = {
        "chrome": ["chrome.exe"],
        "google chrome": ["chrome.exe"],
        "firefox": ["firefox.exe"],
        "edge": ["msedge.exe"],
        "microsoft edge": ["msedge.exe"],
        "vscode": ["code.exe"],
        "vs code": ["code.exe"],
        "visual studio code": ["code.exe"],
        "notepad": ["notepad.exe"],
        "notepad++": ["notepad++.exe"],
        "calculator": ["calculator.exe", "calc.exe"],
        "calc": ["calculator.exe", "calc.exe"],
        "taskmgr": ["taskmgr.exe"],
        "task manager": ["taskmgr.exe"],
        "explorer": ["explorer.exe"],
        "file explorer": ["explorer.exe"],
        "cmd": ["cmd.exe"],
        "command prompt": ["cmd.exe"],
        "powershell": ["powershell.exe"],
        "spotify": ["spotify.exe"],
        "discord": ["discord.exe"],
        "slack": ["slack.exe"],
        "teams": ["teams.exe"],
        "zoom": ["zoom.exe"],
        "telegram": ["telegram.exe"],
        "steam": ["steam.exe", "steamwebhelper.exe"],
        "vlc": ["vlc.exe"],
        "obs": ["obs64.exe", "obs.exe"],
        "obs studio": ["obs64.exe", "obs.exe"],
        "skype": ["skype.exe", "lync.exe"],
        "outlook": ["outlook.exe"],
        "word": ["winword.exe"],
        "excel": ["excel.exe"],
        "powerpoint": ["powerpoint.exe"],
        "hoyoplay": ["hoyoplay.exe", "hoyo_play.exe", "hoyolauncher.exe"],
    }
    for c in _ALIASES.get(name_lower, []):
        candidates.add(c.lower())
    if not candidates:
        candidates.add(f"{name_lower}.exe")

    try:
        import psutil
        for proc in psutil.process_iter(['name']):
            n = (proc.info.get('name') or '').lower()
            if n in candidates:
                return True
        return False
    except ImportError:
        pass

    if platform.system() == "Windows":
        try:
            r = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            out = (r.stdout or "").lower()
            for c in candidates:
                if c in out:
                    return True
            return False
        except Exception:
            return None
    return None


# Known absolute install paths for popular Windows apps whose installer
# doesn't put them on PATH. Tried in order by open_application(); the first
# existing path wins. Keep this list focused on apps a user is most likely
# to invoke by friendly name — bare ``os.startfile("spotify")`` previously
# failed with [WinError 2] because ShellExecuteW needs a registered handler
# or a real path.
def _local_appdata() -> str:
    return os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local"))


def _appdata() -> str:
    return os.environ.get("APPDATA", os.path.expanduser("~\\AppData\\Roaming"))


def _program_files() -> str:
    return os.environ.get("ProgramFiles", "C:\\Program Files")


def _program_files_x86() -> str:
    return os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")


_LAUNCH_ALIASES = {
    "spotify": (
        os.path.join(_local_appdata(), "Microsoft", "WindowsApps", "Spotify.exe"),
        os.path.join(_local_appdata(), "Spotify", "Spotify.exe"),
        os.path.join(_appdata(), "Spotify", "Spotify.exe"),
    ),
    "steam": (
        os.path.join(_program_files_x86(), "Steam", "steam.exe"),
        "C:\\Steam\\steam.exe",
    ),
    "discord": (
        os.path.join(_local_appdata(), "Discord", "Update.exe"),
        os.path.join(_local_appdata(), "Discord", "Discord.exe"),
    ),
    "telegram": (
        os.path.join(_program_files_x86(), "Telegram Desktop", "Telegram.exe"),
    ),
    "epic games": (
        os.path.join(_program_files_x86(), "Epic Games", "Launcher", "Portal", "Binaries", "Win32", "EpicGamesLauncher.exe"),
    ),
    "epic games launcher": (
        os.path.join(_program_files_x86(), "Epic Games", "Launcher", "Portal", "Binaries", "Win32", "EpicGamesLauncher.exe"),
    ),
    "hoyoplay": (r"D:\HoYoPlay\launcher.exe",),
    "hoyo play": (r"D:\HoYoPlay\launcher.exe",),
}


# Process name → executable name mappings for cross-platform close
_CLOSE_NAME_ALIASES = {
    "chrome": ["chrome.exe"],
    "google chrome": ["chrome.exe"],
    "firefox": ["firefox.exe"],
    "edge": ["msedge.exe"],
    "microsoft edge": ["msedge.exe"],
    "vscode": ["code.exe"],
    "vs code": ["code.exe"],
    "visual studio code": ["code.exe"],
    "notepad": ["notepad.exe"],
    "notepad++": ["notepad++.exe"],
    "calculator": ["calculator.exe", "calc.exe"],
    "calc": ["calculator.exe", "calc.exe"],
    "taskmgr": ["taskmgr.exe"],
    "task manager": ["taskmgr.exe"],
    "explorer": ["explorer.exe"],
    "file explorer": ["explorer.exe"],
    "cmd": ["cmd.exe"],
    "command prompt": ["cmd.exe"],
    "powershell": ["powershell.exe"],
    "spotify": ["spotify.exe"],
    "discord": ["discord.exe"],
    "slack": ["slack.exe"],
    "teams": ["teams.exe"],
    "zoom": ["zoom.exe"],
    "telegram": ["telegram.exe"],
    "steam": ["steam.exe", "steamwebhelper.exe"],
    "vlc": ["vlc.exe"],
    "obs": ["obs64.exe", "obs.exe"],
    "obs studio": ["obs64.exe", "obs.exe"],
    "skype": ["skype.exe", "lync.exe"],
    "outlook": ["outlook.exe"],
    "word": ["winword.exe"],
    "excel": ["excel.exe"],
    "powerpoint": ["powerpoint.exe"],
}


def _resolve_close_targets(target: str) -> list[str]:
    """Resolve a human-friendly app name to a list of candidate process names."""
    target_lower = target.lower().strip()
    if target_lower in _CLOSE_NAME_ALIASES:
        return list(_CLOSE_NAME_ALIASES[target_lower])
    # If user already provided a process name with .exe
    if target_lower.endswith(".exe"):
        return [target_lower]
    # Try as-is with .exe appended
    return [f"{target_lower}.exe"]


def close_application(target: str) -> str:
    """Close (terminate) a running application by name.

    Returns JSON with status, the names that were targeted, and a count of
    processes terminated. Does NOT hallucinate: the "terminated" count
    reflects what the OS actually reported.
    """
    if not target or not isinstance(target, str):
        return json.dumps({"status": "error", "message": "close_application requires a target app name."})

    candidates = _resolve_close_targets(target)
    terminated = 0
    attempted = []

    try:
        # Prefer psutil if available (cleaner)
        try:
            import psutil
            target_set = {c.lower() for c in candidates}
            for proc in psutil.process_iter(['name']):
                name = (proc.info.get('name') or '').lower()
                if name in target_set:
                    attempted.append(name)
                    try:
                        proc.terminate()
                        terminated += 1
                    except Exception:
                        # Insufficient permissions or process already gone
                        pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        except ImportError:
            # Fall back to taskkill on Windows
            if platform.system() == "Windows":
                for cand in candidates:
                    attempted.append(cand.lower())
                    r = subprocess.run(
                        ["taskkill", "/IM", cand, "/F"],
                        capture_output=True, text=True, timeout=10,
                    )
                    # taskkill returns 0 on success, 128 if not found
                    if r.returncode == 0:
                        terminated += 1
            else:
                # macOS/Linux: try pkill
                for cand in candidates:
                    name = cand.replace(".exe", "")
                    attempted.append(name)
                    r = subprocess.run(
                        ["pkill", "-f", name],
                        capture_output=True, text=True, timeout=10,
                    )
                    if r.returncode == 0:
                        terminated += 1

        if terminated > 0:
            return json.dumps({
                "status": "ok",
                "target": target,
                "terminated": terminated,
                "attempted": attempted,
            })
        else:
            return json.dumps({
                "status": "ok",
                "target": target,
                "terminated": 0,
                "attempted": attempted,
                "note": "no running process matched; nothing to close",
            })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def get_clipboard() -> str:
    try:
        clip = _pyperclip_safe()
        if clip is None:
            return json.dumps({"status": "error", "message": "pyperclip not installed. pip install pyperclip"})
        return json.dumps({"status": "ok", "text": clip.paste()})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def set_clipboard(text: str | None = None) -> str:
    try:
        if not _CLIP or pyperclip is None:
            return json.dumps({"status": "error", "message": "pyperclip not installed"})
        value = str(text or "")
        pyperclip.copy(value)
        return json.dumps({"status": "ok"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def computer_control(
    action: str | dict | None = None,
    value=None,
    parameters: dict | None = None,
    response=None,
    player=None,
    session_memory=None,
) -> str:
    """Legacy and controller-compatible unified dispatcher."""
    if isinstance(action, dict):
        params = action
        action_name = str(params.get("action", "")).lower().strip()
        value = params.get("value", params.get("target", params.get("text", params.get("key", params.get("path", params.get("url"))))))
    else:
        params = parameters or {}
        action_name = str(action or params.get("action", "")).lower().strip()
        if value is None:
            value = params.get("value", params.get("text", params.get("key", params.get("path", params.get("url", "")))))

    if not action_name:
        return json.dumps({"status": "error", "message": "No action specified for computer_control."})

    if action_name == "set_volume":
        return set_volume(value=value, percent=params.get("percent", params.get("volume")))
    if action_name == "get_volume":
        return get_volume()
    if action_name == "set_brightness":
        brightness_value = value if value is not None else params.get("percent", params.get("brightness"))
        return set_brightness(brightness_value)
    if action_name == "get_brightness":
        return get_brightness()
    if action_name == "take_screenshot":
        return take_screenshot(value if value is not None else params.get("path"))
    if action_name == "type_keys":
        return type_keys(value if value is not None else params.get("text"))
    if action_name == "press_key":
        return press_key(value if value is not None else params.get("key"))
    if action_name == "hotkey":
        items = value if isinstance(value, (list, tuple)) else [value if value is not None else params.get("keys")]
        return hotkey(*items)
    if action_name == "open":
        return open_application(str(value or params.get("app_or_url", "")))
    if action_name == "open_app" or action_name == "open_url":
        # Aliases for the planner misroute path
        return open_application(str(value or params.get("app_or_url", "")))
    if action_name == "close_app" or action_name == "close" or action_name == "kill_app":
        return close_application(str(value or params.get("app_or_url", "") or ""))
    if action_name == "get_clipboard":
        return get_clipboard()
    if action_name == "set_clipboard":
        return set_clipboard(value)
    return json.dumps({"status": "error", "message": f"Unknown computer_control action: {action_name}"})