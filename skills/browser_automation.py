"""Browser automation skill for Fairy 2.0."""
import json
import os
import subprocess
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Any

Page = Any
_sync_playwright = None

try:
    from playwright.sync_api import sync_playwright as _sync_playwright
    _PLAYWRIGHT = True
except Exception:
    _PLAYWRIGHT = False

if _PLAYWRIGHT and _sync_playwright is None:
    _PLAYWRIGHT = False

_pw = None
_browser = None
_context = None
_page = None

# ─────────────────────────────────────────────────────────────────
# Browser executable locations
# ─────────────────────────────────────────────────────────────────
_BROWSER_EXES = {
    "opera_gx": [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Opera GX\opera.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Opera\opera.exe"),
        r"C:\Program Files\Opera GX\opera.exe",
        r"C:\Program Files (x86)\Opera GX\opera.exe",
    ],
    "chrome": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ],
    "edge": [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ],
    "brave": [
        os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
    ],
    "firefox": [
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
    ],
}

_BROWSER_PROCESS_NAMES = {
    "opera_gx": ["opera.exe"],
    "chrome":   ["chrome.exe"],
    "edge":     ["msedge.exe"],
    "brave":    ["brave.exe"],
    "firefox":  ["firefox.exe"],
}

_BROWSER_USER_DATA = {
    # Opera GX stores its real profile (with logins/cookies) in Roaming, NOT Local.
    # The Local path only contains a cached/secondary copy. Always prefer Roaming.
    "opera_gx": Path.home() / "AppData" / "Roaming" / "Opera Software" / "Opera GX Stable",
    "chrome":   Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data",
    "edge":     Path.home() / "AppData" / "Local" / "Microsoft" / "Edge" / "User Data",
    "brave":    Path.home() / "AppData" / "Local" / "BraveSoftware" / "Brave-Browser" / "User Data",
}


def _find_exe(browser: str) -> str | None:
    for p in _BROWSER_EXES.get(browser, []):
        if os.path.exists(p):
            return p
    return None


def _is_running(browser: str) -> bool:
    """Check if the browser process is currently running."""
    try:
        import psutil
        targets = {n.lower() for n in _BROWSER_PROCESS_NAMES.get(browser, [])}
        for proc in psutil.process_iter(["name"]):
            name = (proc.info.get("name") or "").lower()
            if name in targets:
                return True
    except Exception:
        pass
    return False


def _open_in_running_browser(url: str, browser: str) -> bool:
    """
    Pass URL to an already-running browser via its command line.
    Chromium-based browsers (Opera GX, Chrome, Edge, Brave) accept a URL
    argument and will open it as a new tab in the existing window — keeping
    all cookies and logged-in sessions intact.
    """
    exe = _find_exe(browser)
    if not exe:
        return False
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # don't flash a CMD window
        subprocess.Popen(
            [exe, url],
            startupinfo=si,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return True
    except Exception:
        return False


def _launch_browser_with_profile(url: str, browser: str) -> bool:
    """
    Launch the browser from scratch using its real user data directory,
    so it starts already logged in to all accounts.
    """
    exe = _find_exe(browser)
    if not exe:
        return False
    user_data = _BROWSER_USER_DATA.get(browser)
    args = [exe]
    if user_data and user_data.exists():
        args += [f"--user-data-dir={user_data}"]
    args.append(url)
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        subprocess.Popen(args, startupinfo=si, creationflags=subprocess.CREATE_NO_WINDOW)
        return True
    except Exception:
        return False


def _resolve_browser(preferred: str | None) -> str:
    """Return a normalized browser name, defaulting to opera_gx."""
    if not preferred:
        return "opera_gx"
    p = preferred.lower().replace(" ", "").replace("_", "").replace("-", "")
    mapping = {
        "operagx": "opera_gx", "opera": "opera_gx",
        "chrome": "chrome", "googlechrome": "chrome",
        "edge": "edge", "msedge": "edge", "microsoftedge": "edge",
        "brave": "brave",
        "firefox": "firefox",
    }
    return mapping.get(p, "opera_gx")


def _detect_logged_in_browser_profile(preferred_browser: str | None = None):
    """Prefer the user's existing browser profile so the browser stays signed in.

    Opera GX stores its real login data in AppData\\Roaming (not Local).
    The Local path is a secondary/cache copy and will NOT have saved passwords
    or active sessions. Always use Roaming for Opera GX.
    """
    home = Path.home()
    candidates = []

    chrome_root = home / "AppData" / "Local" / "Google" / "Chrome" / "User Data"
    edge_root = home / "AppData" / "Local" / "Microsoft" / "Edge" / "User Data"
    brave_root = home / "AppData" / "Local" / "BraveSoftware" / "Brave-Browser" / "User Data"
    # Opera GX real profile is always in Roaming — not Local
    opera_gx_root = home / "AppData" / "Roaming" / "Opera Software" / "Opera GX Stable"
    opera_gx_root_fallback = home / "AppData" / "Local" / "Opera Software" / "Opera GX Stable"

    # If user explicitly named a browser, check it first
    browser_order = []
    if preferred_browser:
        pb = preferred_browser.lower().replace(" ", "").replace("_", "")
        if pb in ("chrome", "googlechrome"):
            browser_order = [chrome_root, edge_root, brave_root]
        elif pb in ("edge", "msedge", "microsoftedge"):
            browser_order = [edge_root, chrome_root, brave_root]
        elif pb in ("brave",):
            browser_order = [brave_root, chrome_root, edge_root]
        elif pb in ("opera", "operagx", "opera_gx"):
            # Roaming first, Local as fallback
            browser_order = [opera_gx_root, opera_gx_root_fallback, chrome_root, edge_root, brave_root]
        else:
            browser_order = [chrome_root, edge_root, brave_root, opera_gx_root, opera_gx_root_fallback]
    else:
        browser_order = [chrome_root, edge_root, brave_root, opera_gx_root, opera_gx_root_fallback]

    for root in browser_order:
        if root.exists():
            candidates.append((str(root), "Default"))
            # Check for additional profiles
            try:
                for entry in os.listdir(root):
                    if entry.startswith("Profile "):
                        candidates.append((str(root), entry))
            except Exception:
                pass

    for root, profile in candidates:
        profile_path = os.path.join(root, profile)
        if os.path.exists(profile_path):
            return root, profile
    return None, None


def _ensure_browser(timeout: float = 8.0, browser: str | None = None):
    """Launch a Playwright-controlled Chromium for scripted automation.

    NOTE: This is ONLY used for scripted actions (click, type, get_text, etc).
    For simple navigation (navigate_to / browser_control go_to) we always use the
    native browser process instead, so the user stays logged in everywhere.

    Opera GX is NOT a registered Playwright channel — we cannot launch it via
    `channel="opera_gx"`. For automation we fall back to stock Chromium (headless
    or headed). If the user needs their logged-in Opera GX session for scripted
    automation, they should run Playwright against Chrome or Edge instead.
    """
    global _pw, _browser, _context, _page
    if not _PLAYWRIGHT:
        return False

    if _page is not None:
        try:
            if not _page.is_closed():
                return True
        except Exception:
            pass

    try:
        if not _PLAYWRIGHT or _sync_playwright is None:
            return False

        if _pw is None:
            _pw = _sync_playwright().start()

        user_data_dir, profile = _detect_logged_in_browser_profile(browser)

        # Only Chrome and Edge are registered Playwright channels.
        # Opera GX and Brave are not — using them as channels raises an error.
        playwright_channels = {"chrome": "chrome", "edge": "msedge"}
        channel = None
        if user_data_dir:
            if "Google\\Chrome" in user_data_dir or "Google/Chrome" in user_data_dir:
                channel = "chrome"
            elif "Microsoft\\Edge" in user_data_dir or "Microsoft/Edge" in user_data_dir:
                channel = "msedge"
            # Opera GX / Brave: no Playwright channel, skip persistent context

        if channel and user_data_dir:
            try:
                launch_kwargs: dict = {"headless": False, "channel": channel}
                if profile:
                    launch_kwargs["args"] = [f"--profile-directory={profile}"]
                _browser = _pw.chromium.launch_persistent_context(user_data_dir, **launch_kwargs)
                _context = _browser
                _page = _context.pages[0] if _context.pages else _context.new_page()
                if _page is not None:
                    return True
            except Exception:
                pass  # fall through to plain Chromium

        # Plain stock Chromium (no user data — for scripted tasks only)
        try:
            _browser = _pw.chromium.launch(headless=False)
            _context = _browser.new_context()
            _page = _context.new_page()
            return _page is not None
        except Exception:
            _browser = _pw.chromium.launch(headless=True)
            _context = _browser.new_context()
            _page = _context.new_page()
            return _page is not None
    except Exception:
        return False


def _normalize_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return "https://example.com"
    if "://" in value:
        return value
    # If it contains spaces, it's not a valid domain — treat as search query
    if " " in value:
        return ""
    if "." not in value:
        value = f"{value}.com"
    return "https://" + value


def navigate_to(url: str, browser: str | None = None) -> str:
    """
    Open a URL in the user's logged-in browser.

    Priority:
      1. Browser already running → pass URL via CLI (new tab, keeps all logins).
         NOTE: do NOT pass --user-data-dir when the browser is already running —
         that would spawn a second isolated instance instead of opening a new tab.
      2. Browser not running → launch with real user-data-dir (starts logged in).
      3. Playwright persistent context (for scripted automation on top of logins).
      4. webbrowser.open fallback (last resort).
    """
    target = _normalize_url(url)
    if not target:
        return search_on_site(url, "google")

    b = _resolve_browser(browser)

    # ── Priority 1 & 2: native browser process (always logged in) ──
    exe = _find_exe(b)
    if exe:
        if _is_running(b):
            # Browser is already open → just pass the URL, no profile args.
            # Chromium-based browsers open it as a new tab in the existing window,
            # preserving every logged-in session.
            try:
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = 0
                subprocess.Popen(
                    [exe, target],
                    startupinfo=si,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                return json.dumps({"status": "ok", "url": target,
                                   "method": f"{b}_new_tab", "logged_in": True})
            except Exception:
                pass  # fall through to launch
        else:
            # Browser is closed → launch with the real profile so it starts logged in.
            user_data = _BROWSER_USER_DATA.get(b)
            # Fallback: try the secondary Local path for Opera GX if Roaming doesn't exist
            if b == "opera_gx" and (user_data is None or not user_data.exists()):
                alt = Path.home() / "AppData" / "Local" / "Opera Software" / "Opera GX Stable"
                if alt.exists():
                    user_data = alt
            args = [exe]
            if user_data and user_data.exists():
                args.append(f"--user-data-dir={user_data}")
            args.append(target)
            try:
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = 0
                subprocess.Popen(
                    args,
                    startupinfo=si,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                return json.dumps({"status": "ok", "url": target,
                                   "method": f"{b}_with_profile", "logged_in": True})
            except Exception:
                pass

    # ── Priority 3: Playwright (for scripted interactions) ──
    if _ensure_browser(browser=b):
        page = _page
        if page is not None:
            try:
                page.goto(target, wait_until="domcontentloaded", timeout=15000)
                time.sleep(1)
                return json.dumps({"status": "ok", "url": target, "method": "playwright"})
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

    # ── Priority 4: system default (last resort) ──
    try:
        opened = webbrowser.open(target, new=2)
        return json.dumps({"status": "ok" if opened else "error",
                           "url": target, "method": "webbrowser_fallback"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def search_on_site(query: str, engine: str = "google") -> str:
    engines = {
        "google": f"https://www.google.com/search?q={urllib.parse.quote(query)}",
        "duckduckgo": f"https://duckduckgo.com/?q={urllib.parse.quote(query)}",
        "bing": f"https://www.bing.com/search?q={urllib.parse.quote(query)}",
        "youtube": f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}",
        "amazon": f"https://www.amazon.com/s?k={urllib.parse.quote(query)}",
    }
    url = engines.get(str(engine).lower(), engines["google"])
    return navigate_to(url)


def click_element(selector: str | None = None, text: str | None = None) -> str:
    if not _ensure_browser():
        return json.dumps({"status": "error", "message": "Browser not available"})
    page = _page
    if page is None:
        return json.dumps({"status": "error", "message": "Browser not available"})
    try:
        if text:
            page.get_by_text(text, exact=False).first.click()
        elif selector:
            page.locator(selector).first.click()
        else:
            return json.dumps({"status": "error", "message": "Provide selector or text"})
        return json.dumps({"status": "ok"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def type_text(selector: str | None = None, text: str = "") -> str:
    if not _ensure_browser():
        return json.dumps({"status": "error", "message": "Browser not available"})
    page = _page
    if page is None:
        return json.dumps({"status": "error", "message": "Browser not available"})
    try:
        target = page.locator(selector) if selector else page.locator(":focus")
        target.fill(text)
        return json.dumps({"status": "ok", "selector": selector or ":focus", "typed": text[:80]})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def scroll_page(direction: str = "down", amount: int = 3) -> str:
    if not _ensure_browser():
        return json.dumps({"status": "error", "message": "Browser not available"})
    page = _page
    if page is None:
        return json.dumps({"status": "error", "message": "Browser not available"})
    try:
        delta = 800 * max(1, int(amount)) if direction == "down" else -800 * max(1, int(amount))
        page.evaluate(f"window.scrollBy(0, {delta})")
        return json.dumps({"status": "ok", "direction": direction, "amount": amount})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def get_page_text() -> str:
    if not _ensure_browser():
        return json.dumps({"status": "error", "message": "Browser not available"})
    page = _page
    if page is None:
        return json.dumps({"status": "error", "message": "Browser not available"})
    try:
        text = page.evaluate("() => document.body.innerText")
        return json.dumps({"status": "ok", "text": text[:3000]})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def get_current_url() -> str:
    if not _ensure_browser():
        return json.dumps({"status": "error", "message": "Browser not available"})
    page = _page
    if page is None:
        return json.dumps({"status": "error", "message": "Browser not available"})
    try:
        return json.dumps({"status": "ok", "url": page.url})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def get_page_info() -> str:
    if not _ensure_browser():
        return json.dumps({"status": "error", "message": "Browser not available"})
    page = _page
    if page is None:
        return json.dumps({"status": "error", "message": "Browser not available"})
    try:
        return json.dumps({
            "status": "ok",
            "url": page.url,
            "title": page.title(),
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def take_screenshot(path: str | None = None) -> str:
    if not _ensure_browser():
        return json.dumps({"status": "error", "message": "Browser not available"})
    page = _page
    if page is None:
        return json.dumps({"status": "error", "message": "Browser not available"})
    try:
        if path is None:
            path = os.path.join(os.path.expanduser("~"), "browser_screenshot.png")
        page.screenshot(path=path)
        return json.dumps({"status": "ok", "path": path})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def close_browser() -> str:
    global _pw, _browser, _context, _page
    try:
        if _page:
            _page.close()
        if _context:
            _context.close()
        if _browser:
            _browser.close()
        if _pw:
            _pw.stop()
    except Exception:
        pass
    _pw = None
    _browser = None
    _context = None
    _page = None
    return json.dumps({"status": "ok"})


def add_to_cart_amazon(query: str) -> str:
    search_on_site(query, "amazon")
    if not _ensure_browser():
        return json.dumps({"status": "error", "message": "Browser not available"})
    page = _page
    if page is None:
        return json.dumps({"status": "error", "message": "Browser not available"})
    try:
        time.sleep(2)
        page.locator("[data-component-type=\"s-search-result\"] h2 a").first.click()
        time.sleep(2)
        page.locator("#add-to-cart-button").first.click()
        return json.dumps({"status": "ok", "message": f"Added {query} to Amazon cart"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def upload_instagram_reel(video_path: str, caption: str = "") -> str:
    return json.dumps({"status": "error", "message": "Instagram reel upload requires manual authentication and is not fully automated."})


def browser_control(parameters: dict | None = None, response=None, player=None, session_memory=None) -> str:
    """Unified browser dispatcher for modern tool-calling.

    go_to / search always use the native browser process (logged-in sessions).
    click / type / scroll / get_text use Playwright for scripted automation.
    """
    params = parameters if isinstance(parameters, dict) else {}
    action = str(params.get("action", "") or params.get("type", "") or "").lower().strip()

    if not action and isinstance(parameters, dict):
        action = str(parameters.get("command", "") or "").lower().strip()

    # ── Navigation — always use native browser (keeps all logins) ──
    if action == "go_to":
        url = params.get("url", "") or params.get("target", "")
        return navigate_to(url, browser=params.get("browser"))

    if action == "search":
        return search_on_site(params.get("query", ""), params.get("engine", "google"))

    # ── New tab shortcut ──
    if action == "new_tab":
        url = params.get("url", "") or "about:blank"
        return navigate_to(url, browser=params.get("browser"))

    # ── Scripted actions — use Playwright ──
    if action in ("click", "smart_click"):
        return click_element(
            selector=params.get("selector"),
            text=params.get("description") or params.get("text"),
        )
    if action == "type":
        return type_text(params.get("selector") or params.get("into", ""), params.get("text", ""))
    if action == "scroll":
        return scroll_page(params.get("direction", "down"), int(params.get("amount", 3)))
    if action == "get_text":
        return get_page_text()
    if action == "get_url":
        return get_current_url()
    if action == "screenshot":
        return take_screenshot(params.get("path"))
    if action == "close":
        return close_browser()
    if action == "list_browsers":
        available = []
        for b, paths in _BROWSER_EXES.items():
            if any(os.path.exists(p) for p in paths):
                available.append(b)
        return json.dumps({"status": "ok", "browsers": available})

    return json.dumps({"status": "error", "message": f"Unknown browser action: {action}"})