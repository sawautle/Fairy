"""Sandbox for testing generated Python code."""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple, Optional

_BUILTINS = {"os", "sys", "json", "time", "re", "math", "random", "datetime",
             "collections", "itertools", "functools", "typing", "pathlib", "urllib"}

def _extract_missing_module(error_text: str) -> Optional[str]:
    q = chr(39)
    patterns = [
        f"No module named {q}([^q]+){q}",
        f"ModuleNotFoundError: No module named {q}([^q]+){q}",
    ]
    for pat in patterns:
        m = re.search(pat, error_text)
        if m:
            return m.group(1)
    m = re.search(r"ImportError:.*" + q + r"([^" + q + r"]+)" + q, error_text)
    if m:
        return m.group(1)
    return None

def _is_safe_module(name: str) -> bool:
    if name in _BUILTINS:
        return False
    safe = {"requests", "beautifulsoup4", "bs4", "lxml", "html5lib", "httpx",
            "pillow", "pil", "numpy", "pandas", "markdown", "yaml", "toml",
            "psutil", "plyer", "pyautogui", "pyperclip", "pynvml", "wmi",
            "mss", "cv2", "PIL", "google", "genai", "duckduckgo_search", "playwright"}
    return name.split(".")[0] in safe

def try_install(module: str) -> Tuple[bool, str]:
    if os.environ.get("FAIRY_ALLOW_AUTO_INSTALL", "0") != "1":
        return False, "Automatic dependency installation is disabled. Install approved dependencies manually."
    pkg = module.split(".")[0]
    if not _is_safe_module(pkg):
        return False, f"Module {pkg} not in safe install list"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return True, f"Installed {pkg}"
        return False, result.stderr[:500]
    except Exception as e:
        return False, str(e)

def run_sandbox(code: str, kwargs: dict = None, timeout: int = 10) -> dict:
    kwargs = kwargs or {}
    kw_json = json.dumps(json.dumps(kwargs))
    lines = [
        "import json, sys, traceback",
        f"kwargs = json.loads({kw_json})",
        code,
        "try:",
        "    result = run(**kwargs)",
        "    print(chr(10) + '__FAIRY_RESULT__' + json.dumps({'success': True, 'result': result}))",
        "except Exception as e:",
        "    print(chr(10) + '__FAIRY_RESULT__' + json.dumps({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}))",
    ]
    script = chr(10).join(lines)
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)

        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as e:
            partial_out = e.stdout if isinstance(e.stdout, str) else ""
            partial_err = e.stderr if isinstance(e.stderr, str) else ""
            combined = (partial_out + chr(10) + partial_err).strip()
            missing = _extract_missing_module(combined) if combined else None
            return {
                "success": False,
                "error": f"Code timed out after {timeout}s",
                "missing_module": missing,
                "stdout": partial_out[:2000],
                "stderr": partial_err[:2000],
                "timed_out": True,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Sandbox subprocess failed to launch: {e}",
                "missing_module": None,
                "stdout": "",
                "stderr": "",
            }

        stdout = result.stdout
        stderr = result.stderr
        marker = "__FAIRY_RESULT__"
        if marker in stdout:
            json_part = stdout.split(marker)[-1].strip()
            try:
                data = json.loads(json_part)
                if not data.get("success"):
                    missing = _extract_missing_module(data.get("error", "") + stderr)
                    data["missing_module"] = missing
                return data
            except Exception:
                pass
        combined = (stdout + chr(10) + stderr).strip()
        missing = _extract_missing_module(combined)
        return {
            "success": False,
            "error": combined[:2000] or "Unknown sandbox error",
            "missing_module": missing,
            "stdout": stdout[:2000],
            "stderr": stderr[:2000],
        }
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

def test_skill_code(code: str, kwargs: dict = None, timeout: int = 10) -> Tuple[bool, str]:
    """Compatibility wrapper for the older API used by the controller.

    The rest of the app expects a tuple of (passed, output), where
    ``passed`` is a boolean and ``output`` is a human-readable string.
    """
    result = run_sandbox(code, kwargs, timeout=timeout)
    if result.get("success"):
        return True, result.get("result") or "Code executed successfully"
    error = result.get("error") or "Unknown sandbox error"
    if result.get("missing_module"):
        error = f"Missing module: {result['missing_module']}\n{error}"
    if result.get("timed_out"):
        error = f"[TIMEOUT] {error}"
    return False, error


def test_skill_file(path: Path, kwargs: dict = None) -> dict:
    try:
        code = path.read_text(encoding="utf-8")
        return run_sandbox(code, kwargs)
    except Exception as e:
        return {"success": False, "error": str(e)}
