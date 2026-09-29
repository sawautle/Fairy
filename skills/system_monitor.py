"""System monitoring skill for Fairy 2.0.
Merges background metric checks (your working code) with a query-based interface.
"""
import ctypes
import json
import platform
import subprocess
import time

try:
    import psutil
    _PSUTIL = True
except Exception:
    _PSUTIL = False

try:
    import gputil
    _GPUTIL = True
except Exception:
    _GPUTIL = False

_OS = platform.system()

# ── NVML DLL cache ──────────────────────────────────────────
_nvml_lib = None
_nvml_ok = None

DEFAULT_THRESHOLDS = {
    "cpu": 90.0,
    "ram": 90.0,
    "temp": 85.0,
    "gpu": 95.0,
}

_COOLDOWN = 300
_CPU_STREAK = 3


def _nvml_gpu() -> float:
    global _nvml_lib, _nvml_ok
    if _nvml_ok is False:
        return -1.0
    try:
        class _Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        if _nvml_lib is None:
            if _OS == "Windows":
                candidates = ("nvml", r"C:\Windows\System32\nvml.dll")
                _load = ctypes.WinDLL
            else:
                candidates = ("libnvidia-ml.so.1", "libnvidia-ml.so", "libnvidia-ml.dylib")
                _load = ctypes.CDLL
            for name in candidates:
                try:
                    lib = _load(name)
                    lib.nvmlInit_v2()
                    _nvml_lib = lib
                    break
                except Exception:
                    continue

        if _nvml_lib is None:
            _nvml_ok = False
            return -1.0

        dev = ctypes.c_void_p()
        _nvml_lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev))
        u = _Util()
        _nvml_lib.nvmlDeviceGetUtilizationRates(dev, ctypes.byref(u))
        _nvml_ok = True
        return float(u.gpu)
    except Exception:
        _nvml_ok = False
        return -1.0


def _get_gpu_usage() -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
    except Exception:
        pass
    return _nvml_gpu()


def _get_cpu_temp() -> float:
    try:
        temps = psutil.sensors_temperatures()
        for name in ["coretemp", "k10temp", "cpu_thermal", "acpitz",
                     "cpu-thermal", "zenpower", "it8688"]:
            if name in temps and temps[name]:
                return temps[name][0].current
        for entries in temps.values():
            if entries:
                return entries[0].current
    except Exception:
        pass

    if _OS == "Windows":
        try:
            import wmi
            w = wmi.WMI(namespace="root/wmi")
            tz = w.MSAcpi_ThermalZoneTemperature()
            if tz:
                return (tz[0].CurrentTemperature / 10.0) - 273.15
        except Exception:
            pass
    return -1.0


def _get_nvidia_gpu():
    """Use nvidia-smi for accurate NVIDIA GPU info."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split("\n")
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "name": parts[0],
                    "load_percent": float(parts[1]) if parts[1] else 0.0,
                    "memory_used_mb": float(parts[2]) if parts[2] else 0.0,
                    "memory_total_mb": float(parts[3]) if parts[3] else 0.0,
                    "temperature_c": float(parts[4]) if parts[4] else 0.0,
                })
        return gpus
    except Exception:
        return None


def _get_gputil_gpus():
    if not _GPUTIL:
        return None
    try:
        gpus = gputil.getGPUs()
        return [
            {
                "name": gpu.name,
                "load_percent": gpu.load * 100,
                "memory_used_mb": gpu.memoryUsed,
                "memory_total_mb": gpu.memoryTotal,
                "temperature_c": gpu.temperature,
            }
            for gpu in gpus
        ]
    except Exception:
        return None


def _get_battery():
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return None
        return {
            "percent": battery.percent,
            "is_charging": battery.power_plugged,
            "time_left_minutes": battery.secsleft // 60 if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None,
        }
    except Exception:
        return None


def _get_temps():
    try:
        temps = psutil.sensors_temperatures()
        result = {}
        for name, entries in temps.items():
            result[name] = [{"label": e.label, "current_c": e.current} for e in entries]
        return result
    except Exception:
        return {}


def _top_processes(n=5):
    try:
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent"]):
            try:
                procs.append(p.info)
            except Exception:
                pass
        procs.sort(key=lambda x: x.get("cpu_percent", 0), reverse=True)
        return procs[:n]
    except Exception:
        return []


def get_system_status() -> dict:
    """Snapshot of current system metrics for background monitoring."""
    cpu = psutil.cpu_percent(interval=0.2)
    ram = psutil.virtual_memory()
    temp = _get_cpu_temp()
    gpu = _get_gpu_usage()

    boot_time = psutil.boot_time()
    uptime_secs = time.time() - boot_time
    uptime_h = int(uptime_secs // 3600)
    uptime_m = int((uptime_secs % 3600) // 60)

    return {
        "cpu_percent": round(cpu, 1),
        "ram_percent": round(ram.percent, 1),
        "ram_used_gb": round(ram.used / 1024 ** 3, 1),
        "ram_total_gb": round(ram.total / 1024 ** 3, 1),
        "cpu_temp_c": round(temp, 1) if temp > 0 else None,
        "gpu_percent": round(gpu, 1) if gpu >= 0 else None,
        "uptime": f"{uptime_h}h {uptime_m}m",
        "process_count": len(psutil.pids()),
    }


def get_system_summary() -> dict:
    """Backward-compatible alias used by the controller."""
    return get_system_status()


def get_gpu_info() -> dict:
    """Return the latest GPU summary in the format expected by legacy callers."""
    gpus = _get_nvidia_gpu() or _get_gputil_gpus() or []
    if not gpus:
        gpu = _get_gpu_usage()
        return {"gpus": [], "gpu_percent": round(gpu, 1) if gpu >= 0 else None}
    return {"gpus": gpus}


def system_monitor(query: str = "summary") -> str:
    """
    Unified system monitor dispatcher.
    query: summary | cpu | memory | gpu | disk | network | battery | temp | processes
    """
    if not _PSUTIL:
        return json.dumps({"error": "psutil not installed. Run: pip install psutil"})

    query = str(query).lower().strip()
    result = {"platform": _OS, "platform_version": platform.version()}

    if query in ("summary", "cpu"):
        result["cpu"] = {
            "percent": psutil.cpu_percent(interval=0.5),
            "cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "freq_mhz": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "per_core": psutil.cpu_percent(interval=0.1, percpu=True),
        }

    if query in ("summary", "memory", "ram"):
        mem = psutil.virtual_memory()
        result["memory"] = {
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "percent": mem.percent,
        }
        swap = psutil.swap_memory()
        result["swap"] = {
            "total_gb": round(swap.total / (1024**3), 2),
            "used_gb": round(swap.used / (1024**3), 2),
            "percent": swap.percent,
        }

    if query in ("summary", "gpu"):
        gpus = _get_nvidia_gpu()
        if gpus is None:
            gpus = _get_gputil_gpus()
        if gpus is None:
            gpus = []
        result["gpus"] = gpus

    if query in ("summary", "disk"):
        disks = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                disks.append({
                    "device": part.device,
                    "mountpoint": part.mountpoint,
                    "total_gb": round(usage.total / (1024**3), 2),
                    "used_gb": round(usage.used / (1024**3), 2),
                    "percent": usage.percent,
                })
            except Exception:
                pass
        result["disks"] = disks

    if query in ("summary", "network"):
        net = psutil.net_io_counters()
        result["network"] = {
            "bytes_sent_mb": round(net.bytes_sent / (1024**2), 2),
            "bytes_recv_mb": round(net.bytes_recv / (1024**2), 2),
        }

    if query in ("summary", "battery"):
        bat = _get_battery()
        if bat:
            result["battery"] = bat

    if query in ("summary", "temp", "temperatures"):
        result["temperatures"] = _get_temps()

    if query in ("summary", "processes"):
        result["top_processes"] = _top_processes(5)

    return json.dumps(result, indent=2)


class SystemMonitor:
    """
    Stateful background monitor with voice alert support.
    Cooldown state persists across session reconnections.
    """

    def __init__(self, thresholds: dict = None):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._last_alert = {}
        self._cpu_streak = 0

    def _can_alert(self, key: str) -> bool:
        return (time.monotonic() - self._last_alert.get(key, 0)) > _COOLDOWN

    def _record(self, key: str):
        self._last_alert[key] = time.monotonic()

    def check(self) -> str:
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            temp = _get_cpu_temp()
            gpu = _get_gpu_usage()
        except Exception:
            return None

        alerts = []

        if cpu >= self.thresholds["cpu"]:
            self._cpu_streak += 1
            if self._cpu_streak >= _CPU_STREAK and self._can_alert("cpu"):
                alerts.append(
                    f"[SYSTEM_ALERT] CPU usage has been critically high ({cpu:.0f}%) "
                    "for several seconds. Warn the user in their language and suggest "
                    "closing heavy applications."
                )
                self._record("cpu")
                self._cpu_streak = 0
        else:
            self._cpu_streak = 0

        if ram >= self.thresholds["ram"] and self._can_alert("ram"):
            alerts.append(
                f"[SYSTEM_ALERT] RAM is at {ram:.0f}% — nearly exhausted. "
                "Warn the user in their language and suggest freeing memory."
            )
            self._record("ram")

        if temp > 0 and temp >= self.thresholds["temp"] and self._can_alert("temp"):
            alerts.append(
                f"[SYSTEM_ALERT] CPU temperature is {temp:.0f}°C — above the safe limit. "
                "Warn the user in their language and advise reducing system load "
                "or checking cooling."
            )
            self._record("temp")

        if gpu >= 0 and gpu >= self.thresholds["gpu"] and self._can_alert("gpu"):
            alerts.append(
                f"[SYSTEM_ALERT] GPU load is at {gpu:.0f}%. "
                "Briefly inform the user in their language."
            )
            self._record("gpu")

        return " ".join(alerts) if alerts else None


# ═══════════════════════════════════════════════════════════════════
# Edge TTS Voice Alert Support
# ═══════════════════════════════════════════════════════════════════

_EDGE_TTS_VOICE = "en-US-AriaNeural"  # Default Edge TTS voice — same as Fairy
_EDGE_TTS_RATE = "+0%"
_EDGE_TTS_VOLUME = "+0%"


def _speak_with_edge_tts(text: str) -> bool:
    """Speak an alert using Edge TTS (same voice as Fairy)."""
    try:
        import asyncio
        import edge_tts
        import tempfile
        import subprocess

        async def _run():
            communicate = edge_tts.Communicate(
                text,
                voice=_EDGE_TTS_VOICE,
                rate=_EDGE_TTS_RATE,
                volume=_EDGE_TTS_VOLUME,
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            await communicate.save(tmp_path)
            # Play the audio (Windows)
            subprocess.Popen(
                ["start", "", tmp_path],
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True

        return asyncio.run(_run())
    except Exception:
        return False


def speak_alert(text: str, use_tts: bool = True) -> str:
    """
    Speak a system alert using Fairy's Edge TTS voice.
    Falls back to returning the text if TTS is unavailable.

    DISABLED 2026-08-29: Voice alerts fire every time RAM/GPU cross their
    threshold (with a 300s cooldown), which plays an MP3 in the user's
    default media player — annoying and repetitive. The threshold-trip
    itself is still logged via _debug in the background-monitor loop, so
    telemetry is preserved. Re-enable by restoring the body below.
    """
    return text
    # if use_tts and _speak_with_edge_tts(text):
    #     return f"[SPOKEN] {text}"
    # return text