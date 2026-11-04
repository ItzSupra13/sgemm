"""Environment / device introspection.

The only place that knows which platform we run on (Colab, Modal, local) is detect_platform(),
and it is used for logging and for naming result folders. Kernel and benchmark logic must
never branch on it.
"""
from __future__ import annotations

import datetime
import os
import platform as _platform
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sh(cmd: list[str], timeout: int = 15) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def slug(s: str, limit: int = 60) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:limit] or "unknown"


def detect_platform() -> str:
    if os.environ.get("MODAL_TASK_ID"):
        return "modal"
    if any(k.startswith("COLAB_") for k in os.environ):
        return "colab"
    return "local"


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return _platform.processor() or "unknown-cpu"


def cpu_flags() -> set[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def gpu_info() -> dict:
    out = _sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
               "--format=csv,noheader"])
    if not out:
        return {}
    name, driver, mem, clk = [x.strip() for x in out.splitlines()[0].split(",")]
    info = {"name": name, "driver": driver, "memory": mem, "max_sm_clock": clk}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except Exception as e:  # torch missing is fine for info-only calls
        info["torch"] = f"unavailable ({type(e).__name__})"
    return info


def tpu_info() -> dict:
    try:
        import jax
        devs = jax.devices()
        return {"jax": jax.__version__, "jax_backend": jax.default_backend(),
                "device_kind": devs[0].device_kind, "device_count": len(devs),
                "platform": devs[0].platform}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def collect(backend: str) -> dict:
    info = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": detect_platform(),
        "backend": backend,
        "python": sys.version.split()[0],
        "os": _platform.platform(),
        "git_commit": _sh(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_sh(["git", "status", "--porcelain"])),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
    }
    if backend == "cpu":
        info["device"] = info["cpu_model"]
    elif backend == "gpu":
        info["gpu"] = gpu_info()
        info["device"] = info["gpu"].get("name", "no-gpu")
    elif backend == "tpu":
        info["tpu"] = tpu_info()
        t = info["tpu"]
        info["device"] = t.get("device_kind", "no-tpu") if t.get("platform") == "tpu" else "no-tpu"
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return info


def check_expected_device(info: dict, cfg: dict) -> None:
    """Fail loudly if Colab handed us different hardware than the experiment assumes."""
    want = (cfg.get("expect_device") or "").lower()
    if not want or cfg.get("any_device"):
        return
    have = info["device"].lower()
    if want not in have:
        raise SystemExit(
            f"Expected a device matching {cfg['expect_device']!r} but found {info['device']!r}.\n"
            "Change the Colab runtime type, or pass --any-device to run anyway "
            "(results then land under a different tag, so nothing gets mixed up)."
        )
