"""Where results and plots live, and how to read/write them.

Each unit owns its own folders:

    sgemm/<backend>/results/<tag>/*.csv, env.json, *.log
    sgemm/<backend>/plots/<tag>/*.png

<tag> defaults to "<platform>-<device>" (e.g. colab-Tesla_T4), so numbers from different
machines never overwrite or mix with each other.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from .env import slug

PKG_DIR = Path(__file__).resolve().parents[1]

# Columns every backend writes (the C harness writes extra cpu/mhz_end columns after these).
COMMON_FIELDS = ["kernel", "build", "n", "median_gflops", "best_gflops",
                 "median_ms", "max_rel_err", "ok"]


def unit_dir(backend: str) -> Path:
    return PKG_DIR / backend


def default_tag(info: dict) -> str:
    return slug(f"{info['platform']}-{info['device']}")


def results_dir(backend: str, tag: str, create: bool = True, fresh: bool = False) -> Path:
    p = unit_dir(backend) / "results" / tag
    if fresh and p.exists():
        shutil.rmtree(p)
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def plots_dir(backend: str, tag: str) -> Path:
    p = unit_dir(backend) / "plots" / tag
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_tag(backend: str) -> str | None:
    root = unit_dir(backend) / "results"
    if not root.exists():
        return None
    dirs = [d for d in root.iterdir() if d.is_dir() and any(d.glob("*.csv"))]
    return max(dirs, key=lambda d: d.stat().st_mtime).name if dirs else None


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _num(v: str):
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    return v


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def append_row(path: Path, row: dict, fields: list[str] = COMMON_FIELDS) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)
