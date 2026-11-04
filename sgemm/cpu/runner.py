"""CPU unit: builds the C harness with make, runs every binary pinned to one core,
and drops one CSV per binary into sgemm/cpu/results/<tag>/.

Contract shared with the other units:  build(cfg)  and  run_benchmark(cfg) -> dict
"""
from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import time
from pathlib import Path

from ..common.env import cpu_flags, collect
from ..common.results import default_tag, read_csv, results_dir, write_json

CPU_DIR = Path(__file__).resolve().parent


def parse_sizes(s) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


def preflight() -> None:
    missing = [t for t in ("gcc", "make", "taskset") if shutil.which(t) is None]
    if missing:
        raise SystemExit(f"Missing tools: {', '.join(missing)}. On Colab/Debian/Ubuntu: "
                         "apt-get install -y build-essential util-linux")
    r = subprocess.run(["gcc", "-E", "-x", "c", "-"], input="#include <cblas.h>\n",
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("cblas.h not found. On Colab/Debian/Ubuntu: apt-get install -y libopenblas-dev "
                         "(Arch: pacman -S openblas)")


def build(cfg: dict) -> None:
    preflight()
    if cfg.get("fresh"):
        subprocess.run(["make", "clean"], cwd=CPU_DIR, check=True)
    subprocess.run(["make", "-j2"], cwd=CPU_DIR, check=True)


def pick_cpu(cfg: dict) -> int:
    """Pin to cfg['cpu_pin'] if this machine has it (laptop: 2), else the first allowed CPU."""
    allowed = sorted(os.sched_getaffinity(0))
    want = cfg.get("cpu_pin")
    return want if want in allowed else allowed[0]


def max_n_for(name: str, cfg: dict) -> int:
    caps = cfg.get("max_n", {})
    for pattern, cap in caps.items():
        if pattern != "default" and fnmatch.fnmatch(name, pattern):
            return int(cap)
    return int(caps.get("default", 1 << 30))


def list_binaries() -> list[str]:
    bindir = CPU_DIR / "bin"
    if not bindir.is_dir():
        return []
    return sorted(p.name for p in bindir.iterdir() if p.is_file() and p.name != "peak")


def _run_to_file(cmd: list[str], out_path: Path, log_path: Path, env: dict, append: bool) -> int:
    """stdout (CSV) goes straight to the file; stderr (env info) is echoed and logged."""
    with open(out_path, "a" if append else "w") as fout, open(log_path, "a" if append else "w") as flog:
        p = subprocess.Popen(cmd, cwd=CPU_DIR, env=env, stdout=fout,
                             stderr=subprocess.PIPE, text=True)
        for line in p.stderr:
            print("   ", line, end="", flush=True)
            flog.write(line)
        return p.wait()


def run_benchmark(cfg: dict) -> dict:
    preflight()
    info = collect("cpu")
    tag = cfg.get("tag") or default_tag(info)
    out = results_dir("cpu", tag, fresh=bool(cfg.get("fresh")))
    cpu = pick_cpu(cfg)
    info["pinned_cpu"] = cpu
    write_json(out / "env.json", {**info, "config": cfg})

    bins = list_binaries()
    if not bins:
        raise SystemExit("No binaries in sgemm/cpu/bin. Run: python run.py cpu build")

    sizes, reps = parse_sizes(cfg["sizes"]), int(cfg["reps"])
    env = {**os.environ, "CPU": str(cpu)}
    print(f"[cpu] device={info['device']!r} tag={tag} pinned_cpu={cpu} sizes={sizes} reps={reps}")

    failed, files = [], []
    for name in bins:
        cap = max_n_for(name, cfg)
        want = [n for n in sizes if n <= cap]
        csv_path, log_path = out / f"{name}.csv", out / f"{name}.log"
        have = {r["n"] for r in read_csv(csv_path)} if csv_path.exists() else set()
        todo = [n for n in want if n not in have]
        if not todo:
            print(f"[cpu] {name}: nothing to do (cap n<={cap}, already have {sorted(have)})")
            continue
        append = bool(have)
        cmd = ["bash", "run.sh", f"bin/{name}", "-n", ",".join(map(str, todo)), "-r", str(reps)]
        if append:
            cmd.append("-H")  # no second header when resuming
        print(f"[cpu] {name}: n={todo}", flush=True)
        t0 = time.time()
        rc = _run_to_file(cmd, csv_path, log_path, env, append)
        print(f"[cpu] {name}: {'done' if rc == 0 else f'FAILED rc={rc}'} in {time.time() - t0:.0f}s")
        if rc != 0:
            failed.append(name)
        files.append(str(csv_path))

    peak_csv = out / "peak.csv"
    if cfg.get("peak", True) and not peak_csv.exists() and (CPU_DIR / "bin" / "peak").exists():
        if "avx2" in cpu_flags() and "fma" in cpu_flags():
            print("[cpu] peak FMA microbenchmark")
            if _run_to_file(["bash", "run.sh", "bin/peak"], peak_csv, out / "peak.log", env, False) != 0:
                failed.append("peak")
        else:
            print("[cpu] skipping peak: CPU lacks AVX2/FMA")

    return {"backend": "cpu", "tag": tag, "results_dir": str(out), "files": files,
            "failed": failed, "env": info}
