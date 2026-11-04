"""Backend-agnostic benchmark loop for the Python backends (gpu/torch, tpu/jax).

The CPU backend uses the C harness (sgemm/cpu/bench.c) and writes the same CSV schema, so one
plotting module serves all three units.

A Kernel is described by four small callables, so this file never imports torch or jax:

    setup(A_np, B_np) -> run        move data to the device, return a zero-arg callable that
                                    performs ONE matmul and returns the (device) result
    fetch(out)        -> np.ndarray result back on the host as float32
    sync(out)                       block until the device has finished (backend-wide)
    ref_round(x)      -> np.ndarray round inputs the way the kernel does (fp16/bf16) so the
                                    reference sees the same numbers and only real error is left
"""
from __future__ import annotations

import fnmatch
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .results import COMMON_FIELDS, append_row, read_csv, results_dir, write_json, default_tag


@dataclass
class Kernel:
    name: str
    build: str                                   # goes in the CSV "build" column (dtype/precision)
    setup: Callable[[np.ndarray, np.ndarray], Callable[[], Any]]
    fetch: Callable[[Any], np.ndarray]
    tol: float                                   # max |C-R| / max |R|
    ref_key: str = "fp32"                        # kernels with the same key share a reference
    ref_round: Callable[[np.ndarray], np.ndarray] | None = None
    max_n: int | None = None


def make_inputs(n: int, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, (n, n)).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, (n, n)).astype(np.float32)
    return A, B


def time_it(run: Callable[[], Any], sync: Callable[[Any], None], reps: int,
            min_sample_s: float = 0.05, max_inner: int = 100000) -> list[float]:
    """Seconds per call, one entry per rep, sorted. `run` must already have been called once
    (warmup / jit compile / lazy init). Several calls per sample so small sizes beat timer and
    dispatch noise; one sync per sample so async devices are measured, not just enqueued."""
    t0 = time.perf_counter()
    out = run()
    sync(out)
    t_est = time.perf_counter() - t0
    inner = min(max_inner, max(1, math.ceil(min_sample_s / max(t_est, 1e-9))))
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            out = run()
        sync(out)
        samples.append((time.perf_counter() - t0) / inner)
    samples.sort()
    return samples


def run_sweep(kernels: list[Kernel], sizes: list[int], sync: Callable[[Any], None],
              out_dir: Path, reps: int = 10, log: Callable[[str], None] = print) -> list[dict]:
    """Sizes outermost (inputs and references are built once per n), one CSV per kernel.
    Resumable: (kernel, n) pairs already present in a CSV are skipped."""
    rows_out: list[dict] = []
    for n in sizes:
        todo = []
        for k in kernels:
            csv_path = out_dir / f"{k.name}.csv"
            have = {r["n"] for r in read_csv(csv_path)} if csv_path.exists() else set()
            if (k.max_n and n > k.max_n) or n in have:
                continue
            todo.append(k)
        if not todo:
            continue

        A, B = make_inputs(n)
        refs: dict[str, np.ndarray] = {}
        flops = 2.0 * n * n * n
        for k in todo:
            if k.ref_key not in refs:
                Ar = k.ref_round(A) if k.ref_round else A
                Br = k.ref_round(B) if k.ref_round else B
                refs[k.ref_key] = (Ar @ Br).astype(np.float64)
            R = refs[k.ref_key]

            run = k.setup(A, B)
            out = run()
            sync(out)                                    # warmup / compile
            C = k.fetch(out).astype(np.float64)
            bad = not np.all(np.isfinite(C))
            rel = math.inf if bad else float(np.max(np.abs(C - R)) / np.max(np.abs(R)))
            ok = (not bad) and rel <= k.tol

            t = time_it(run, sync, reps)
            med, best = t[len(t) // 2], t[0]
            row = {"kernel": k.name, "build": k.build, "n": n,
                   "median_gflops": round(flops / med * 1e-9, 2),
                   "best_gflops": round(flops / best * 1e-9, 2),
                   "median_ms": round(med * 1e3, 3),
                   "max_rel_err": f"{rel:.2e}", "ok": int(ok)}
            append_row(out_dir / f"{k.name}.csv", row)
            rows_out.append(row)
            log(f"  {k.name:<20} n={n:<5} {row['median_gflops']:>10.1f} GFLOPS  "
                f"{row['median_ms']:>9.3f} ms  err={rel:.1e}  {'ok' if ok else 'FAILED CHECK'}")
            del run, out
    return rows_out


def run_python_backend(backend: str, info: dict, kernels: list[Kernel],
                       sync: Callable[[Any], None], cfg: dict) -> dict:
    tag = cfg.get("tag") or default_tag(info)
    out_dir = results_dir(backend, tag, fresh=bool(cfg.get("fresh")))
    write_json(out_dir / "env.json", {**info, "config": {k: v for k, v in cfg.items()}})
    sizes = [int(s) for s in str(cfg["sizes"]).split(",") if s.strip()]
    pats = [x for x in str(cfg.get("kernels") or "").split(",") if x.strip()]
    if pats:
        kernels = [k for k in kernels if any(fnmatch.fnmatch(k.name, pt.strip()) for pt in pats)]
        if not kernels:
            raise SystemExit(f"--kernels {cfg['kernels']!r} matched nothing")
    print(f"[{backend}] device={info['device']!r} tag={tag} sizes={sizes} reps={cfg['reps']}")
    rows = run_sweep(kernels, sizes, sync, out_dir, reps=int(cfg["reps"]))
    return {"backend": backend, "tag": tag, "results_dir": str(out_dir), "rows": rows, "env": info}
