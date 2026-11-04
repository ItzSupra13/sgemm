"""Plots for any unit, from its results/<tag>/*.csv. Runs anywhere (laptop, Colab, Modal)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter

from .results import latest_tag, plots_dir, read_csv, results_dir

# Kernel whose GFLOPS are "100%" in the percent-of-vendor plot.
REFERENCE = {"cpu": "openblas", "gpu": "cublas_fp32", "tpu": "xla_fp32_highest"}
OPTIONAL_PREFIX = {"gpu": "torch_"}

def _label(row: dict) -> str:
    build = str(row.get("build") or "").strip()
    short = build.split()[0] if build and build != "unknown" else ""
    return f"{row['kernel']} [{short}]" if short else str(row["kernel"])


def load_series(backend: str, tag: str, include_optional: bool = False) -> dict[str, list[dict]]:
    series: dict[str, list[dict]] = {}
    source: dict[str, str] = {}
    for p in sorted(results_dir(backend, tag, create=False).glob("*.csv")):
        if p.name == "peak.csv":
            continue
        if not include_optional and p.name.startswith(OPTIONAL_PREFIX.get(backend, "\0")):
            continue
        for r in read_csv(p):
            label = _label(r)
            # Two binaries with the same kernel+flags (e.g. 02_ikj and 02_ikj_O3) would otherwise
            # merge into one zigzag series: keep the first file, skip the duplicate.
            if source.setdefault(label, p.name) != p.name:
                print(f"note: {p.name} has the same label as {source[label]} ({label}); not plotted")
                break
            series.setdefault(label, []).append(r)
    for rows in series.values():
        rows.sort(key=lambda r: r["n"])
    return series


def _style_n_axis(ax, ns):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(ns))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("matrix size n (n x n x n)")
    ax.grid(True, which="major", alpha=0.3)


def make_plots(backend: str, tag: str | None = None, include_optional: bool = False) -> list[str]:
    tag = tag or latest_tag(backend)
    if not tag:
        raise SystemExit(f"No results for '{backend}' yet. Run: python run.py {backend} run")
    series = load_series(backend, tag, include_optional)
    if not series:
        raise SystemExit(f"No CSV rows under {results_dir(backend, tag, create=False)}")
    out = plots_dir(backend, tag)
    paths: list[str] = []
    ns = {r["n"] for rows in series.values() for r in rows}

    peak_gf, peak_rows = None, []
    peak_csv = results_dir(backend, tag, create=False) / "peak.csv"
    if peak_csv.exists():
        peak_rows = read_csv(peak_csv)
        if peak_rows:
            peak_gf = max(r["gflops"] for r in peak_rows)

    # 1) GFLOPS vs n
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, rows in series.items():
        good = [r for r in rows if r["ok"]]
        bad = [r for r in rows if not r["ok"]]
        ax.plot([r["n"] for r in good], [r["median_gflops"] for r in good], marker="o", label=label)
        if bad:
            print(f"WARNING: {label}: {len(bad)} size(s) failed the correctness check (shown as red x)")
            ax.scatter([r["n"] for r in bad], [r["median_gflops"] for r in bad],
                       marker="x", color="red", zorder=5)
    if peak_gf:
        ax.axhline(peak_gf, ls="--", color="gray", label=f"measured 1-core FMA peak ({peak_gf:.0f})")
    ax.set_yscale("log")
    ax.set_ylabel("GFLOPS (median)")
    _style_n_axis(ax, ns)
    ax.set_title(f"{backend.upper()} SGEMM: {tag}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "gflops_vs_n.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # 2) percent of vendor library
    ref_name = REFERENCE.get(backend)
    ref_label = next((l for l in series if l.split(" [")[0] == ref_name), None)
    if ref_label:
        ref = {r["n"]: r["median_gflops"] for r in series[ref_label] if r["ok"]}
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, rows in series.items():
            if label == ref_label:
                continue
            pts = [(r["n"], 100.0 * r["median_gflops"] / ref[r["n"]])
                   for r in rows if r["ok"] and r["n"] in ref]
            if pts:
                ax.plot(*zip(*pts), marker="o", label=label)
        ax.axhline(100, ls="--", color="gray")
        ax.set_ylabel(f"% of {ref_label}")
        _style_n_axis(ax, ns)
        ax.set_title(f"{backend.upper()} SGEMM, percent of {ref_label}: {tag}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out / "pct_of_vendor.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(str(p))

    # 3) FMA-chain peak (cpu only: exists when peak.csv does)
    if peak_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["chains"] for r in peak_rows], [r["gflops"] for r in peak_rows], marker="o")
        ax.set_xlabel("independent FMA chains")
        ax.set_ylabel("GFLOPS (1 core, AVX2)")
        ax.set_title(f"Peak FMA throughput: {tag}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out / "peak.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(str(p))
    return paths
