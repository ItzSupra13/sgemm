#!/usr/bin/env python3
"""The single entry point (what `modal run` would have been).

    python run.py <backend> <action> [options]

    backend: cpu | gpu | tpu
    action:  info   print the environment this would run on
             build  compile whatever the unit needs (cpu: make, gpu: nvcc on kernels/*.cu, tpu: nothing)
             run    benchmark, writing sgemm/<backend>/results/<tag>/*.csv   (resumable)
             plot   read those CSVs, write sgemm/<backend>/plots/<tag>/*.png
             all    build + run + plot

Examples:
    python run.py cpu all --quick
    python run.py gpu run --sizes 512,1024,2048 --reps 20
    python run.py gpu run --kernels cublas_fp32,00_naive --sizes 1024
    python run.py tpu plot --tag colab-TPU_v5_lite
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

QUICK = {"cpu": {"sizes": "64,128,256,512,1024", "reps": 3},
         "gpu": {"sizes": "256,1024,2048", "reps": 5},
         "tpu": {"sizes": "256,1024,2048", "reps": 5}}


def build_cfg(args) -> dict:
    cfg = json.loads((ROOT / "configs" / "default.json").read_text())[args.backend]
    if args.quick:
        cfg.update(QUICK[args.backend])
    if args.sizes:
        cfg["sizes"] = args.sizes
    if args.reps:
        cfg["reps"] = args.reps
    cfg.update(tag=args.tag, fresh=args.fresh, any_device=args.any_device, kernels=args.kernels, arch=args.arch, torch=args.torch)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", choices=["cpu", "gpu", "tpu"])
    ap.add_argument("action", choices=["info", "build", "run", "plot", "all"])
    ap.add_argument("--quick", action="store_true", help="small sizes and few reps, to check the pipeline")
    ap.add_argument("--sizes", help="comma list, e.g. 512,1024,2048")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--kernels", help="gpu/tpu: comma list of names or globs, e.g. cublas_fp32,00_*")
    ap.add_argument("--arch", help="gpu: nvcc arch (default sm_75 = T4)")
    ap.add_argument("--tag", help="results subfolder name (default: <platform>-<device>)")
    ap.add_argument("--fresh", action="store_true", help="discard existing results for this tag first")
    ap.add_argument("--any-device", action="store_true", help="skip the expected-device check")
    ap.add_argument("--torch", action="store_true", help="gpu: also run/plot the torch.matmul baselines (off by default)")
    args = ap.parse_args()
    cfg = build_cfg(args)

    if args.action == "info":
        from sgemm.common.env import collect
        print(json.dumps(collect(args.backend), indent=2, default=str))
        return

    runner = importlib.import_module(f"sgemm.{args.backend}.runner")  # lazy: only this unit's deps load
    res = None
    if args.action in ("build", "all"):
        runner.build(cfg)
    if args.action in ("run", "all"):
        res = runner.run_benchmark(cfg)
        if res.get("failed"):
            print(f"WARNING: these binaries failed: {res['failed']}")
        print(f"\nResults: {res['results_dir']}")
    if args.action in ("plot", "all"):
        from sgemm.common.plotting import make_plots
        tag = res["tag"] if res else args.tag
        for p in make_plots(args.backend, tag):
            print("Plot:", p)


if __name__ == "__main__":
    main()
