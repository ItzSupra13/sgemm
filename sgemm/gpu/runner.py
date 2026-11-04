"""GPU unit (Colab T4).

Kernels benchmarked, all through the same shared loop in common/bench.py:
  every sgemm/gpu/kernels/*.cu   compiled by build.py, called through ctypes (auto-discovered)
      cublas_fp32                cuBLAS sgemm: the "100%" reference for the fp32 ladder
      00_naive, ...              your hand-written kernels
  torch_fp32 / torch_fp16        torch.matmul baselines (fp16 = tensor-core ceiling for later WMMA work)

Contract shared with the other units:  build(cfg)  and  run_benchmark(cfg) -> dict
torch is imported lazily so `python run.py cpu ...` works on machines without it.
"""
from __future__ import annotations

import numpy as np

from ..common.bench import Kernel, run_python_backend
from ..common.env import check_expected_device, collect
from . import build as cuda_build


def build(cfg: dict) -> None:
    cuda_build.build(cfg)


def cuda_kernels(torch, arch: str) -> list[Kernel]:
    """One Kernel per compiled .cu file. Missing .so means `build` was not run: say so clearly."""
    out = []
    for src in cuda_build.sources():
        if not cuda_build.lib_path(src).exists():
            raise SystemExit(f"{src.name} is not built. Run: python run.py gpu build")
        lib = cuda_build.load(src)
        name = src.stem

        def setup(A, B, lib=lib, name=name):
            n = A.shape[0]
            a = torch.from_numpy(A).to("cuda")
            b = torch.from_numpy(B).to("cuda")
            # NaN-filled so a kernel that forgets to write part of C fails the correctness check.
            c = torch.full_like(a, float("nan"))
            pa, pb, pc = a.data_ptr(), b.data_ptr(), c.data_ptr()

            def run():
                rc = lib.launch(n, pa, pb, pc)
                if rc != 0:
                    raise RuntimeError(f"{name}: launch() returned error code {rc}")
                return c
            run._keepalive = (a, b)   # the raw pointers above must outlive the tensors
            return run

        out.append(Kernel(name, f"fp32 {arch}", setup, lambda o: o.float().cpu().numpy(), tol=1e-4))
    return out


def kernels(cfg: dict) -> list[Kernel]:
    import torch

    # No TF32 on a T4, but make it explicit so an Ampere+ GPU can't silently use it.
    torch.backends.cuda.matmul.allow_tf32 = False

    def torch_matmul(dtype):
        def setup(A, B):
            a = torch.from_numpy(A).to("cuda", dtype)
            b = torch.from_numpy(B).to("cuda", dtype)
            return lambda: torch.matmul(a, b)
        return setup

    fetch = lambda out: out.float().cpu().numpy()
    fp16_round = lambda x: x.astype(np.float16).astype(np.float32)
    arch = cfg.get("arch") or cuda_build.DEFAULT_ARCH
    return cuda_kernels(torch, arch) + [
        Kernel("torch_fp32", "fp32", torch_matmul(torch.float32), fetch, tol=1e-4),
        Kernel("torch_fp16", "fp16", torch_matmul(torch.float16), fetch, tol=1e-2,
               ref_key="fp16", ref_round=fp16_round),
    ]


def run_benchmark(cfg: dict) -> dict:
    info = collect("gpu")
    if not info["gpu"]:
        raise SystemExit("No NVIDIA GPU visible (nvidia-smi failed). In Colab: Runtime > Change "
                         "runtime type > T4 GPU, then reconnect.")
    check_expected_device(info, cfg)
    import torch
    torch.cuda.init()
    return run_python_backend("gpu", info, kernels(cfg), lambda out: torch.cuda.synchronize(), cfg)
