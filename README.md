# SGEMM ladder on CPU, GPU (T4) and TPU

Colab-first, laptop-friendly, Modal-ready later. Each unit (`cpu`, `gpu`, `tpu`) owns its own
`results/` and `plots/` folders; all logic lives here in git and the Colab notebooks are thin launchers.

```
run.py                      python run.py <cpu|gpu|tpu> <info|build|run|plot|all>
configs/default.json        sizes / reps / per-kernel size caps / expected device
sgemm/common/               env.py  results.py  bench.py  plotting.py   (no Colab/Modal/torch/jax imports)
sgemm/cpu/                  C harness (bench.c, kernels/*.c, Makefile, run.sh) + runner.py
sgemm/gpu/                  runner.py (torch/cuBLAS baselines now; your CUDA kernels next)
sgemm/tpu/                  runner.py (XLA baselines now; your Pallas kernels next)
   <unit>/results/<tag>/    one CSV per kernel (+ env.json, logs)     <tag> = <platform>-<device>
   <unit>/plots/<tag>/      gflops_vs_n.png, pct_of_vendor.png (+ peak.png on cpu)
notebooks/                  colab_{cpu,gpu,tpu}.ipynb   (regenerate: python tools/make_notebooks.py)
tests/test_common.py        numpy-only smoke test of the shared loop + plotting
```

## Quick start

```bash
# laptop (CPU)
python run.py cpu all --quick          # pipeline check, ~1 min
python run.py cpu all                  # full sweep

# Colab: open notebooks/colab_gpu.ipynb (or _tpu / _cpu), edit REPO_URL, Run all
```

No `pip install` is needed (run from the repo root). Don't pip-install torch/jax on Colab: they come
with the GPU/TPU runtimes.

## Rules that keep the Modal port cheap

1. Library code never imports `google.colab` or `modal`, and never branches on the platform.
2. `run_benchmark(cfg) -> dict` is the single entry; launchers (CLI, notebook, `modal_app.py`) just call it.
3. Paths come from `common/results.py`, never hardcoded (`/content/...`).
4. Builds are explicit (`python run.py <unit> build`) so Modal can run them at image-build time.
5. Backends import torch/jax lazily, so a unit's missing dependency never breaks the others.

## Notes

- Runs are resumable: finished (kernel, n) pairs are skipped. `--fresh` starts over, `--tag NAME` keeps a separate set.
- `gpu`/`tpu` refuse to run on the wrong hardware (`expect_device` in the config) unless `--any-device`.
- `cpu_pin` in the config is the laptop's logical CPU (2); on a machine without it the runner pins to the first allowed CPU.
- Colab VMs do not expose cpufreq, so `mhz_end` is `-1` there; use the peak benchmark's implied GHz instead.

## GPU kernels

Every `sgemm/gpu/kernels/*.cu` is built by `python run.py gpu build` (nvcc, `-arch=sm_75`, only changed
files rebuild) and benchmarked automatically. A kernel file exports one function:

```c
extern "C" int launch(int n, const float *A, const float *B, float *C);  // device ptrs, row-major fp32,
                                                                        // overwrite C, return cudaGetLastError()
```

- `cublas_fp32.cu`: cuBLAS `sgemm` through that same interface, the "100%" line for `pct_of_vendor.png`.
- `00_naive.cu`: one thread per element, deliberately uncoalesced (the comment explains why).
- Next stage = a new file, e.g. `01_coalesced.cu`. No Python change needed.

```bash
python run.py gpu build
python run.py gpu run --kernels cublas_fp32,00_naive --sizes 512,1024,2048   # just these two
python run.py gpu all                                                         # build + everything + plots
```
