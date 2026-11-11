"""nvcc build + ctypes loading for the .cu kernels in sgemm/gpu/kernels/.

Every kernels/*.cu becomes bin/lib<stem>.so exporting  int launch(int n, A, B, C)  (device
pointers, row-major fp32, C overwritten). Drop in a new .cu file and it is built and benchmarked
with no other change. No torch import here, so `python run.py gpu build` also works on a
machine that only has the CUDA toolkit (e.g. a Modal image build step).
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

GPU_DIR = Path(__file__).resolve().parent
SRC_DIR = GPU_DIR / "kernels"
BIN_DIR = GPU_DIR / "bin"
CUTLASS_DIR = GPU_DIR / "third_party" / "cutlass"   # fetched on demand, git-ignored
CUTLASS_TAG = "v3.5.1"

DEFAULT_ARCH = "sm_75"  # Tesla T4 (Turing). Pinned on purpose so Colab and Modal builds are identical.


def find_nvcc() -> str:
    for cand in (shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if cand and os.path.exists(cand):
            return cand
    raise SystemExit("nvcc not found. On Colab pick a GPU runtime (the CUDA toolkit is preinstalled); "
                     "elsewhere install the CUDA toolkit and put nvcc on PATH.")

def ensure_cutlass() -> Path:
    """CUTLASS is header-only: shallow-clone a pinned tag once, return its include dir."""
    inc = CUTLASS_DIR / "include"
    if not (inc / "cutlass" / "cutlass.h").exists():
        print(f"[gpu] fetching CUTLASS {CUTLASS_TAG} (one-off)", flush=True)
        CUTLASS_DIR.parent.mkdir(exist_ok=True)
        r = subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", CUTLASS_TAG,
                            "https://github.com/NVIDIA/cutlass.git", str(CUTLASS_DIR)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"could not fetch CUTLASS:\n{r.stderr}")
    return inc


def extra_flags(src: Path) -> list[str]:
    """Files that include CUTLASS headers get the include path and C++17 (CUTLASS requirement)."""
    if "cutlass/" in src.read_text():
        return ["-std=c++17", "--expt-relaxed-constexpr", "-I", str(ensure_cutlass())]
    return []

def sources() -> list[Path]:
    return sorted(SRC_DIR.glob("*.cu"))


def lib_path(src: Path) -> Path:
    return BIN_DIR / f"lib{src.stem}.so"


def build(cfg: dict) -> list[Path]:
    nvcc = find_nvcc()
    arch = cfg.get("arch") or DEFAULT_ARCH
    BIN_DIR.mkdir(exist_ok=True)
    out = []
    for src in sources():
        lib = lib_path(src)
        stamp = lib.with_suffix(".arch")
        stale = (bool(cfg.get("fresh")) or not lib.exists() or not stamp.exists()
                 or stamp.read_text() != arch
                 or lib.stat().st_mtime < src.stat().st_mtime)
        if stale:
            cmd = [nvcc, "-O3", f"-arch={arch}", "-lineinfo", "-shared", "-Xcompiler", "-fPIC",
                   *extra_flags(src), str(src), "-o", str(lib), "-lcublas"]
            print("[gpu] " + " ".join(cmd), flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise SystemExit(f"nvcc failed for {src.name}:\n{r.stderr}")
            if r.stderr.strip():
                print(r.stderr)
            lib.with_suffix(".arch").write_text(arch)
        else:
            print(f"[gpu] {src.name}: up to date ({arch})")
        out.append(lib)
    return out


def load(src: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(lib_path(src)))
    lib.launch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.launch.restype = ctypes.c_int
    return lib
