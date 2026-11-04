"""Pipeline smoke test with a pure-numpy fake backend: exercises the shared benchmark loop,
CSV writing, resume logic and plotting without needing a GPU, a TPU or a C compiler.

    python -m pytest tests -q        (or: python tests/test_common.py)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sgemm.common import results
from sgemm.common.bench import Kernel, run_sweep
from sgemm.common.results import read_csv


def _kernels():
    def good_setup(A, B):
        return lambda: A @ B

    def broken_setup(A, B):  # forgets to write most of C -> must fail the check
        def run():
            C = np.full_like(A, np.nan)
            C[0, 0] = 1.0
            return C
        return run

    ident = lambda out: out
    return [Kernel("fake_good", "fp32", good_setup, ident, tol=1e-4),
            Kernel("fake_broken", "fp32", broken_setup, ident, tol=1e-4),
            Kernel("fake_capped", "fp32", good_setup, ident, tol=1e-4, max_n=64)]


def test_sweep_resume_and_plots(monkeypatch=None):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        sync = lambda o: None
        rows = run_sweep(_kernels(), [32, 64, 128], sync, out, reps=3, log=lambda s: None)
        assert (out / "fake_good.csv").exists()
        good = read_csv(out / "fake_good.csv")
        assert [r["n"] for r in good] == [32, 64, 128] and all(r["ok"] == 1 for r in good)
        assert all(r["ok"] == 0 for r in read_csv(out / "fake_broken.csv"))
        assert [r["n"] for r in read_csv(out / "fake_capped.csv")] == [32, 64]   # max_n respected
        n_rows = len(rows)

        again = run_sweep(_kernels(), [32, 64, 128], sync, out, reps=3, log=lambda s: None)
        assert again == [] and n_rows == 8                                       # resume: nothing redone


def test_plotting_roundtrip():
    from sgemm.common import plotting
    with tempfile.TemporaryDirectory() as d:
        pkg = Path(d)
        results.PKG_DIR = pkg                                    # redirect unit dirs to a temp tree
        rd = results.results_dir("gpu", "unit-test")
        run_sweep(_kernels()[:1], [32, 64], lambda o: None, rd, reps=3, log=lambda s: None)
        paths = plotting.make_plots("gpu", "unit-test")
        assert paths and all(Path(p).exists() for p in paths)
        assert str(pkg / "gpu" / "plots" / "unit-test") in paths[0]


if __name__ == "__main__":
    test_sweep_resume_and_plots()
    test_plotting_roundtrip()
    print("ok")
