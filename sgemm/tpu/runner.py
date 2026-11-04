"""TPU unit (Colab TPU runtime). Currently: XLA (jnp.dot) baselines so the loop works end to end.

Add Pallas kernels by appending Kernel(...) entries to kernels(). Sources live in
sgemm/tpu/kernels/.  jax is imported lazily so other units don't need it.

Precision matters on TPU, so each precision is its own kernel row:
  xla_bf16          bf16 in/out, the MXU's native mode (the speed ceiling)
  xla_fp32_default  fp32 arrays, default precision: XLA may run ONE bf16 pass on the MXU, so it is
                    fast but only roughly fp32-accurate. Included to make that trade-off visible.
  xla_fp32_highest  fp32 arrays, Precision.HIGHEST (multi-pass bf16): the fair fp32 "SGEMM" reference.

Contract shared with the other units:  build(cfg)  and  run_benchmark(cfg) -> dict
"""
from __future__ import annotations

import numpy as np

from ..common.bench import Kernel, run_python_backend
from ..common.env import check_expected_device, collect


def build(cfg: dict) -> None:
    print("[tpu] nothing to build (JAX jit-compiles at first call).")


def kernels() -> list[Kernel]:
    import jax
    import jax.numpy as jnp

    def fetch(out):
        return np.asarray(out.astype(jnp.float32))

    def xla_dot(dtype, precision):
        def setup(A, B):
            a = jax.device_put(jnp.asarray(A, dtype))
            b = jax.device_put(jnp.asarray(B, dtype))
            f = jax.jit(lambda x, y: jnp.dot(x, y, precision=precision))
            return lambda: f(a, b)
        return setup

    bf16_round = lambda x: np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))
    P = jax.lax.Precision
    return [
        Kernel("xla_fp32_highest", "fp32", xla_dot(jnp.float32, P.HIGHEST), fetch, tol=1e-3),
        Kernel("xla_fp32_default", "fp32-default", xla_dot(jnp.float32, None), fetch, tol=5e-2),
        Kernel("xla_bf16", "bf16", xla_dot(jnp.bfloat16, None), fetch, tol=2e-2,
               ref_key="bf16", ref_round=bf16_round),
    ]


def run_benchmark(cfg: dict) -> dict:
    info = collect("tpu")
    if info["device"] == "no-tpu" and not cfg.get("any_device"):
        raise SystemExit(f"JAX does not see a TPU ({info['tpu']}). In Colab: Runtime > Change runtime "
                         "type > TPU, reconnect, and check `import jax; jax.devices()`. "
                         "Pass --any-device to run the same code on CPU for a dry run.")
    check_expected_device({**info, "device": info["tpu"].get("platform", "none")}, cfg)
    return run_python_backend("tpu", info, kernels(), lambda out: out.block_until_ready(), cfg)
