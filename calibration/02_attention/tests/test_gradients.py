"""Attention gradient checks (T5): analytic gradients vs gradcheck central
differences, rtol 1e-6."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq

N, DK, DV = 5, 4, 3


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((N, DK)), rng.standard_normal((N, DK)),
            rng.standard_normal((N, DV)), rng.standard_normal((N, DV)))


def test_grads_q_k_v():
    for name, shape in (("Q", (N, DK)), ("K", (N, DK)), ("V", (N, DV))):
        Q, K, V, W = _fixture(seed=7)
        args = {"Q": Q, "K": K, "V": V}
        analytic = core_eq.attention_grads(Q, K, V, W, causal=True)[
            ["Q", "K", "V"].index(name)]

        def f(x, name=name, args=args, W=W):
            a = dict(args)
            a[name] = x.reshape(shape)
            out, _ = core_eq.attention(a["Q"], a["K"], a["V"], causal=True)
            return float(np.sum(W * out))

        err = assert_grad_close(f, args[name].ravel(), analytic,
                                rtol=1e-6, name=f"dL/d{name}")
        print(f"\n[attention] dL/d{name} max rel err = {err:.2e}")


def test_causal_row_grad_wrt_V():
    """T2 gradient structure: for the single-row loss L = w·out_i0, the analytic
    gradient w.r.t. V is exactly 0 for j>i0 (= A[i0,j]·w for j≤i0), matching
    central differences — including the exact zeros."""
    Q, K, V, W = _fixture(seed=7)
    i0 = 2
    w = W[i0]  # row-loss weight vector (DV,)
    _, A = core_eq.attention(Q, K, V, causal=True)
    analytic = np.zeros((N, DV))
    for j in range(i0 + 1):
        analytic[j] = A[i0, j] * w

    def f(x):
        out, _ = core_eq.attention(Q, K, x.reshape(N, DV), causal=True)
        return float(np.sum(w * out[i0]))

    err = assert_grad_close(f, V.ravel(), analytic, rtol=1e-6, name="d out_i0/dV")
    print(f"\n[attention] row-grad max rel err = {err:.2e}")
