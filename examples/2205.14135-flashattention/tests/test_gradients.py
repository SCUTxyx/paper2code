"""FlashAttention gradient check (F6): gradients through the TILED forward must
equal the analytic gradients of direct attention (those analytic formulas are
themselves finite-difference-verified in calibration/02_attention).

Hand derivation (standard softmax-attention backward, loss L = Σ W ⊙ out):
  ∂L/∂A = W Vᵀ; softmax backward: A ⊙ (∂L/∂A − rowsum(A ⊙ ∂L/∂A));
  ∂L/∂Q = dS·K/√d; ∂L/∂K = dSᵀ·Q/√d; ∂L/∂V = Aᵀ·W.
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq
import core_pseudo

N_Q, N_K, D, D_V = 6, 5, 4, 3


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((N_Q, D)), rng.standard_normal((N_K, D)),
            rng.standard_normal((N_K, D_V)), rng.standard_normal((N_Q, D_V)))


def _direct_grads(Q, K, V, W, causal):
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    if causal:
        S = np.where(np.triu(np.ones(S.shape, dtype=bool), k=1), -np.inf, S)
    S = S - np.max(S, axis=-1, keepdims=True)
    e = np.exp(S)
    A = e / e.sum(axis=-1, keepdims=True)
    dA = W @ V.T
    dS = A * (dA - (A * dA).sum(axis=-1, keepdims=True))
    return dS @ (K / np.sqrt(d)), dS.T @ (Q / np.sqrt(d)), A.T @ W


def test_grads_through_tiled_forward():
    """∂L/∂(Q,K,V) computed THROUGH the tiled implementations (both forms),
    compared against the direct analytic gradients, rtol 1e-6."""
    Q, K, V, W = _fixture(seed=7)
    dQ, dK, dV = _direct_grads(Q, K, V, W, causal=False)
    targets = [
        ("Q", (N_Q, D), Q, dQ, core_eq.flash_attention_eq),
        ("K", (N_K, D), K, dK, core_eq.flash_attention_eq),
        ("V", (N_K, D_V), V, dV, core_eq.flash_attention_eq),
        ("V-stream", (N_K, D_V), V, dV, core_pseudo.flash_attention_stream),
    ]
    for name, shape, base, analytic, fn in targets:
        def f(x, base=base, shape=shape, name=name[0], fn=fn):
            env = {"Q": Q, "K": K, "V": V}
            env[name] = x.reshape(shape)
            out = fn(env["Q"], env["K"], env["V"], block_q=2, block_k=2)
            return float(np.sum(W * out))

        err = assert_grad_close(f, base.ravel(), analytic, rtol=1e-6,
                                name=f"dL/d{name} (tiled)")
        print(f"\n[flashattention] dL/d{name} max rel err = {err:.2e}")


def test_grads_causal_tiled():
    """Same check under causal masking — mask semantics must survive tiling."""
    Q, K, V, W = _fixture(seed=8)
    dQ, dK, dV = _direct_grads(Q, K, V, W, causal=True)

    def f(x):
        out = core_eq.flash_attention_eq(Q, x.reshape(N_K, D), V,
                                         block_q=2, block_k=2, causal=True)
        return float(np.sum(W * out))

    err = assert_grad_close(f, K.ravel(), dK, rtol=1e-6, name="dL/dK (causal tiled)")
    print(f"\n[flashattention] causal dL/dK max rel err = {err:.2e}")
