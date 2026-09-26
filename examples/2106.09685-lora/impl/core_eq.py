"""LoRA — literal implementation of paper §4.1 (formula version).

Source: Hu et al., arXiv:2106.09685, §4.1 Eq.(4): h = W0·x + (α/r)·B·A·x,
initialization A ~ Gaussian, B = 0. numpy-only, float64.
"""

import numpy as np


def init_lora(d, k, r, rng, scale=0.02):
    """Paper initialization: A ~ N(0, scale²) per entry, B = 0 (§4.1)."""
    A = rng.standard_normal((r, k)) * scale
    B = np.zeros((d, r))
    return B, A


def lora_forward_eq(W0, B, A, x, alpha=1.0):
    """Eq.(4): h = W0 x + (α/r)·(B A) x — ΔW formed explicitly, then applied.

    Returns (h, delta_W) with delta_W = (α/r)·B@A.
    """
    W0 = np.asarray(W0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    r = A.shape[0]
    delta_W = (alpha / r) * (B @ A)                        # Eq.(4), ΔW = (α/r)BA
    return W0 @ x + delta_W @ x, delta_W


def lora_gradients(B, A, x, w, alpha=1.0):
    """Analytic gradients of the linear loss L = wᵀh w.r.t. B and A (L5).

    ∂L/∂B = (α/r)·w (Ax)ᵀ   (outer product, d×k)
    ∂L/∂A = (α/r)·Bᵀw xᵀ    (outer product, r×k)
    """
    r = A.shape[0]
    s = alpha / r
    grad_B = s * np.outer(w, A @ x)
    grad_A = s * (B.T @ w)[:, None] * x[None, :]
    return grad_B, grad_A
