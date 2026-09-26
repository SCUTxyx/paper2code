"""RMSNorm — vectorized implementation of paper Eq.(4) (formula version).

Source: Zhang & Sennrich, arXiv:1910.07467, §3 Eq.(4), with the common
stabilization ε inside the sqrt (implementation convention pinned in EQ_MAP).
numpy-only, float64.
"""

import numpy as np


def rmsnorm(x, g, eps=1e-8):
    """Eq.(4): y = x / sqrt(mean(x²) + ε) ⊙ g, computed over the last axis.

    x: (..., d); g: (d,). Returns y with x's shape.
    """
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    r = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)   # Eq.(3)+ε
    return x / r * g                                            # Eq.(4)


def rmsnorm_grad(x, g, w, eps=1e-8):
    """Analytic gradient of the linear loss L = Σ w·y w.r.t. x (C4):

    ∂L/∂x_k = w_k g_k / r − x_k · Σ_i (w_i g_i x_i) / (d r³),  r = √(m+ε).
    """
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    r = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    wg = w * g
    return wg / r - x * np.sum(wg * x, axis=-1, keepdims=True) / (d * r ** 3)
