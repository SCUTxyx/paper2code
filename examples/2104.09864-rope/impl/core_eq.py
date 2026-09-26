"""RoPE — block-wise rotation per paper Eq.(13)-(15) (formula version).

Source: Su et al., RoFormer, arXiv:2104.09864, §3.2.2 Eq.(13)(14)(15).
numpy-only, float64. Index convention: the paper's θ_i uses 1-based i = 1..d/2
(Eq.15's i−1); this implementation's 0-based code index i corresponds to the
paper's i−1 — identical semantics (the classic off-by-one risk is discussed in
REPORT finding 1).
"""

import numpy as np


def theta_seq(d, base=10000.0):
    """Eq.(15): θ_i = base^{-2(i-1)/d}, i = 1..d/2. Returns (d/2,)."""
    i = np.arange(d // 2)                      # code 0-based ↔ paper i-1
    return base ** (-2.0 * i / d)              # Eq.(15)


def rope_rotate(x, m, base=10000.0):
    """Eq.(13)(14): rotate x by m steps in 2-dimensional blocks. x shape (..., d),
    d even."""
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    ang = m * theta_seq(d, base)               # per-block rotation angle m·θ_i
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]        # each block's (a_i, b_i)
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin       # Eq.(14), first row of each block
    out[..., 1::2] = x1 * sin + x2 * cos       # Eq.(14), second row of each block
    return out


def rope_score(q, k, m, n, base=10000.0):
    """Attention score ⟨f(q,m), f(k,n)⟩."""
    return float(rope_rotate(q, m, base) @ rope_rotate(k, n, base))
