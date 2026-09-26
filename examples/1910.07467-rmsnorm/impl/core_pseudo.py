"""RMSNorm — independent formulation: per-token loop with an explicit sum.

Difference vs the formula version: no vectorized mean over the last axis; each
token's RMS is accumulated explicitly. For the cross-check in test_crosscheck.py.
"""

import numpy as np


def rmsnorm_loop(x, g, eps=1e-8):
    """Same semantics as core_eq.rmsnorm, per token. x: (..., d)."""
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    d = x.shape[-1]
    flat = x.reshape(-1, d)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        acc = 0.0
        for v in flat[i]:                   # explicit Σ x_j²
            acc += v * v
        r = np.sqrt(acc / d + eps)
        out[i] = flat[i] / r * g
    return out.reshape(x.shape)
