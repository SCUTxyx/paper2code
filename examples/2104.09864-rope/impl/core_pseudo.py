"""RoPE — independent formulations: the explicit block-diagonal matrix (the literal
Eq.14 shape) and the complex-number form (§3.2.1).

The formula version (core_eq) rotates via slicing; this module provides two other
paths: (1) build the full R^m_Θ matrix explicitly and multiply; (2) for d=2,
multiply by e^{imθ} per the §3.2.1 complex formulation. For the cross-check in
test_crosscheck.py (R3).
"""

import numpy as np


def rotation_matrix(m, d, base=10000.0):
    """Eq.(14) literal shape: explicit block-diagonal R^m_Θ, (d, d)."""
    half = d // 2
    i = np.arange(half)
    thetas = base ** (-2.0 * i / d)            # Eq.(15)
    ang = m * thetas
    cos, sin = np.cos(ang), np.sin(ang)
    R = np.zeros((d, d))
    R[0::2, 0::2] = np.diag(cos)               # block (1,1)
    R[0::2, 1::2] = np.diag(-sin)              # block (1,2)
    R[1::2, 0::2] = np.diag(sin)               # block (2,1)
    R[1::2, 1::2] = np.diag(cos)               # block (2,2)
    return R


def rope_rotate_matrix(x, m, base=10000.0):
    """f(x, m) = R^m_Θ x (explicit-matrix path)."""
    return rotation_matrix(m, np.asarray(x).shape[-1], base) @ np.asarray(x, dtype=np.float64)


def rope_rotate_complex(x, m, base=10000.0):
    """§3.2.1 complex form: each block (a, b) as a+bi, multiplied by e^{imθ_i}."""
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    half = d // 2
    thetas = base ** (-2.0 * np.arange(half) / d)   # Eq.(15)
    ang = m * thetas
    out = np.empty_like(x)
    for i in range(half):
        a, b = x[..., 2 * i], x[..., 2 * i + 1]
        z = (a + 1j * b) * np.exp(1j * ang[i])      # q·e^{imθ} (§3.2.1)
        out[..., 2 * i] = z.real
        out[..., 2 * i + 1] = z.imag
    return out
