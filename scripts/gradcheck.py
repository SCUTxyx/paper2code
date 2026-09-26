"""Shared finite-difference gradient checker — reused by every repro and exam
(the core of verification-ladder level L2).

Numeric specification (references/verification.md):
- float64, central differences;
- analytic gradient of a scalar loss L: R^n -> R compared component-wise;
- relative-error threshold 1e-6 by default;
- step h_i = cbrt(float64 eps) * max(1, |x_i|) ≈ 6e-6·max(1,|x_i|)
  — the optimal central-difference step sits at the balance of the truncation
  error O(h²) and the rounding error O(ε/h), i.e. at cbrt(ε) — NOT the
  forward-difference optimum sqrt(ε).
"""

from __future__ import annotations

import numpy as np

__all__ = ["central_diff_grad", "max_rel_err", "assert_grad_close"]


def central_diff_grad(f, x, h=None):
    """f: accepts an array of x's shape, returns a scalar; returns the central-difference
    gradient (same shape as x)."""
    x = np.asarray(x, dtype=np.float64)
    if h is None:
        h = np.cbrt(np.finfo(np.float64).eps) * np.maximum(1.0, np.abs(x))
    flat = x.ravel()
    g = np.zeros_like(flat)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + h.ravel()[i]
        fp = f(flat.reshape(x.shape))
        flat[i] = orig - h.ravel()[i]
        fm = f(flat.reshape(x.shape))
        flat[i] = orig
        g[i] = (fp - fm) / (2.0 * h.ravel()[i])
    return g.reshape(x.shape)


def max_rel_err(numeric, analytic, atol=1e-12):
    """Component-wise relative error with a floored denominator (avoids blowup on
    all-zero comparisons)."""
    numeric = np.asarray(numeric, dtype=np.float64)
    analytic = np.asarray(analytic, dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(numeric), np.abs(analytic)), atol)
    return float(np.max(np.abs(numeric - analytic) / denom))


def assert_grad_close(f, x, analytic, rtol=1e-6, name="grad"):
    """Compare the analytic gradient against central differences; raises AssertionError
    beyond rtol. Returns the max relative error (for REPORT's results matrix)."""
    numeric = central_diff_grad(f, x)
    analytic = np.asarray(analytic, dtype=np.float64)
    if numeric.shape != analytic.shape and numeric.size == analytic.size:
        numeric = numeric.reshape(analytic.shape)  # allow matrix-shaped analytic grads
    err = max_rel_err(numeric, analytic)
    if err > rtol:
        idx = np.unravel_index(
            np.argmax(np.abs(numeric - analytic)
                      / np.maximum(np.abs(numeric) + np.abs(analytic), 1e-12)),
            np.asarray(analytic).shape)
        raise AssertionError(
            f"{name}: max rel err {err:.3e} > rtol {rtol:g} at {idx} "
            f"(analytic={analytic[idx]:.6e}, numeric={numeric[idx]:.6e})")
    return err
