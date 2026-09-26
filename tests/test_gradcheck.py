"""Regression tests for gradcheck.py itself — verify the verifier.

Includes negative tests: feeding a wrong gradient MUST raise AssertionError
(a gradient checker that never fails is a dud).
"""

import numpy as np
import pytest

from gradcheck import assert_grad_close, central_diff_grad, max_rel_err


def test_linear_function_exact():
    """For f(x) = a·x the gradient is exactly a. Note: central differences are NOT
    machine-exact — rounding inside f (|f|·eps/(2h)) puts an honest floor at ~1e-10
    relative error for O(1) values."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(6)
    err = assert_grad_close(lambda x: float(a @ x), rng.standard_normal(6),
                            a, rtol=1e-8, name="linear")
    assert err < 1e-9


def test_quadratic_function():
    """f(x) = xᵀAx has gradient (A + Aᵀ)x."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4))
    x = rng.standard_normal(4)
    analytic = (A + A.T) @ x
    err = assert_grad_close(lambda v: float(v @ A @ v), x, analytic, rtol=1e-6)
    assert err < 1e-9


def test_catches_wrong_gradient():
    """Negative test: a 1% perturbation of one gradient component must be caught
    (well above the 1e-6 threshold)."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal(5)
    x = rng.standard_normal(5)
    wrong = a.copy()
    wrong[2] *= 1.01
    with pytest.raises(AssertionError, match="max rel err"):
        assert_grad_close(lambda v: float(a @ v), x, wrong, rtol=1e-6)


def test_catches_missing_component():
    """Negative test: a gradient missing a nonzero component (a common
    implementation bug) must be caught."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal(4)
    wrong = a.copy()
    wrong[1] = 0.0
    with pytest.raises(AssertionError):
        assert_grad_close(lambda v: float(a @ v), rng.standard_normal(4),
                          wrong, rtol=1e-6)


def test_step_size_is_cbrt_eps():
    """The central-difference optimum is cbrt(eps)·max(1,|x|), not the
    forward-difference sqrt(eps). Same-scale x used: with mixed magnitudes
    (e.g. [1, 100]) the small component's difference is swamped by the large
    component's ulp, pushing the error floor to ~1e-8 — a lesson in itself."""
    x = np.array([1.0, 2.0])
    g = central_diff_grad(lambda v: float(np.sum(v * v)), x)
    expected = 2.0 * x
    assert max_rel_err(g, expected) < 1e-9
    # Verify the h formula explicitly: cbrt(eps)/sqrt(eps) ≈ 400×
    h = np.cbrt(np.finfo(np.float64).eps) * np.maximum(1.0, np.abs(x))
    assert h[0] / (np.sqrt(np.finfo(np.float64).eps)) > 100.0
