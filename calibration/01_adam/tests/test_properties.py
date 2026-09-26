"""Adam property tests (claims A1 / A2 / A4)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]  # guard against cross-repro module-name collisions
import numpy as np

import core_eq


def test_first_step_size_independent_of_scale():
    """A1: first step has m̂_1=g, v̂_1=g², so the update is α·g/(|g|+ε) ≈ α,
    independent of the gradient scale."""
    g = np.array([3.7, -2.1, 0.9])
    out = core_eq.adam_run(g[None, :], theta0=np.zeros(3), lr=0.1)
    step = out["thetas"][1] - out["thetas"][0]
    assert np.allclose(np.abs(step), 0.1, rtol=1e-6), f"|step|={np.abs(step)}"
    assert np.all(np.sign(step) == np.sign(-g))


def test_beta_to_zero_degenerates():
    """A2: β→0 degenerates the moments to the current gradient; the update must be
    α·g/(|g|+ε) (hand-written reference, not a restatement of the implementation)."""
    g = np.array([0.5, -1.3, 2.0])
    eps = 1e-8
    out = core_eq.adam_run(g[None, :], theta0=np.zeros(3), lr=0.1,
                           beta1=1e-12, beta2=1e-12, eps=eps)
    step = out["thetas"][1] - out["thetas"][0]
    expected = -0.1 * g / (np.abs(g) + eps)  # hand-written reference
    assert np.allclose(step, expected, rtol=1e-12, atol=1e-15)
    # Deviation from the sign update has the analytic bound α·ε/(|g|+ε) ≤ α·ε/0.5
    assert np.max(np.abs(step - (-0.1 * np.sign(g)))) < 0.1 * eps / 0.5


def test_constant_gradient_exact():
    """A4: under a constant gradient, m̂_t=g and v̂_t=g² hold exactly for every t
    (the geometric sum and the bias correction cancel)."""
    c = np.array([0.7, -0.4])
    T = 200
    seq = np.tile(c, (T, 1))
    out = core_eq.adam_run(seq, theta0=np.zeros(2), lr=0.1)
    assert np.allclose(out["m_hats"], c[None, :], rtol=0, atol=1e-12)
    assert np.allclose(out["v_hats"], (c * c)[None, :], rtol=0, atol=1e-12)
    # Every step moves by exactly α·c/(|c|+ε) → θ_t is linear in t
    per_step = -0.1 * c / (np.abs(c) + 1e-8)
    expect = np.array([per_step * t for t in range(T + 1)])
    assert np.allclose(out["thetas"], expect, rtol=0, atol=1e-12)


def test_zero_gradient_stability():
    """Zero gradient gives exactly zero update (ε floors the denominator; no NaN)."""
    out = core_eq.adam_run(np.zeros((5, 3)), theta0=np.array([1.0, 2.0, 3.0]))
    assert np.allclose(out["thetas"], np.array([[1.0, 2.0, 3.0]] * 6), rtol=0, atol=0)
