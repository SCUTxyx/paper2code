"""DDPM anchor test (D1): closed-form marginal vs step-by-step noising Monte Carlo.

Tolerances derived analytically (spec §2.4, no eyeballed numbers):
- mean: per-dimension std = √((1-ᾱ_t)/N) → tolerance 5σ/√N;
- variance: relative standard error of a Gaussian sample variance ≈ √(2/N)
  → relative tolerance 5·√(2/N).
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo

T, D, T_T = 1000, 3, 400   # schedule length / data dim / calibration time step
N = 8000                   # Monte Carlo trajectory count


def _setup():
    betas = core_eq.linear_beta_schedule(T)
    abars = core_eq.alpha_bar_seq(betas)
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-1.0, 1.0, size=D)   # paper's setting x0 ∈ [-1,1] (§4)
    return betas, abars, x0


def test_closed_form_vs_iterative_monte_carlo():
    """D1: at t=400, both the closed-form and iterative samplers must land within
    the analytic tolerances of the same mean/variance."""
    betas, abars, x0 = _setup()
    t = T_T
    ab = abars[t - 1]

    rng = np.random.default_rng(42)
    xs_closed = core_pseudo.q_sample_batch(x0, t, rng, abars, N)
    rng = np.random.default_rng(43)
    xs_iter = core_pseudo.q_sample_iterative_batch(x0, t, rng, betas, N)

    tol_mean = 5.0 * np.sqrt((1.0 - ab) / N)          # per-dim mean tolerance
    tol_var = 5.0 * np.sqrt(2.0 / N)                  # variance relative tolerance
    for name, xs in (("closed", xs_closed), ("iterative", xs_iter)):
        mean_err = np.abs(xs.mean(axis=0) - np.sqrt(ab) * x0)
        var = xs.var(axis=0)
        assert np.all(mean_err < tol_mean), f"{name} mean err {mean_err} ≥ {tol_mean}"
        assert np.all(np.abs(var - (1.0 - ab)) / (1.0 - ab) < tol_var), (
            f"{name} var rel err {np.abs(var - (1-ab))/(1-ab)} ≥ {tol_var}")


def test_two_paths_agree_in_distribution():
    """D1, strengthened: the two paths' means must also agree with each other,
    far inside the sum of their tolerances (a cross-check of same-distribution)."""
    betas, abars, x0 = _setup()
    t = T_T
    rng = np.random.default_rng(7)
    xs_closed = core_pseudo.q_sample_batch(x0, t, rng, abars, N)
    rng = np.random.default_rng(8)
    xs_iter = core_pseudo.q_sample_iterative_batch(x0, t, rng, betas, N)
    tol = 5.0 * np.sqrt((1.0 - abars[t - 1]) / N)
    diff = np.abs(xs_closed.mean(axis=0) - xs_iter.mean(axis=0))
    assert np.all(diff < np.sqrt(2) * tol), f"mean gap between paths {diff} too large"
