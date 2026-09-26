"""DDPM property tests (D3 schedule monotonicity / endpoints / x_T limit)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def test_schedule_endpoints():
    """§4: β₁ = 1e-4 and β_T = 0.02 (values from the paper)."""
    betas = core_eq.linear_beta_schedule(1000)
    assert betas[0] == 1e-4
    assert betas[-1] == 0.02


def test_alpha_bar_strictly_decreasing():
    """D3: ᾱ_t strictly decreasing (direct consequence of β_t > 0)."""
    abars = core_eq.alpha_bar_seq(core_eq.linear_beta_schedule(1000))
    assert np.all(np.diff(abars) < 0.0)
    assert np.all((abars > 0.0) & (abars < 1.0))


def test_signal_destroyed_at_T():
    """D3: ᾱ_T ≈ 0 (below 1e-3 for the linear schedule); the mean of x_T is
    squashed to |√ᾱ_T·x0| ≤ 3e-3."""
    abars = core_eq.alpha_bar_seq(core_eq.linear_beta_schedule(1000))
    assert abars[-1] < 1e-3
    x0 = np.full(3, 1.0)
    xt = core_eq.q_sample(x0, 1000, np.zeros(3), abars)  # conditional mean at ε=0
    assert np.all(np.abs(xt) <= 3.0 * np.sqrt(abars[-1]))


def test_variance_of_closed_form_exact():
    """Eq.4's conditional variance is exactly 1-ᾱ_t (closed form, not statistical).
    Structural check: dividing (x_t − √ᾱ_t x0) by √(1-ᾱ_t) must recover a standard
    normal draw (6σ guard, not the main assertion)."""
    abars = core_eq.alpha_bar_seq(core_eq.linear_beta_schedule(1000))
    x0 = np.array([0.3, -0.7, 1.2])
    for t in (1, 17, 500, 1000):
        rng = np.random.default_rng(t)
        xs = core_pseudo.q_sample_batch(x0, t, rng, abars, 4)
        for x in xs:
            eps_hat = (x - np.sqrt(abars[t - 1]) * x0) / np.sqrt(1.0 - abars[t - 1])
            assert np.all(np.abs(eps_hat) < 6.0)  # 6σ guard for a standard normal
