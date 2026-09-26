"""DDPM dual-implementation cross-check (D2): under the SAME noise sequence,
recursion = unrolled weighted sum = closed form (with the composed noise).

D2 is an exact identity (derivation in METHOD_CARD), tolerance 1e-10; the
closed-form path must first compose the step noises into ε_comb by their weights,
then go through Eq.(4) — also exactly equal.
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

T, D = 1000, 4


def _combined_eps(eps_seq, t, betas):
    """ε_comb = Σ_s √(ᾱ_t/ᾱ_s)·√(1-α_s)·ε_s / √(1-ᾱ_t)."""
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    acc = np.zeros(D)
    for s in range(1, t + 1):
        coef = np.sqrt(abar[t - 1] / abar[s - 1]) * np.sqrt(1.0 - alphas[s - 1])
        acc += coef * eps_seq[s - 1]
    return acc / np.sqrt(1.0 - abar[t - 1])


def test_iterative_equals_unrolled():
    """D2: the two paths are exactly equal under the same ε_s prefix."""
    betas = core_eq.linear_beta_schedule(T)
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-1, 1, D)
    eps_seq = rng.standard_normal((T, D))
    for t in (1, 17, 500, 1000):
        a = core_pseudo.q_sample_iterative(x0, t, eps_seq, betas)
        b = core_pseudo.q_sample_unrolled(x0, t, eps_seq, betas)
        assert np.allclose(a, b, rtol=0, atol=1e-10), f"t={t}: {a} vs {b}"


def test_closed_form_with_combined_eps():
    """The closed form Eq.(4) with the composed noise ε_comb must exactly reproduce
    the step-by-step recursion."""
    betas = core_eq.linear_beta_schedule(T)
    abars = core_eq.alpha_bar_seq(betas)
    rng = np.random.default_rng(1)
    x0 = rng.uniform(-1, 1, D)
    eps_seq = rng.standard_normal((T, D))
    for t in (1, 17, 500, 1000):
        a = core_pseudo.q_sample_iterative(x0, t, eps_seq, betas)
        b = core_eq.q_sample(x0, t, _combined_eps(eps_seq, t, betas), abars)
        assert np.allclose(a, b, rtol=0, atol=1e-10), f"t={t}: {a} vs {b}"
