"""Diffusion Policy cross-checks: closed-form vs iterative forward (same-ε exact
identity); vectorized vs loop DDIM chain; two receding-horizon executors."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo

T, D_A, T_P, T_A = 60, 4, 16, 8


def test_closed_form_vs_iterative():
    """P-1: under the same ε sequence, the closed-form marginal equals the
    step-by-step recursion exactly (unrolled identity, re-derived in METHOD_CARD)."""
    betas = core_eq.linear_beta_schedule(T)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((T_P, D_A))
    eps_seq = rng.standard_normal((T, T_P, D_A))
    abars = core_eq.alpha_bar_seq(betas)
    for t in (1, 17, T):
        a = core_eq.q_sample(x0, t, _combined(eps_seq, t, betas), abars)
        b = core_pseudo.q_sample_iterative(x0, t, eps_seq, betas)
        assert np.allclose(a, b, rtol=0, atol=1e-10), f"t={t}"


def _combined(eps_seq, t, betas):
    """ε_comb = Σ_s √(ᾱ_t/ᾱ_s)·√(1-α_s)·ε_s / √(1-ᾱ_t) (unrolled-identity weights)."""
    alphas = 1.0 - betas
    abar = np.concatenate([[1.0], np.cumprod(alphas)])
    acc = np.zeros_like(eps_seq[0])
    for s in range(1, t + 1):
        coef = np.sqrt(abar[t] / abar[s]) * np.sqrt(1.0 - alphas[s - 1])
        acc = acc + coef * eps_seq[s - 1]
    return acc / np.sqrt(1.0 - abar[t])


def test_ddim_two_forms():
    """Direct Eq.12 form vs predicted-x_0 form: same chain, 1e-12 equal — and the
    perfect-ε invariance holds through both."""
    betas = core_eq.linear_beta_schedule(T)
    abars = core_eq.alpha_bar_seq(betas)
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal((T_P, D_A))
    eps = rng.standard_normal((T_P, D_A))
    x_T = core_eq.q_sample(x0, T, eps, abars)
    a = core_eq.run_ddim(x_T, [eps] * T, abars)
    b = core_pseudo.run_ddim_loop(x_T, [eps] * T, abars)
    assert np.allclose(a, b, rtol=0, atol=1e-12)
    assert np.allclose(a, x0, rtol=0, atol=1e-10)


def test_executors_match():
    """Two independent receding-horizon executors agree on random chunks."""
    rng = np.random.default_rng(2)
    chunks = [rng.standard_normal((T_P, D_A)) for _ in range(5)]
    for horizon in (13, 40, 80):
        a = core_eq.execute_receding(chunks, T_A, horizon)
        b = core_pseudo.execute_receding_loop(chunks, T_A, horizon)
        assert np.allclose(a, b, rtol=0, atol=1e-12), f"horizon={horizon}"
