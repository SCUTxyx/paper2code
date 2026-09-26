"""Diffusion Policy gradient checks (P-4)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq

T, D_A, T_P, T_T = 60, 3, 16, 41


def test_grad_wrt_x0():
    """P-4: ∂x_t/∂x0 = √ᾱ_t·I (the closed-form marginal is linear in x_0)."""
    betas = core_eq.linear_beta_schedule(T)
    abars = core_eq.alpha_bar_seq(betas)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((T_P, D_A))
    eps = rng.standard_normal((T_P, D_A))
    w = rng.standard_normal((T_P, D_A))
    a = np.sqrt(abars[T_T])

    def f(x_flat):
        x0_v = x_flat.reshape(T_P, D_A)
        return float(np.sum(w * core_eq.q_sample(x0_v, T_T, eps, abars)))

    err = assert_grad_close(f, x0.ravel(), (a * w), rtol=1e-6, name="d x_t/d x0")
    print(f"\n[diffusion-policy] d x_t/d x0 max rel err = {err:.2e}")


def test_loss_grad_wrt_eps_hat():
    """P-4: ∂L/∂ε̂ = 2(ε̂−ε)/N for the training objective (Eq.5) — the analytic
    gradient that ε̂_θ's backbone receives in DP training."""
    rng = np.random.default_rng(1)
    eps = rng.standard_normal((T_P, D_A))
    eps_hat = rng.standard_normal((T_P, D_A))
    L, g = core_eq.training_loss(eps, eps_hat)
    err = assert_grad_close(
        lambda e: core_eq.training_loss(eps, e.reshape(T_P, D_A))[0],
        eps_hat.ravel(), g, rtol=1e-6, name="dL/d ε̂")
    print(f"\n[diffusion-policy] dL/d ε̂ max rel err = {err:.2e}")
