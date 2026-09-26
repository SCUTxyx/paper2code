"""DDPM gradient checks (D4): analytic gradients of the closed-form / unrolled
forms w.r.t. their inputs vs central differences."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq
import core_pseudo

T, D, T_T = 1000, 4, 233


def test_grad_wrt_x0_and_eps():
    """D4: ∂x_t/∂x0 = √ᾱ_t·I and ∂x_t/∂ε = √(1-ᾱ_t)·I (Eq.4's linear
    reparameterization)."""
    betas = core_eq.linear_beta_schedule(T)
    abars = core_eq.alpha_bar_seq(betas)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(D)
    eps = rng.standard_normal(D)
    w = rng.standard_normal(D)
    a = np.sqrt(abars[T_T - 1])

    err1 = assert_grad_close(
        lambda x: float(w @ core_eq.q_sample(x, T_T, eps, abars)),
        x0, (a * w), rtol=1e-6, name="d x_t/d x0")
    err2 = assert_grad_close(
        lambda e: float(w @ core_eq.q_sample(x0, T_T, e, abars)),
        eps, (np.sqrt(1 - a**2) * w), rtol=1e-6, name="d x_t/d ε")
    print(f"\n[ddpm] d/dx0 err={err1:.2e}, d/dε err={err2:.2e}")


def test_grad_unrolled_wrt_eps_s():
    """The unrolled identity's gradient w.r.t. ε_s = √(ᾱ_t/ᾱ_s)·√(1-α_s)·I,
    checked for every s."""
    betas = core_eq.linear_beta_schedule(T)
    alphas = 1.0 - betas
    abars = core_eq.alpha_bar_seq(betas)
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal(D)
    eps_seq = rng.standard_normal((T_T, D))
    w = rng.standard_normal(D)

    analytic = np.zeros(T_T * D)
    for s in range(1, T_T + 1):
        coef = np.sqrt(abars[T_T - 1] / abars[s - 1]) * np.sqrt(1.0 - alphas[s - 1])
        analytic[(s - 1) * D:(s - 1) * D + D] = coef * w

    def f(e_flat):
        e = e_flat.reshape(T_T, D)
        return float(w @ core_pseudo.q_sample_unrolled(x0, T_T, e, betas))

    err = assert_grad_close(f, eps_seq.ravel(), analytic, rtol=1e-6,
                            name="d x_t/d ε_s (unrolled)")
    print(f"\n[ddpm] unrolled grad err = {err:.2e}")
