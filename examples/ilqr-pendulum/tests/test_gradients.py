"""iLQR gradient checks (I-1 dynamics Jacobians, I-2 exact discrete adjoint)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq

T = 60


def test_dynamics_jacobians():
    """I-1: the analytic A, B equal central differences of the step function
    (component-wise, rtol 1e-6)."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.uniform(-3.0, 3.0, 2)
        u = rng.uniform(-3.0, 3.0)
        A, Bc = core_eq.pendulum_jacobians(x, u)
        for comp in (0, 1):
            err = assert_grad_close(
                lambda v: core_eq.pendulum_step(v, u)[comp], x, A[comp],
                rtol=1e-6, name=f"d f_{comp}/dx")
            err_u = assert_grad_close(
                lambda s: core_eq.pendulum_step(x, s[0])[comp], np.array([u]), Bc[comp],
                rtol=1e-6, name=f"d f_{comp}/du")
            assert err < 1e-8 and err_u < 1e-8
    print("\n[ilqr] dynamics jacobians verified at 5 random points")


def test_rollout_cost_grad():
    """I-2: the backward pass's first-order term is the exact discrete-adjoint
    gradient of the rolled-out cost — checked against central differences of the
    full rollout (rtol 1e-6; the adjoint is exact for interior controls)."""
    x0 = np.array([np.pi, 0.0])
    us = 1.5 * np.sin(np.arange(T) * 0.3)          # non-optimal, interior nominal
    w_x, w_u, w_f = (1.0, 0.1), 1e-2, (20.0, 2.0)
    analytic = core_eq.rollout_cost_grad(x0, us, np.zeros(2), w_x, w_u, w_f)

    g0 = np.zeros(2)

    def f(u_flat):
        xs = [x0.copy()]
        for u in u_flat:
            xs.append(core_eq.pendulum_step(xs[-1], u))
        xs = np.array(xs)
        J = 0.0
        for t in range(T):
            e = xs[t] - g0
            J += 0.5 * np.asarray(w_x) @ (e * e) + 0.5 * w_u * u_flat[t] ** 2
        e = xs[-1] - g0
        return J + 0.5 * np.asarray(w_f) @ (e * e)

    err = assert_grad_close(f, us, analytic, rtol=1e-6, name="dJ/du (adjoint)")
    print(f"\n[ilqr] adjoint grad max rel err = {err:.2e}")
