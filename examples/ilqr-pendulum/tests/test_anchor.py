"""iLQR anchor tests: swing-up task + the 1-step LQR closed-form oracle."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def test_swing_up():
    """I-6: swing up from hanging (π, 0) to upright within 100 steps at
    |u| ≤ 3 N·m, with KKT violation < 1e-6 (a genuine constrained stationary
    point, not just a stall). Deterministic solver + fixed config."""
    out = core_eq.swing_up()
    xf = out["trajectory"][-1]
    assert abs(xf[0]) < 0.15, f"final angle {xf[0]:.4f} rad from upright"
    assert abs(xf[1]) < 0.3, f"final velocity {xf[1]:.4f} rad/s"
    assert out["grad_inf"] < 1e-6
    assert out["costs"][-1] < 200.0


def test_one_step_lqr_closed_form():
    """I-5: on a 1-step linear-quadratic problem, iLQR must reproduce the closed
    form exactly.

    Hand derivation: min ½q x₁² + ½r u² s.t. x₁ = a x₀ + b u
      dJ/du = r u + q b x₁ = 0, x₁ = a x₀ + b u
      ⇒ u* = −(a b q x₀)/(r + b² q)
    a=0.9, b=0.5, q=2, r=1, x₀=1.3 ⇒ u* = −1.17/1.5 = −0.78 exactly."""
    a, b, q, r, x0 = 0.9, 0.5, 2.0, 1.0, 1.3

    def step(x, u):
        return np.array([a * x[0] + b * u])

    def jac(x, u):
        return np.array([[a]]), np.array([[b]])

    u_star = -(a * b * q * x0) / (r + b ** 2 * q)
    assert abs(u_star - (-0.78)) < 1e-12          # the hand constant itself

    out = core_eq.ilqr(x0=np.array([x0]), T=1, step_fn=step, jac_fn=jac,
                       x_goal=np.zeros(1), w_x=(0.0,), w_u=r, w_f=(q,),
                       u_max=10.0)
    # the solver stops at KKT violation ≤ 1e-6 with a decaying regularizer,
    # so the distance to the exact closed form is ~1e-12 in practice; assert 1e-9
    assert abs(out["controls"][0] - u_star) < 1e-9, out["controls"][0]
    # and the reached terminal state matches the dynamics exactly
    assert np.allclose(out["trajectory"][1], a * x0 + b * out["controls"][0],
                       rtol=0, atol=1e-12)
