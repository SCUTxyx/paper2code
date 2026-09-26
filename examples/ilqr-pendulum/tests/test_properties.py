"""iLQR property tests (I-3 descent/convergence, I-4 KKT stationarity, I-6 box).

Stationarity is tested in two configurations with different honesty levels:
- stabilize (constraints never active): the exact discrete-adjoint gradient must
  vanish at every control — the clean optimality statement;
- swing-up (constraints binding): the forward-clamp variant terminates at points
  whose TRUE rolled-out gradient keeps O(1) residuals near saturated spans (a
  known deficiency of the naive clamp vs Tassa's box-aware QP — REPORT finding 2).
  There we assert the solver's internal criterion, the box, and a loose bound on
  the residual so regressions stay visible.
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq

X0 = np.array([np.pi, 0.0])
GOAL = np.array([0.0, 0.0])
W = dict(w_x=(1.0, 0.1), w_u=1e-2, w_f=(20.0, 2.0))


def _solve(x0, u_max, T=100):
    return core_eq.ilqr(x0=np.asarray(x0, dtype=np.float64), T=T,
                        step_fn=core_eq.pendulum_step,
                        jac_fn=core_eq.pendulum_jacobians,
                        x_goal=GOAL.copy(), u_max=u_max, **W)


def _adjoint(out, x0):
    return core_eq.rollout_cost_grad(np.asarray(x0, dtype=np.float64),
                                     out["controls"], GOAL.copy(),
                                     W["w_x"], W["w_u"], W["w_f"])


def test_cost_monotone():
    """I-3: accepted iterations never increase the cost; converges within cap."""
    out = _solve(X0, 3.0)
    diffs = np.diff(out["costs"])
    assert np.all(diffs <= 1e-12), f"cost increased: {diffs[diffs > 0]}"
    assert len(out["costs"]) < 500, "did not converge within the iteration cap"


def test_box_respected():
    """I-6: every control lies inside the torque box, exactly."""
    out = _solve(X0, 3.0)
    assert np.all(out["controls"] <= 3.0) and np.all(out["controls"] >= -3.0)


def test_stationarity_clean_interior_case():
    """I-4 (clean): a regulation task whose optimal controls never touch the box —
    the exact discrete-adjoint gradient must vanish at every control."""
    out = _solve(np.array([0.3, 0.2]), 10.0)
    us = out["controls"]
    assert np.max(np.abs(us)) < 10.0 - 1e-6, "controls unexpectedly saturated"
    grad = _adjoint(out, np.array([0.3, 0.2]))
    assert np.max(np.abs(grad)) < 1e-3, f"max |grad| = {np.max(np.abs(grad)):.2e}"
    assert out["grad_inf"] < 1e-6


def test_swing_up_clamp_residual_documented():
    """I-4 (binding case, honest limitation): the forward-clamp variant converges
    by its internal criterion, box respected — but the TRUE rolled-out gradient
    keeps an O(1) residual near saturated spans (REPORT finding 2). Asserted as a
    loose bound so regressions are visible; NOT claimed stationary."""
    out = _solve(X0, 3.0)
    assert out["grad_inf"] < 1e-6                     # internal criterion
    assert np.all(np.abs(out["controls"]) <= 3.0)     # box
    grad = _adjoint(out, X0)
    assert np.max(np.abs(grad)) < 5.0, (
        f"clamp residual exploded: {np.max(np.abs(grad)):.2f}")


def test_robustness_across_inits():
    """I-6 robustness: three different hanging-ish inits all reach upright."""
    for theta0 in (np.pi, 2.8, 3.3):
        out = _solve(np.array([theta0, 0.0]), 3.0)
        xf = out["trajectory"][-1]
        assert abs(xf[0]) < 0.2 and abs(xf[1]) < 0.5, f"init {theta0}: {xf}"
