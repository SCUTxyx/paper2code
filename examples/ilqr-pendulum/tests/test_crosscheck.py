"""iLQR cross-checks: analytic vs finite-difference Jacobians; the analytic and
FD-based solvers on the full swing-up problem."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def test_analytic_jacobians_vs_fd():
    """I-1 cross-check: hand-derived Jacobians == FD Jacobians at random points."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.uniform(-3.0, 3.0, 2)
        u = rng.uniform(-3.0, 3.0)
        A_a, B_a = core_eq.pendulum_jacobians(x, u)
        A_f, B_f = core_pseudo.pendulum_jacobians_fd(x, u)
        assert np.allclose(A_a, A_f, rtol=1e-6, atol=1e-8)
        assert np.allclose(B_a, B_f, rtol=1e-6, atol=1e-8)


def test_solvers_agree_on_swing_up():
    """The analytic-Jacobian solver and the FD-Jacobian solver converge to the
    same constrained optimum: final cost rtol 1e-5, trajectories atol 1e-3."""
    kwargs = dict(x_goal=np.array([0.0, 0.0]), w_x=(1.0, 0.1),
                  w_u=1e-2, w_f=(20.0, 2.0), u_max=core_eq.U_MAX)
    a = core_eq.ilqr(x0=np.array([np.pi, 0.0]), T=100, step_fn=core_eq.pendulum_step,
                     jac_fn=core_eq.pendulum_jacobians, **kwargs)
    b = core_pseudo.ilqr_fd(x0=np.array([np.pi, 0.0]), T=100, **kwargs)
    Ja, Jb = a["costs"][-1], b["costs"][-1]
    assert abs(Ja - Jb) / max(1.0, abs(Ja)) < 1e-5, (Ja, Jb)
    assert np.allclose(a["trajectory"], b["trajectory"], rtol=0, atol=1e-3)
    assert a["grad_inf"] < 1e-6 and b["grad_inf"] < 1e-6
