"""AdamW property tests (W1 decoupling — the paper's core claim)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def test_moments_independent_of_decay():
    """W1: the moment trajectories (m̂, v̂) are IDENTICAL for every λ — the decay
    never enters the adaptive path. Also, the decay contribution is exactly
    lr·λ·θ_{t-1} regardless of the gradient history.

    A coupled implementation (Adam with λ·g folded into the gradient) fails this:
    its moments would shrink with λ.
    """
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((30, 4))
    theta0 = rng.standard_normal(4)
    ref = core_eq.adamw_run(seq, theta0, weight_decay=0.0)
    for wd in (1e-4, 0.01, 0.5):
        out = core_eq.adamw_run(seq, theta0, weight_decay=wd)
        assert np.allclose(out["m_hats"], ref["m_hats"], rtol=0, atol=0), f"λ={wd}"
        assert np.allclose(out["v_hats"], ref["v_hats"], rtol=0, atol=0), f"λ={wd}"


def test_decay_term_exact():
    """W1, second half: with the same gradient history, the parameter gap between
    two runs that differ ONLY in λ follows a closed-form linear recursion. Since
    the adaptive step A_t is gradient-only (independent of θ — itself a statement
    of decoupling), the gap d_t = θ_t^{wd} − θ_t^{base} satisfies
    d_t = (1−lr·λ)·d_{t-1} − lr·λ·θ_{t-1}^{base}."""
    rng = np.random.default_rng(1)
    seq = rng.standard_normal((25, 3))
    theta0 = rng.standard_normal(3)
    lr, wd = 0.05, 0.1
    out = core_eq.adamw_run(seq, theta0, lr=lr, weight_decay=wd)
    base = core_eq.adamw_run(seq, theta0, lr=lr, weight_decay=0.0)
    d = np.zeros_like(theta0)
    for t in range(1, 26):
        d = (1 - lr * wd) * d - lr * wd * base["thetas"][t - 1]
        expect = base["thetas"][t] + d
        assert np.allclose(out["thetas"][t], expect, rtol=0, atol=1e-12), f"t={t}"
