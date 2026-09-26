"""RMSNorm gradient check (C4): closed-form backward vs central differences."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq

D = 8


def test_grad_wrt_x():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(D)
    g = rng.standard_normal(D)
    w = rng.standard_normal(D)
    analytic = core_eq.rmsnorm_grad(x, g, w, eps=1e-8)
    err = assert_grad_close(
        lambda v: float(w @ core_eq.rmsnorm(v, g, eps=1e-8)),
        x, analytic, rtol=1e-6, name="dL/dx (rmsnorm)")
    print(f"\n[rmsnorm] dL/dx max rel err = {err:.2e}")


def test_grad_with_gain_near_zero():
    """Edge: g with zero components and a small-magnitude x (scale 0.05) — the
    backward must stay exact; this is what ε stabilizes.

    Scale note (dry-run lesson): shrinking x to 1e-3 makes r ~ 1e-3, where the
    loss's third derivative ~1/r³ explodes and the central-difference truncation
    term h²·f'''/6 breaches rtol — an artifact of the CHECKER, not the code.
    x ~ 0.05 keeps truncation ~1e-7 relative; documented per
    references/verification.md §2.6."""
    rng = np.random.default_rng(1)
    x = 0.05 * rng.standard_normal(D)
    g = rng.standard_normal(D)
    g[3] = 0.0
    w = rng.standard_normal(D)
    analytic = core_eq.rmsnorm_grad(x, g, w, eps=1e-8)
    err = assert_grad_close(
        lambda v: float(w @ core_eq.rmsnorm(v, g, eps=1e-8)),
        x, analytic, rtol=1e-6, name="dL/dx (small x, g has zeros)")
    print(f"\n[rmsnorm] small-x grad err = {err:.2e}")
