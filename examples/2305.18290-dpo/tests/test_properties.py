"""DPO property tests (P1 degeneracy / P2 monotonicity & swap identity)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def test_reference_policy_gives_log2():
    """P1: π_θ = π_ref → z=0 → L = log 2 (exact)."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(7)
    b = rng.standard_normal(7)
    L = core_eq.dpo_loss(a, b, a, b, beta=0.1)   # log π_θ identical to log π_ref
    assert abs(L - np.log(2.0)) < 1e-15


def test_monotone_decreasing_in_z():
    """P2: the loss is strictly decreasing in z (monotonicity of σ)."""
    beta = 0.1
    base = np.array([1.0])
    losses = []
    for delta in np.linspace(-3.0, 3.0, 13):
        # fix the reference and the rejected item, raise the chosen one →
        # z increases monotonically → L decreases monotonically
        losses.append(core_eq.dpo_loss(base + delta, base, np.zeros(1),
                                       np.zeros(1), beta=beta))
    diffs = np.diff(losses)
    assert np.all(diffs < 0.0), f"not strictly decreasing: {losses}"


def test_swap_identity():
    """P2: L(l,w) − L(w,l) = z(w,l) exactly (derivation: logσ(−z) = −z + logσ(z))."""
    rng = np.random.default_rng(1)
    lw = rng.standard_normal(5)
    ll = rng.standard_normal(5)
    rw = rng.standard_normal(5)
    rl = rng.standard_normal(5)
    beta = 0.3
    z = beta * ((lw - rw) - (ll - rl))
    l_wl = core_eq.dpo_loss(lw, ll, rw, rl, beta=beta)
    l_lw = core_eq.dpo_loss(ll, lw, rl, rw, beta=beta)
    assert np.allclose(l_lw - l_wl, np.mean(z), rtol=0, atol=1e-12)
