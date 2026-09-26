"""RMSNorm property tests (C1 RMS bound / C2 scale invariance / C3 shift sensitivity)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def _data(seed=0, shape=(6, 8)):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape), rng.standard_normal(shape[-1])


def test_output_rms():
    """C1: with g=1, mean(y²) = m/(m+ε) — deviation from 1 is exactly
    ε/(m+ε) ≤ ε/m (analytic bound, verified to 1e-12)."""
    x, _ = _data()
    eps = 1e-8
    y = core_eq.rmsnorm(x, np.ones(x.shape[-1]), eps=eps)
    m = np.mean(x * x, axis=-1)
    expect_ms = m / (m + eps)
    assert np.allclose(np.mean(y * y, axis=-1), expect_ms, rtol=0, atol=1e-12)
    assert np.all(np.abs(np.mean(y * y, axis=-1) - 1.0) <= eps / m)


def test_scale_invariance():
    """C2: at ε=0 the norm is exactly scale-homogeneous — RMSNorm(cx)=RMSNorm(x)
    for c>0, = −RMSNorm(x) for c<0. With ε>0 the homogeneity is broken by the
    stabilization, bounded analytically (asserted with the derived bound)."""
    x, g = _data(seed=1)
    one = np.ones(x.shape[-1])
    ref = core_eq.rmsnorm(x, one, eps=0.0)
    for c in (0.3, 2.0, 100.0):
        assert np.allclose(core_eq.rmsnorm(c * x, one, eps=0.0), ref,
                           rtol=0, atol=1e-12), f"c={c}"
    assert np.allclose(core_eq.rmsnorm(-2.0 * x, one, eps=0.0), -ref,
                       rtol=0, atol=1e-12)
    # ε>0: deviation from exact homogeneity, bounded by the ε-induced shift of r
    eps = 1e-8
    c = 2.0
    dev = np.max(np.abs(core_eq.rmsnorm(c * x, one, eps=eps)
                        - core_eq.rmsnorm(x, one, eps=eps)))
    m = np.mean(x * x, axis=-1, keepdims=True)
    # |x/√(c²m+ε) − x/√(m+ε)| ≤ |x|·ε·(c²−1)/(c²·(m+ε)^{3/2})·... use the
    # conservative bound |Δ| ≤ |x|·ε·|1/c² − 1|/(2·min(m+ε, c²m+ε)^{3/2}/...) —
    # instead assert against the exact closed form (still oracle-free: it is the
    # definition, not the implementation's code path):
    exact = c * x / np.sqrt(c * c * m + eps)
    assert np.allclose(core_eq.rmsnorm(c * x, one, eps=eps), exact, rtol=0, atol=1e-15)
    assert dev > 0  # invariance is genuinely approximate once ε>0


def test_not_shift_invariant():
    """C3 (negative property): RMSNorm(x + c·1) ≠ RMSNorm(x) — the design
    difference from LayerNorm. Asserts a real, order-1 gap for c=1."""
    x, _ = _data(seed=2)
    one = np.ones(x.shape[-1])
    ref = core_eq.rmsnorm(x, one, eps=0.0)
    shifted = core_eq.rmsnorm(x + 1.0, one, eps=0.0)
    gap = np.max(np.abs(shifted - ref))
    assert gap > 0.1, f"shift changed output only by {gap} — unexpected"


def test_gain_and_broadcasting():
    """g applies per feature; the (..., d) broadcast matches per-token calls."""
    x, g = _data(seed=3, shape=(4, 5, 8))
    batch = core_eq.rmsnorm(x, g)
    for i in range(4):
        for j in range(5):
            assert np.allclose(batch[i, j], core_eq.rmsnorm(x[i, j], g),
                               rtol=0, atol=1e-15)
