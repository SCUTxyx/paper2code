"""Kalman property tests (K3 covariance shrinkage / K5 NIS consistency)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def test_covariance_shrinks():
    """K3: for the static model P_t = (P_{t-1}⁻¹ + HᵀR⁻¹H)⁻¹ ⪯ P_{t-1}
    → trace is monotonically non-increasing (per-step diff ≤ 1e-12)."""
    rng = np.random.default_rng(0)
    H = rng.standard_normal((4, 3))
    ys = rng.standard_normal((15, 4))
    out = core_eq.kf_filter(ys, H, R=np.eye(4), P0=np.eye(3))
    tr = np.trace(out["covs"], axis1=1, axis2=2)
    assert np.all(np.diff(tr) <= 1e-12), f"trace not monotone: {tr}"


def test_nis_consistency():
    """K5: under a correctly specified model, E[NIS] = d_y = 2.
    NIS ~ χ²(2), Var = 4; the standard error of the mean over samples is
    ≈ √(4/samples), so the 15% tolerance (0.3) is far above it — the test must
    pass with high statistical significance if the implementation is right."""
    rng = np.random.default_rng(1)
    dy, n, steps, trials = 2, 2, 15, 1500
    H = rng.standard_normal((dy, n))
    nis_sum, count = 0.0, 0
    for _ in range(trials):
        x_true = rng.standard_normal(n)              # x0 ~ N(0, P0), P0 = I
        ys = x_true @ H.T + rng.standard_normal((steps, dy))
        out = core_eq.kf_filter(ys, H, R=np.eye(dy), P0=np.eye(n))
        nis_sum += out["nis"].sum()
        count += steps
    mean_nis = nis_sum / count
    assert abs(mean_nis - dy) < 0.15 * dy, f"mean NIS = {mean_nis:.4f}, expected {dy}"
