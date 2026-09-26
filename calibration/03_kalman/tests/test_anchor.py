"""Kalman anchor tests (K1 / K2): recursion posterior mean vs batch closed forms.

K1's truth source is np.linalg.lstsq — an authoritative path fully independent of
the implementation (the calibration point chosen in the plan): under a static
linear-Gaussian model with a diffuse prior, the posterior = OLS + O(P0⁻¹) prior bias.

Numeric finding (recorded in calibration/README.md): the textbook covariance
update (I-KH)P is catastrophically cancelled at P0=1e10 — the covariance turns
indefinite and the mean drifts ~1e-1 from OLS. The Joseph form (algebraically
identical at the optimal gain) gives 2.4e-8 at P0=1e8. core_eq therefore uses the
Joseph form and this anchor takes P0=1e8: prior bias ~1e-8/λ_min ≪ rtol and the
numerical error is ~2e-8.
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq

T, N, DY = 25, 3, 4  # state dim 3, observation dim 4, 25 time steps (full-rank design)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((DY, N))
    x_true = rng.standard_normal(N)
    ys = H @ x_true + rng.standard_normal((T, DY))
    return H, ys


def test_static_diffuse_prior_equals_lstsq():
    H, ys = _data()
    out = core_eq.kf_filter(ys, H, R=np.eye(DY), P0=1e8 * np.eye(N))
    # batch least squares: stack the T observations into a (T·dy, n) design matrix
    design = np.tile(H, (T, 1))
    target = ys.reshape(-1)
    x_ols, *_ = np.linalg.lstsq(design, target, rcond=None)
    assert np.allclose(out["means"][-1], x_ols, rtol=1e-6)


def test_informative_prior_gls_closed_form():
    """K2: with an informative prior the posterior mean is the GLS closed form
    (HᵀR⁻¹H + P0⁻¹)⁻¹(HᵀR⁻¹y + P0⁻¹x0)."""
    H, ys = _data(seed=1)
    P0 = 0.5 * np.eye(N)
    x0 = np.array([0.3, -0.2, 0.9])
    R = np.diag([1.0, 0.7, 1.3, 0.4])
    out = core_eq.kf_filter(ys, H, R=R, x0=x0, P0=P0)
    P0inv = np.linalg.inv(P0)
    Rinv = np.linalg.inv(R)
    Lam = P0inv + T * (H.T @ Rinv @ H)
    eta = P0inv @ x0 + sum(H.T @ Rinv @ y for y in ys)
    x_gls = np.linalg.solve(Lam, eta)
    assert np.allclose(out["means"][-1], x_gls, rtol=1e-9)
