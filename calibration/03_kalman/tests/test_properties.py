"""Kalman 性质测试(K3 协方差收缩 / K5 NIS 一致性)。"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def test_covariance_shrinks():
    """K3: 静态模型下 P_t = (P_{t-1}⁻¹ + HᵀR⁻¹H)⁻¹ ⪯ P_{t-1}
    → trace 单调不增(逐差 ≤ 1e-12)。"""
    rng = np.random.default_rng(0)
    H = rng.standard_normal((4, 3))
    ys = rng.standard_normal((15, 4))
    out = core_eq.kf_filter(ys, H, R=np.eye(4), P0=np.eye(3))
    tr = np.trace(out["covs"], axis1=1, axis2=2)
    assert np.all(np.diff(tr) <= 1e-12), f"trace 非单调: {tr}"


def test_nis_consistency():
    """K5: 模型正确设定时 E[NIS] = d_y = 2。
    NIS ~ χ²(2),Var = 4;样本 = 试验数×步数,均值标准误 ≈ √(4/样本数),
    15% 容差 = 0.3 远大于标准误,统计上极显著地应当通过。"""
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
    assert abs(mean_nis - dy) < 0.15 * dy, f"mean NIS = {mean_nis:.4f}, 期望 {dy}"
