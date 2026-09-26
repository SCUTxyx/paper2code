"""Kalman 锚点测试(K1 / K2):递推后验均值 vs 批量闭式解。

K1 的真值来源 np.linalg.lstsq —— 与实现完全独立的权威路径(规划选定的校准点):
线性高斯静态模型 + 扩散先验时,后验 = OLS + O(P0⁻¹) 的先验偏置。

数值发现(记入 calibration/README.md):教科书协方差更新 (I-KH)P 在
P0=1e10 时因灾难性消去产生负定协方差、均值偏离 OLS 达 1e-1;Joseph 形式
(与教科书式在最优增益下代数恒等)在 P0=1e8 时误差 2.4e-8。故 core_eq 采用
Joseph 形式,本锚点取 P0=1e8:先验偏置 ~1e-8/λ_min ≪ rtol,数值误差 ~2e-8。
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq

T, N, DY = 25, 3, 4  # 状态维 3,观测维 4,25 个时刻(设计矩阵满秩)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((DY, N))
    x_true = rng.standard_normal(N)
    ys = H @ x_true + rng.standard_normal((T, DY))
    return H, ys


def test_static_diffuse_prior_equals_lstsq():
    H, ys = _data()
    out = core_eq.kf_filter(ys, H, R=np.eye(DY), P0=1e8 * np.eye(N))
    # 批量最小二乘:把 T 个观测堆叠成 (T·dy, n) 设计矩阵
    design = np.tile(H, (T, 1))
    target = ys.reshape(-1)
    x_ols, *_ = np.linalg.lstsq(design, target, rcond=None)
    assert np.allclose(out["means"][-1], x_ols, rtol=1e-6)


def test_informative_prior_gls_closed_form():
    """K2: 信息先验时后验均值 = (HᵀR⁻¹H + P0⁻¹)⁻¹(HᵀR⁻¹y + P0⁻¹x0)。"""
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
