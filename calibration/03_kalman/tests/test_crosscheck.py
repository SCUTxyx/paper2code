"""Kalman 双实现互对拍:标准递推(增益公式) vs 信息形式(信息累加)。

覆盖静态(F=I, Q=0)与动态(F≠I, Q≠0)两种情形;
信息形式对线性高斯动态模型同样精确(Λ_t = (FΛ_{t-1}⁻¹Fᵀ+Q)⁻¹ + HᵀR⁻¹H)。
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def _data(seed=0):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((4, 3))
    ys = rng.standard_normal((20, 4))
    R = np.diag([1.0, 0.7, 1.3, 0.4])
    P0 = 0.8 * np.eye(3)
    x0 = rng.standard_normal(3)
    return H, ys, R, P0, x0


def test_static_model():
    H, ys, R, P0, x0 = _data()
    a = core_eq.kf_filter(ys, H, R=R, x0=x0, P0=P0)
    b = core_pseudo.kf_information(ys, H, R=R, x0=x0, P0=P0)
    assert np.allclose(a["means"], b["means"], rtol=1e-9, atol=1e-12)
    assert np.allclose(a["covs"], b["covs"], rtol=1e-9, atol=1e-12)
    assert np.allclose(a["gains"], b["gains"], rtol=1e-9, atol=1e-14)
    assert np.allclose(a["nis"], b["nis"], rtol=1e-9, atol=1e-14)


def test_dynamic_model():
    H, ys, R, P0, x0 = _data(seed=1)
    rng = np.random.default_rng(2)
    th = 0.7  # 旋转 + 轻微收缩的转移矩阵,过程噪声非零
    F = 0.98 * np.array([[np.cos(th), -np.sin(th), 0],
                         [np.sin(th), np.cos(th), 0],
                         [0, 0, 1.0]])
    Q = 0.05 * np.eye(3)
    a = core_eq.kf_filter(ys, H, R=R, F=F, Q=Q, x0=x0, P0=P0)
    b = core_pseudo.kf_information(ys, H, R=R, F=F, Q=Q, x0=x0, P0=P0)
    assert np.allclose(a["means"], b["means"], rtol=1e-9, atol=1e-12)
    assert np.allclose(a["covs"], b["covs"], rtol=1e-9, atol=1e-12)
    assert np.allclose(a["gains"], b["gains"], rtol=1e-9, atol=1e-14)
