"""Kalman 梯度检查(K4):x_T 对全部观测的雅可比。

手写推导(信息形式,METHOD_CARD K4):
  x_T = Λ_T⁻¹(P0⁻¹x0 + Σ_t HᵀR⁻¹y_t),Λ_T = P0⁻¹ + T·HᵀR⁻¹H
  ⇒ ∂x_T/∂y_t = Λ_T⁻¹HᵀR⁻¹(t = 1..T 共用同一矩阵)。
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq

T, N, DY = 8, 3, 4


def test_jacobian_wrt_observations():
    rng = np.random.default_rng(0)
    H = rng.standard_normal((DY, N))
    ys = rng.standard_normal((T, DY))
    P0 = 0.7 * np.eye(N)
    x0 = np.array([0.3, -0.2, 0.9])
    R = np.diag([1.0, 0.6, 0.9, 1.2])

    Rinv = np.linalg.inv(R)
    Lam = np.linalg.inv(P0) + T * (H.T @ Rinv @ H)
    M = np.linalg.solve(Lam, H.T @ Rinv)         # ∂x_T/∂y_t (n × dy)
    analytic = np.tile(M, (1, T))                # (n, T·dy) 按观测块平铺

    def f(y_flat):
        out = core_eq.kf_filter(y_flat.reshape(T, DY), H, R=R, x0=x0, P0=P0)
        return float(out["means"][-1][0])        # 取均值第一个分量做标量损失

    err = assert_grad_close(f, ys.ravel(), analytic[0], rtol=1e-6,
                            name="d x_T[0] / d y")
    print(f"\n[kalman] jacobian max rel err = {err:.2e}")
