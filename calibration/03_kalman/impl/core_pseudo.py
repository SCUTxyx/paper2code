"""Kalman 滤波 —— 信息形式(独立表述)。

维护信息矩阵 Λ = P⁻¹ 与信息向量 η = Λ x̂(推导见 METHOD_CARD「等价表述」):
- 预测:协方差域完成(P⁻ = F Λ⁻¹ Fᵀ + Q,再转回信息域);
- 更新:信息域直接累加 Λ += HᵀR⁻¹H、η += HᵀR⁻¹y —— 不经过增益公式;
- 增益用后验恒等式 K = Λ⁻¹HᵀR⁻¹(与标准式 K = P⁻HᵀS⁻¹ 是不同计算路径)。
供 test_crosscheck.py 互对拍。
"""

import numpy as np


def kf_information(ys, H, R, F=None, Q=None, x0=None, P0=None):
    """与 core_eq.kf_filter 同接口、同返回结构。"""
    ys = np.asarray(ys, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n = H.shape[1]
    F = np.eye(n) if F is None else np.asarray(F, dtype=np.float64)
    Q = np.zeros((n, n)) if Q is None else np.asarray(Q, dtype=np.float64)
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    P = np.eye(n) * 1e10 if P0 is None else np.asarray(P0, dtype=np.float64).copy()

    Lam = np.linalg.inv(P)                    # Λ_0 = P0⁻¹
    eta = Lam @ x                             # η_0 = P0⁻¹ x0
    Rinv = np.linalg.inv(R)
    Ht_Rinv_H = H.T @ Rinv @ H
    Ht_Rinv = H.T @ Rinv

    means, covs, gains, nis = [x.copy()], [P.copy()], [], []
    for t in range(ys.shape[0]):
        # 预测(协方差域):先用旧 Λ(后验)恢复 x_post 再外推,顺序不可换
        P_pred = F @ np.linalg.inv(Lam) @ F.T + Q
        x_pred = F @ np.linalg.solve(Lam, eta)
        Lam = np.linalg.inv(P_pred)
        eta = Lam @ x_pred
        # 更新(信息域直接累加)
        Lam = Lam + Ht_Rinv_H
        y = ys[t]
        eta = eta + Ht_Rinv @ y
        x = np.linalg.solve(Lam, eta)         # 后验均值 = Λ⁻¹η
        P = np.linalg.inv(Lam)                # 后验协方差
        # 辅助输出(NIS 与增益,口径与 core_eq 一致)
        S = H @ P_pred @ H.T + R
        nu = y - H @ x_pred
        nis.append(float(nu @ np.linalg.solve(S, nu)))
        gains.append(np.linalg.inv(Lam) @ Ht_Rinv)   # K = Λ⁻¹HᵀR⁻¹(后验恒等式)
        means.append(x.copy())
        covs.append(P.copy())
    return {
        "means": np.array(means),
        "covs": np.array(covs),
        "gains": np.array(gains),
        "nis": np.array(nis),
    }
