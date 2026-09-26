"""Kalman 滤波 —— 标准预测-更新递推实现(公式版)。

来源: Kalman (1960) 滤波方程组(预测 / 增益 / 更新)。
numpy-only, float64。缺省 F=I、Q=0(静态模型)。
"""

import numpy as np


def kf_filter(ys, H, R, F=None, Q=None, x0=None, P0=None):
    """标准 Kalman 递推。

    ys: (T, dy) 观测序列;H: (dy, n);R: (dy, dy)。
    返回 dict:means (T+1, n)、covs (T+1, n, n)、gains (T, n, dy)、nis (T,)。
    """
    ys = np.asarray(ys, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n = H.shape[1]
    F = np.eye(n) if F is None else np.asarray(F, dtype=np.float64)
    Q = np.zeros((n, n)) if Q is None else np.asarray(Q, dtype=np.float64)
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    P = np.eye(n) * 1e10 if P0 is None else np.asarray(P0, dtype=np.float64).copy()

    means, covs, gains, nis = [x.copy()], [P.copy()], [], []
    Rinv = np.linalg.inv(R)
    for t in range(ys.shape[0]):
        # 预测
        x = F @ x
        P = F @ P @ F.T + Q
        # 更新
        S = H @ P @ H.T + R                      # 新息协方差
        K = P @ H.T @ np.linalg.inv(S)           # Kalman 增益
        nu = ys[t] - H @ x                       # 新息
        x = x + K @ nu
        # 协方差更新用 Joseph 形式:与教科书式 (I-KH)P 在最优增益下恒等,
        # 但对大 P0(扩散先验)数值稳定 —— 教科书式在 P0=1e10 时因灾难性
        # 消去产生负定协方差(本考卷 test_anchor.py 记录了该对照)。
        ImKH = np.eye(n) - K @ H
        P = ImKH @ P @ ImKH.T + K @ R @ K.T      # Joseph (1968) 形式
        means.append(x.copy())
        covs.append(P.copy())
        gains.append(K.copy())
        nis.append(float(nu @ np.linalg.solve(S, nu)))
    return {
        "means": np.array(means),
        "covs": np.array(covs),
        "gains": np.array(gains),
        "nis": np.array(nis),
    }
