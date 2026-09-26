"""Kalman filter — standard predict/update recursion (formula version).

Source: Kalman (1960) filter equations (predict / gain / update).
numpy-only, float64. Defaults F=I, Q=0 (static model).
"""

import numpy as np


def kf_filter(ys, H, R, F=None, Q=None, x0=None, P0=None):
    """Standard Kalman recursion.

    ys: (T, dy) observation sequence; H: (dy, n); R: (dy, dy).
    Returns dict: means (T+1, n), covs (T+1, n, n), gains (T, n, dy), nis (T,).
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
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        # update
        S = H @ P @ H.T + R                      # innovation covariance
        K = P @ H.T @ np.linalg.inv(S)           # Kalman gain
        nu = ys[t] - H @ x                       # innovation
        x = x + K @ nu
        # covariance update in Joseph form: algebraically identical to the textbook
        # (I-KH)P at the optimal gain, but numerically stable for large P0 (diffuse
        # priors) — the textbook form produces an indefinite covariance via
        # catastrophic cancellation at P0=1e10 (documented in test_anchor.py).
        ImKH = np.eye(n) - K @ H
        P = ImKH @ P @ ImKH.T + K @ R @ K.T      # Joseph (1968) form
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
