"""Kalman filter — information form (independent formulation).

Maintains the information matrix Λ = P⁻¹ and information vector η = Λ x̂
(derivation in METHOD_CARD "equivalent formulation"):
- predict: done in the covariance domain (P⁻ = F Λ⁻¹ Fᵀ + Q, then back to
  information form);
- update: information accumulates directly, Λ += HᵀR⁻¹H, η += HᵀR⁻¹y — the gain
  formula is never used;
- the gain is computed via the posterior identity K = Λ⁻¹HᵀR⁻¹ (a different
  computation path from the standard K = P⁻HᵀS⁻¹).
For the cross-check in test_crosscheck.py.
"""

import numpy as np


def kf_information(ys, H, R, F=None, Q=None, x0=None, P0=None):
    """Same interface and return structure as core_eq.kf_filter."""
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
        # predict (covariance domain): recover x_post with the OLD Λ first —
        # the order matters (overwrite Λ only after the extrapolation)
        P_pred = F @ np.linalg.inv(Lam) @ F.T + Q
        x_pred = F @ np.linalg.solve(Lam, eta)
        Lam = np.linalg.inv(P_pred)
        eta = Lam @ x_pred
        # update (information domain, direct accumulation)
        Lam = Lam + Ht_Rinv_H
        y = ys[t]
        eta = eta + Ht_Rinv @ y
        x = np.linalg.solve(Lam, eta)         # posterior mean = Λ⁻¹η
        P = np.linalg.inv(Lam)                # posterior covariance
        # auxiliary outputs (NIS and gain, same semantics as core_eq)
        S = H @ P_pred @ H.T + R
        nu = y - H @ x_pred
        nis.append(float(nu @ np.linalg.solve(S, nu)))
        gains.append(np.linalg.inv(Lam) @ Ht_Rinv)   # K = Λ⁻¹HᵀR⁻¹ (posterior identity)
        means.append(x.copy())
        covs.append(P.copy())
    return {
        "means": np.array(means),
        "covs": np.array(covs),
        "gains": np.array(gains),
        "nis": np.array(nis),
    }
