"""CLIP contrastive loss — vectorized implementation of paper Eq.(1)(2) (formula
version), with analytic gradients.

Sources: Radford et al., arXiv:2103.00020 §2.3; Oord et al., arXiv:1807.03748 §2.2.
numpy-only, float64.
"""

import numpy as np


def _normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _log_softmax_ce(S):
    """Cross-entropy with the diagonal as labels: -1/N Σ_i (S[i,i] - logsumexp(S[i]))."""
    n = S.shape[0]
    m = np.max(S, axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(S - m).sum(axis=1))
    return float(-(np.diag(S) - lse).mean())


def _softmax(S):
    S = S - np.max(S, axis=1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(axis=1, keepdims=True)


def clip_loss(I, T, tau=0.07, return_grads=False):
    """Eq.(2): L = (L_i2t + L_t2i)/2, with L_i2t per Eq.(1).

    return_grads=True also returns (L, dL/dI, dL/dT) — analytic, including the
    L2-normalization projection (N5).
    """
    I = np.asarray(I, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    assert I.shape == T.shape
    In, Tn = _normalize(I), _normalize(T)
    S = In @ Tn.T / tau                                   # Eq.(1): sim/τ
    L_i2t = _log_softmax_ce(S)                            # Eq.(1)
    L_t2i = _log_softmax_ce(S.T)                          # Eq.(1), symmetric direction
    L = 0.5 * (L_i2t + L_t2i)                             # Eq.(2)
    if not return_grads:
        return L
    n = S.shape[0]
    P, Pt = _softmax(S), _softmax(S.T)
    E = np.eye(n)
    # dL/dS = ½[(P-E)/N + ((Pt-E)/N)ᵀ] (the second term chains through CE(Sᵀ))
    dS = 0.5 * ((P - E) / n + ((Pt - E) / n).T)
    dIn = dS @ Tn / tau                                   # S = In Tnᵀ/τ
    dTn = dS.T @ In / tau
    # normalization chain rule: X = r·X̂ ⇒ ∂L/∂X = (∂L/∂X̂ - X̂·rowsum(∂L/∂X̂⊙X̂)) / r
    dI = (dIn - In * np.sum(dIn * In, axis=1, keepdims=True)) \
        / np.linalg.norm(I, axis=1, keepdims=True)
    dT = (dTn - Tn * np.sum(dTn * Tn, axis=1, keepdims=True)) \
        / np.linalg.norm(T, axis=1, keepdims=True)
    return L, dI, dT
