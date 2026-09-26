"""Adam — independent formulation: the Algorithm 1 recursion unrolled into a
weighted sum over the gradient history.

Mathematical basis (hand derivation, METHOD_CARD claim A3):
    m_t = (1-β1) Σ_{k=1..t} β1^{t-k} g_k        (first-moment EMA unrolled)
    v_t = (1-β2) Σ_{k=1..t} β2^{t-k} g_k²
Bias correction and the update step are identical to the formula version. The
computation path is entirely different (lower-triangular weighted sums vs state
recursion) — for the cross-check in test_crosscheck.py.
"""

import numpy as np


def _ema_history(seq, beta):
    """Per-step EMA values computed directly as weighted sums over history, no
    recursion. seq: (T, n) -> (T, n)."""
    s = np.asarray(seq, dtype=np.float64)
    T = s.shape[0]
    idx = np.arange(T)
    # W[t, k] = β^{t-k} (k ≤ t), lower triangular; W @ s gives Σ_k β^{t-k} s_k
    W = np.tril(beta ** (idx[:, None] - idx[None, :]))
    return (1.0 - beta) * (W @ s)


def adam_run_history(grad_seq, theta0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    """Equivalent closed-form path: history-weighted sums -> bias correction -> update
    (same interface as core_eq.adam_run)."""
    g = np.asarray(grad_seq, dtype=np.float64)
    T = g.shape[0]
    theta = np.array(theta0, dtype=np.float64)

    m_seq = _ema_history(g, beta1)                    # m_t = (1-β1)Σ β1^{t-k} g_k
    v_seq = _ema_history(g * g, beta2)                # v_t = (1-β2)Σ β2^{t-k} g_k²
    tvec = np.arange(1, T + 1, dtype=np.float64)
    m_hats = m_seq / (1.0 - beta1 ** tvec)[:, None]   # as Algorithm 1 L9
    v_hats = v_seq / (1.0 - beta2 ** tvec)[:, None]   # as Algorithm 1 L10

    thetas = [theta.copy()]
    for t in range(T):
        theta = theta - lr * m_hats[t] / (np.sqrt(v_hats[t]) + eps)  # as L11
        thetas.append(theta.copy())
    return {
        "thetas": np.array(thetas),
        "ms": m_seq,
        "vs": v_seq,
        "m_hats": m_hats,
        "v_hats": v_hats,
    }
