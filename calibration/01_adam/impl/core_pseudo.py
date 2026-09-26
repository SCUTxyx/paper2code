"""Adam — independent formulation: the Algorithm 1 recursion unrolled into a
weighted sum over the gradient history.

Mathematical basis (hand derivation, METHOD_CARD claim A3):
    m_t = (1-β1) Σ_{k=1..t} β1^{t-k} g_k        (first-moment EMA unrolled)
    v_t = (1-β2) Σ_{k=1..t} β2^{t-k} g_k²
Bias correction and the update step are identical to the formula version. The
computation path is entirely different (explicit per-step weighted accumulation
vs state recursion) — for the cross-check in test_crosscheck.py.

Implementation note: the accumulation deliberately avoids BLAS matmul — pip-wheel
numpy's OpenBLAS kernel emits spurious divide-by-zero/overflow FP flags on some
matmul shapes (results are bit-correct; surfaced by the fresh-clone test). Pure
numpy elementwise loops are warning-free on every BLAS.
"""

import numpy as np


def _ema_history(seq, beta):
    """Per-step EMA values computed directly as weighted sums over history, no
    recursion. seq: (T, n) -> (T, n)."""
    s = np.asarray(seq, dtype=np.float64)
    T, n = s.shape
    out = np.empty_like(s)
    for t in range(T):
        acc = np.zeros(n)
        decay = 1.0                      # β^{t-k}, k running down from t
        for k in range(t, -1, -1):
            acc += decay * s[k]
            decay *= beta
        out[t] = (1.0 - beta) * acc
    return out


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
