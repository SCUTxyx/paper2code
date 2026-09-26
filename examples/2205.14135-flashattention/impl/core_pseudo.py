"""FlashAttention tiled attention — independent formulation: streaming accumulator.

Difference vs the formula version (core_eq): the output accumulator stays
UNNORMALIZED while streaming (no diag matrices, no per-step division) and is
divided by the final normalizer once, at the end — the form used in practice.
Same recurrence, different computation path. For the cross-check in
test_crosscheck.py.
"""

import numpy as np


def _blocks(n, b):
    return [np.arange(s, min(s + b, n)) for s in range(0, n, max(b, 1))]


def flash_attention_stream(Q, K, V, block_q=2, block_k=2, causal=False):
    """Streaming form: maintain (m, l, U) with U unnormalized; O = U / l at the end.

        m_new = max(m_old, rowmax(S))
        U_new = e^{m_old−m_new}·U_old + exp(S − m_new)ᵀ V_j
        l_new = e^{m_old−m_new}·l_old + rowsum(exp(S − m_new))

    First block: m_old = −∞ ⇒ rescale exactly 0; U_old = l_old = 0 (claim F3).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n_q, d = Q.shape
    n_k = K.shape[0]
    U = np.zeros((n_q, V.shape[1]))
    m = np.full(n_q, -np.inf)
    l = np.zeros(n_q)
    for js in _blocks(n_k, block_k):
        Kj, Vj = K[js], V[js]
        for is_ in _blocks(n_q, block_q):
            S = Q[is_] @ Kj.T / np.sqrt(d)
            if causal:
                S = np.where(js[None, :] > is_[:, None], -np.inf, S)
            m_new = np.maximum(m[is_], S.max(axis=1))
            P_tilde = np.exp(S - m_new[:, None])
            rescale = np.exp(m[is_] - m_new)
            U[is_] = rescale[:, None] * U[is_] + P_tilde @ Vj    # row-scaling, no diag()
            l[is_] = rescale * l[is_] + P_tilde.sum(axis=1)
            m[is_] = m_new
    return U / l[:, None]                                        # single final division


def attention_naive_unstable(Q, K, V):
    """Deliberately naive softmax WITHOUT the max shift — used only by the
    stability test to document WHY the running max matters (overflow at
    |logit| ≳ 709 in float64). Not a reference for anything else.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    A = np.exp(S) / np.exp(S).sum(axis=-1, keepdims=True)        # overflows for large |S|
    return A @ V
