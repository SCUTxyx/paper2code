"""FlashAttention tiled attention — transliteration of paper Algorithm 1 (formula version).

Sources: Dao et al., arXiv:2205.14135, §3.1 Algorithm 1 (forward); online-softmax trick
from Milakov & Gimelshein, arXiv:1805.02867. The definitional (non-tiled) reference
attention lives here too, marked `direct`, so the repro is self-contained.
numpy-only, float64.
"""

import numpy as np


def _blocks(n, b):
    """Split [0, n) into consecutive blocks of size b (last one may be short)."""
    return [np.arange(s, min(s + b, n)) for s in range(0, n, max(b, 1))]


def attention_direct(Q, K, V, causal=False):
    """Definitional reference: softmax(QKᵀ/√d)·V with a max-subtracted (stable) softmax.

    This is the function the tiled algorithm must reproduce exactly (claim F1).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)                                  # Vaswani 2017 Eq.(1)
    if causal:
        S = np.where(np.triu(np.ones(S.shape, dtype=bool), k=1), -np.inf, S)
    S = S - np.max(S, axis=-1, keepdims=True)                 # shift by row max (F2)
    P = np.exp(S)
    A = P / P.sum(axis=-1, keepdims=True)
    return A @ V


def flash_attention_eq(Q, K, V, block_q=2, block_k=2, causal=False):
    """Tiled attention, normalized-at-every-step form with explicit diag matrices.

        m_new = max(m_old, rowmax(S))
        P_tilde = exp(S − m_new)
        l_new   = e^{m_old−m_new}·l_old + rowsum(P_tilde)
        O_new   = diag(l_new)⁻¹ ( diag(e^{m_old−m_new}·l_old)·O_old + P̃ᵀ V_j )

    The `l_old` factor is forced by algebra: O_old is normalized (O_old = U_old/l_old),
    so rescaling the accumulator by e^{Δm} alone would mix a normalized quantity with
    an unnormalized one — the exact bug our first attempt had, caught immediately by
    claim F1's exactness test (see REPORT finding 1). First block: m_old = −∞ ⇒
    rescale factor 0 exactly and l_old = 0, so O_new = P̃V/l_new (claim F3).
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n_q, d = Q.shape
    n_k = K.shape[0]
    O = np.zeros((n_q, V.shape[1]))
    m = np.full(n_q, -np.inf)
    l = np.zeros(n_q)
    for js in _blocks(n_k, block_k):
        Kj, Vj = K[js], V[js]
        for is_ in _blocks(n_q, block_q):
            S = Q[is_] @ Kj.T / np.sqrt(d)                    # score block
            if causal:
                S = np.where(js[None, :] > is_[:, None], -np.inf, S)   # §3.2 mask
            m_new = np.maximum(m[is_], S.max(axis=1))         # running max
            P_tilde = np.exp(S - m_new[:, None])              # exp(−inf − m_new)=0 on masks
            rescale = np.exp(m[is_] - m_new)                  # F3: e^{Δm}, 0 on first block
            l_new = rescale * l[is_] + P_tilde.sum(axis=1)
            acc = np.diag(rescale * l[is_]) @ O[is_] + P_tilde @ Vj   # e^{Δm}·l_old·O_old + P̃ᵀV
            O[is_] = np.diag(1.0 / l_new) @ acc               # diag(l_new)⁻¹·acc
            m[is_], l[is_] = m_new, l_new
    return O
