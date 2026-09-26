"""Attention — independent formulation: per-query loop with explicit normalization
over the allowed key set only.

Difference vs the formula version: no -inf mask matrix; for each query the
"allowed keys" are selected and the softmax is taken within that subset
(a literal transcription of the §3.2.3 mask statement). For the cross-check in
test_crosscheck.py.
"""

import numpy as np


def attention_loop(Q, K, V, causal=False):
    """Same interface as core_eq.attention, implemented per query. Returns (out, A)."""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n_q, d_k = Q.shape[0], Q.shape[-1]
    n_k = K.shape[0]
    out = np.zeros((n_q, V.shape[1]))
    A = np.zeros((n_q, n_k))
    for i in range(n_q):
        scores = np.array([Q[i] @ K[j] for j in range(n_k)]) / np.sqrt(d_k)  # Eq.(1) dot/√d_k
        allowed = np.arange(n_k) <= i if causal else np.ones(n_k, dtype=bool)
        e = np.exp(scores[allowed] - np.max(scores[allowed]))
        a = np.zeros(n_k)
        a[allowed] = e / np.sum(e)               # within-subset normalization = -inf-masked softmax
        A[i] = a
        out[i] = a @ V
    return out, A
