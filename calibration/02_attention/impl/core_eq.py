"""Scaled dot-product attention — vectorized implementation of paper Eq.(1) (formula version).

Source: Vaswani et al., arXiv:1706.03762, §3.2.1 Eq.(1); causal mask per §3.2.3.
numpy-only, float64.
"""

import numpy as np


def softmax_rows(S):
    """Row-wise numerically stable softmax."""
    S = S - np.max(S, axis=-1, keepdims=True)
    e = np.exp(S)
    return e / np.sum(e, axis=-1, keepdims=True)


def causal_mask(n):
    """Position (i, j): True where j > i (must be masked)."""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def attention(Q, K, V, causal=False):
    """Eq.(1): A = softmax(QKᵀ/√d_k), out = A V. Returns (out, A)."""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)                     # Eq.(1): QKᵀ/√d_k
    if causal:
        S = np.where(causal_mask(S.shape[0]), -np.inf, S)   # §3.2.3 mask
    A = softmax_rows(S)
    return A @ V, A


def attention_grads(Q, K, V, W, causal=False):
    """Analytic gradients of the scalar loss L = Σ W ⊙ out w.r.t. Q/K/V (T5)."""
    out, A = attention(Q, K, V, causal)
    dA = W @ V.T                                   # ∂L/∂A = W Vᵀ
    dot = np.sum(A * dA, axis=-1, keepdims=True)
    dS = A * (dA - dot)                            # softmax backward (masked A=0 contributes 0)
    d_k = Q.shape[-1]
    dQ = dS @ (K / np.sqrt(d_k))                   # ∂S/∂Q = K/√d_k
    dK = dS.T @ (Q / np.sqrt(d_k))                 # ∂S/∂K = Q/√d_k
    dV = A.T @ W                                   # ∂L/∂V = Aᵀ W
    return dQ, dK, dV
