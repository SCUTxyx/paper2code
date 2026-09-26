"""Scaled dot-product attention —— 按论文 Eq.(1) 的向量化实现(公式版)。

来源: Vaswani et al., arXiv:1706.03762, §3.2.1 Eq.(1);因果掩码见 §3.2.3。
numpy-only, float64。
"""

import numpy as np


def softmax_rows(S):
    """逐行数值稳定 softmax。"""
    S = S - np.max(S, axis=-1, keepdims=True)
    e = np.exp(S)
    return e / np.sum(e, axis=-1, keepdims=True)


def causal_mask(n):
    """位置 (i, j):j > i 为 True(需屏蔽)。"""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def attention(Q, K, V, causal=False):
    """Eq.(1): A = softmax(QKᵀ/√d_k), out = A V。返回 (out, A)。"""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)                     # Eq.(1): QKᵀ/√d_k
    if causal:
        S = np.where(causal_mask(S.shape[0]), -np.inf, S)   # §3.2.3 掩码
    A = softmax_rows(S)
    return A @ V, A


def attention_grads(Q, K, V, W, causal=False):
    """标量损失 L = Σ W ⊙ out 对 Q/K/V 的解析梯度(T5)。"""
    out, A = attention(Q, K, V, causal)
    dA = W @ V.T                                   # ∂L/∂A = W Vᵀ
    dot = np.sum(A * dA, axis=-1, keepdims=True)
    dS = A * (dA - dot)                            # softmax 反传(掩码处 A=0 自动归零)
    d_k = Q.shape[-1]
    dQ = dS @ (K / np.sqrt(d_k))                   # ∂S/∂Q = K/√d_k
    dK = dS.T @ (Q / np.sqrt(d_k))                 # ∂S/∂K = Q/√d_k
    dV = A.T @ W                                   # ∂L/∂V = Aᵀ W
    return dQ, dK, dV
