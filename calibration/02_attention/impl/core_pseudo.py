"""Attention 的独立表述实现:逐查询循环 + 只对未掩码集合显式归一化。

与公式版的差别:不做 -inf 掩码矩阵,而是对每个查询单独取「允许的键集合」,
在该子集上做 max 减除与指数归一化(§3.2.3 掩码声明的直译)。
供 test_crosscheck.py 互对拍。
"""

import numpy as np


def attention_loop(Q, K, V, causal=False):
    """与 core_eq.attention 同接口,逐查询实现。返回 (out, A)。"""
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n_q, d_k = Q.shape[0], Q.shape[-1]
    n_k = K.shape[0]
    out = np.zeros((n_q, V.shape[1]))
    A = np.zeros((n_q, n_k))
    for i in range(n_q):
        scores = np.array([Q[i] @ K[j] for j in range(n_k)]) / np.sqrt(d_k)  # Eq.(1) 内积/√d_k
        allowed = np.arange(n_k) <= i if causal else np.ones(n_k, dtype=bool)
        e = np.exp(scores[allowed] - np.max(scores[allowed]))
        a = np.zeros(n_k)
        a[allowed] = e / np.sum(e)               # 子集内归一化 = softmax 含 -inf 掩码
        A[i] = a
        out[i] = a @ V
    return out, A
