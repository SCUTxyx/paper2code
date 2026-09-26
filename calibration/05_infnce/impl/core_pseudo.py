"""CLIP 对比损失 —— 独立表述:逐对循环 + 显式 softmax(InfoNCE §2.2 直译)。

对每个 anchor 单独算「正样本 logit − logsumexp(全部 logits)」,
不构造相似度矩阵,不做矩阵化 softmax。供 test_crosscheck.py 互对拍。
"""

import numpy as np


def clip_loss_loop(I, T, tau=0.07):
    """与 core_eq.clip_loss 同口径,逐对实现。"""
    I = np.asarray(I, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    In = I / np.sqrt((I * I).sum(axis=1, keepdims=True))
    Tn = T / np.sqrt((T * T).sum(axis=1, keepdims=True))
    n = I.shape[0]

    def direction(A, B):
        total = 0.0
        for i in range(n):
            sims = np.array([A[i] @ B[j] for j in range(n)]) / tau
            pos = sims[i]
            lse = np.max(sims) + np.log(np.exp(sims - np.max(sims)).sum())
            total += -(pos - lse)
        return total / n

    return 0.5 * (direction(In, Tn) + direction(Tn, In))
