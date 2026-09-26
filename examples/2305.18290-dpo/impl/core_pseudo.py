"""DPO —— 独立表述:Bradley–Terry 路径(Eq.6→7 的推导顺序)与最优策略(Eq.4)(5)。

公式版直接写 Eq.(7) 的 z 与 softplus;本实现按论文推导顺序:
先算偏好概率 p = σ(z),损失 = −log p(逐样本),再配套 Eq.(4)(5) 的
最优策略构造与奖励反演,供 test_crosscheck.py 互对拍。
"""

import numpy as np


def dpo_loss_bt(logp_w, logp_l, logp_w_ref, logp_l_ref, beta=0.1):
    """Eq.(6)→(7) 路径:p = σ(z)(逐样本),L = −log p 的批均值。"""
    logp_w = np.asarray(logp_w, dtype=np.float64)
    logp_l = np.asarray(logp_l, dtype=np.float64)
    logp_w_ref = np.asarray(logp_w_ref, dtype=np.float64)
    logp_l_ref = np.asarray(logp_l_ref, dtype=np.float64)
    total = 0.0
    for i in range(logp_w.size):
        z = beta * ((logp_w.ravel()[i] - logp_w_ref.ravel()[i])
                    - (logp_l.ravel()[i] - logp_l_ref.ravel()[i]))
        p = 1.0 / (1.0 + np.exp(-z))             # σ(z),Bradley–Terry Eq.(6)
        total += -np.log(p)
    return float(total / logp_w.size)


def optimal_policy(r, logp_ref, beta=0.1):
    """Eq.(4): π*(y|x) = π_ref(y|x)·exp(r/β) / Z(x)。
    r: (K,) 每个候选回答的奖励;logp_ref: (K,) log π_ref。返回 (π* (K,), logZ)。"""
    r = np.asarray(r, dtype=np.float64)
    logp_ref = np.asarray(logp_ref, dtype=np.float64)
    logits = logp_ref + r / beta                 # log π_ref + r/β
    m = logits.max()                             # 数值稳定地构造 logZ
    logZ = m + np.log(np.exp(logits - m).sum())
    return np.exp(logits - logZ), float(logZ)


def reward_from_policy(logp, logp_ref, beta=0.1, logZ=0.0):
    """Eq.(5): r = β·log(π/π_ref) + β·logZ。逐点奖励恢复。"""
    logp = np.asarray(logp, dtype=np.float64)
    logp_ref = np.asarray(logp_ref, dtype=np.float64)
    return beta * (logp - logp_ref) + beta * logZ
