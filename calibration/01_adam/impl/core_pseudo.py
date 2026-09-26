"""Adam 的独立表述实现:把 Algorithm 1 的递推展开成对梯度历史的加权和。

数学依据(手写推导,见 METHOD_CARD claim A3):
    m_t = (1-β1) Σ_{k=1..t} β1^{t-k} g_k        (一阶 EMA 展开)
    v_t = (1-β2) Σ_{k=1..t} β2^{t-k} g_k²
偏差修正与更新步与公式版相同。计算路径与递推版完全不同
(下三角矩阵加权求和 vs 状态递推),供 test_crosscheck.py 互对拍。
"""

import numpy as np


def _ema_history(seq, beta):
    """每步的 EMA 值,由历史直接加权和,不做递推。seq: (T, n) -> (T, n)"""
    s = np.asarray(seq, dtype=np.float64)
    T = s.shape[0]
    idx = np.arange(T)
    # W[t, k] = β^{t-k} (k ≤ t),下三角;W @ s 给出 Σ_k β^{t-k} s_k
    W = np.tril(beta ** (idx[:, None] - idx[None, :]))
    return (1.0 - beta) * (W @ s)


def adam_run_history(grad_seq, theta0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    """等价闭式路径:历史加权和 -> 偏差修正 -> 更新(接口与 core_eq.adam_run 一致)。"""
    g = np.asarray(grad_seq, dtype=np.float64)
    T = g.shape[0]
    theta = np.array(theta0, dtype=np.float64)

    m_seq = _ema_history(g, beta1)                    # m_t = (1-β1)Σ β1^{t-k} g_k
    v_seq = _ema_history(g * g, beta2)                # v_t = (1-β2)Σ β2^{t-k} g_k²
    tvec = np.arange(1, T + 1, dtype=np.float64)
    m_hats = m_seq / (1.0 - beta1 ** tvec)[:, None]   # 同 Algorithm 1 L9
    v_hats = v_seq / (1.0 - beta2 ** tvec)[:, None]   # 同 Algorithm 1 L10

    thetas = [theta.copy()]
    for t in range(T):
        theta = theta - lr * m_hats[t] / (np.sqrt(v_hats[t]) + eps)  # 同 L11
        thetas.append(theta.copy())
    return {
        "thetas": np.array(thetas),
        "ms": m_seq,
        "vs": v_seq,
        "m_hats": m_hats,
        "v_hats": v_hats,
    }
