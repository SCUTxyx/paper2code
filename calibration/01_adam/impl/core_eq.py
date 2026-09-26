"""Adam 更新规则 —— 按论文 Algorithm 1 逐行实现(公式版)。

来源: Kingma & Ba, arXiv:1412.6980, Algorithm 1(注释中的 L 行号指算法框行)。
numpy-only, float64。
"""

import numpy as np


def adam_run(grad_seq, theta0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    """按 Algorithm 1 递推跑完整梯度序列。

    grad_seq: (T, n) 梯度序列;theta0: (n,)。
    返回 dict:thetas (T+1, n), ms/vs (T,n), m_hats/v_hats (T,n)。
    """
    g_seq = np.asarray(grad_seq, dtype=np.float64)
    theta = np.array(theta0, dtype=np.float64)
    m = np.zeros_like(theta)                                   # Algorithm 1 L1: m_0 = 0
    v = np.zeros_like(theta)                                   # Algorithm 1 L1: v_0 = 0
    thetas, ms, vs, m_hats, v_hats = [theta.copy()], [], [], [], []
    for t in range(1, g_seq.shape[0] + 1):
        g = g_seq[t - 1]
        m = beta1 * m + (1 - beta1) * g                        # Algorithm 1 L7-8: m_t
        v = beta2 * v + (1 - beta2) * g * g                    # Algorithm 1 L7-8: v_t
        m_hat = m / (1 - beta1 ** t)                           # Algorithm 1 L9: bias-corrected m
        v_hat = v / (1 - beta2 ** t)                           # Algorithm 1 L10: bias-corrected v
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)    # Algorithm 1 L11: update
        thetas.append(theta.copy())
        ms.append(m.copy())
        vs.append(v.copy())
        m_hats.append(m_hat.copy())
        v_hats.append(v_hat.copy())
    return {
        "thetas": np.array(thetas),
        "ms": np.array(ms),
        "vs": np.array(vs),
        "m_hats": np.array(m_hats),
        "v_hats": np.array(v_hats),
    }
