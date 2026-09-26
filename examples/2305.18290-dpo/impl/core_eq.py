"""DPO 损失 —— 按论文 Eq.(7) 直写实现(公式版),含解析梯度。

来源: Rafailov et al., arXiv:2305.18290, §4 Eq.(7)。
数值实现注意:输入用 **log-prob 差**(log π − log π_ref),不要用概率相除
(下溢与精度损失;数学等价、数值不等价,见 REPORT 发现 1)。
numpy-only, float64。
"""

import numpy as np


def _softplus(x):
    """log(1+e^x),数值稳定。"""
    return np.logaddexp(0.0, x)


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.clip(x, None, 700))),
                    np.exp(np.clip(x, -700, None)) / (1.0 + np.exp(np.clip(x, -700, None))))


def dpo_loss(logp_w, logp_l, logp_w_ref, logp_l_ref, beta=0.1):
    """Eq.(7): z = β[(logπ_w − logπ_ref,w) − (logπ_l − logπ_ref,l)],
    L = −log σ(z) = softplus(−z)。标量或同形数组(批均值)。"""
    logp_w = np.asarray(logp_w, dtype=np.float64)
    logp_l = np.asarray(logp_l, dtype=np.float64)
    logp_w_ref = np.asarray(logp_w_ref, dtype=np.float64)
    logp_l_ref = np.asarray(logp_l_ref, dtype=np.float64)
    z = beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref))   # Eq.(7) σ 的自变量
    return float(np.mean(_softplus(-z)))                         # −log σ(z)


def dpo_loss_and_grad(logp_w, logp_l, logp_w_ref, logp_l_ref, beta=0.1):
    """返回 (L, 各输入的解析梯度)。∂L/∂z = −σ(−z);链式分四路,批内取均值。"""
    logp_w = np.asarray(logp_w, dtype=np.float64)
    logp_l = np.asarray(logp_l, dtype=np.float64)
    logp_w_ref = np.asarray(logp_w_ref, dtype=np.float64)
    logp_l_ref = np.asarray(logp_l_ref, dtype=np.float64)
    z = beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref))
    L = float(np.mean(_softplus(-z)))
    s = _sigmoid(-z) / np.asarray(logp_w).size      # ∂L/∂z(批均值)
    g = {}
    g["logp_w"] = -beta * s                          # ∂z/∂logp_w = +β
    g["logp_l"] = +beta * s                          # ∂z/∂logp_l = −β
    g["logp_w_ref"] = +beta * s
    g["logp_l_ref"] = -beta * s
    return L, g
