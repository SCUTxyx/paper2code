"""DDPM 前向过程 —— 按论文 Eq.(4) 的闭式边缘实现(公式版)。

来源: Ho et al., arXiv:2006.11239, §2 Eq.(1)(2)(4)、§4 线性调度。
numpy-only, float64。
"""

import numpy as np


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """§4: β_t 线性从 1e-4 增到 0.02,T=1000。返回 (T,),下标 t-1 ↔ 论文 β_t。"""
    return np.linspace(beta_start, beta_end, T)


def alpha_bar_seq(betas):
    """ᾱ_t = Π_{s≤t} α_s,α_t = 1-β_t(Eq.2 下方记号)。"""
    return np.cumprod(1.0 - np.asarray(betas, dtype=np.float64))


def q_sample(x0, t, eps, abars):
    """Eq.(4): x_t = √ᾱ_t x0 + √(1-ᾱ_t) ε。t 为 1-based。"""
    x0 = np.asarray(x0, dtype=np.float64)
    ab = abars[t - 1]
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * np.asarray(eps, dtype=np.float64)
