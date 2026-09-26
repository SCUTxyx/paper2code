"""DDPM 前向过程 —— 独立表述:逐步递推(Eq.2)与未展开加权和(推导见 METHOD_CARD)。

不使用闭式系数 √ᾱ_t 直接合成,而是逐级加噪 / 对历史噪声加权求和,
供 test_crosscheck.py 与公式版互对拍(D2 精确恒等)。
"""

import numpy as np


def q_sample_iterative(x0, t, eps_seq, betas):
    """Eq.(2) 逐步加噪: x_s = √(1-β_s) x_{s-1} + √β_s ε_s,用 eps_seq 前 t 个。"""
    x = np.array(x0, dtype=np.float64)
    for s in range(1, t + 1):
        e = np.asarray(eps_seq[s - 1], dtype=np.float64)
        x = np.sqrt(1.0 - betas[s - 1]) * x + np.sqrt(betas[s - 1]) * e
    return x


def q_sample_unrolled(x0, t, eps_seq, betas):
    """未展开恒等式: x_t = √ᾱ_t x0 + Σ_s √(ᾱ_t/ᾱ_s)·√(1-α_s)·ε_s。"""
    betas = np.asarray(betas, dtype=np.float64)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    out = np.sqrt(abar[t - 1]) * np.array(x0, dtype=np.float64)
    for s in range(1, t + 1):
        coef = np.sqrt(abar[t - 1] / abar[s - 1]) * np.sqrt(1.0 - alphas[s - 1])
        out = out + coef * np.asarray(eps_seq[s - 1], dtype=np.float64)
    return out


def q_sample_batch(x0, t, rng, abars, n):
    """闭式边缘的蒙特卡洛:x0 广播到 n 条轨迹(Eq.4 采样形式,向量化)。"""
    x0 = np.asarray(x0, dtype=np.float64)
    ab = abars[t - 1]
    eps = rng.standard_normal((n, x0.shape[-1]))
    return np.sqrt(ab) * x0[None, :] + np.sqrt(1.0 - ab) * eps


def q_sample_iterative_batch(x0, t, rng, betas, n):
    """逐步加噪的蒙特卡洛(轨迹级向量化,时间维循环)。"""
    x = np.tile(np.asarray(x0, dtype=np.float64), (n, 1))
    for s in range(1, t + 1):
        eps = rng.standard_normal(x.shape)
        x = np.sqrt(1.0 - betas[s - 1]) * x + np.sqrt(betas[s - 1]) * eps
    return x
