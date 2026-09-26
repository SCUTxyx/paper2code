"""RoPE —— 独立表述:显式块对角矩阵(Eq.14 原文形态)与复数形式(§3.2.1)。

公式版(core_eq)用切片按块旋转;本实现两条独立路径:
1) 显式构造完整 R^m_Θ 矩阵再做矩阵乘法;
2) d=2 时按 §3.2.1 复数表述 q·e^{imθ} 逐块乘法。
供 test_crosscheck.py 互对拍(R3)。
"""

import numpy as np


def rotation_matrix(m, d, base=10000.0):
    """Eq.(14) 原文形态:显式块对角 R^m_Θ,(d, d)。"""
    half = d // 2
    i = np.arange(half)
    thetas = base ** (-2.0 * i / d)            # Eq.(15)
    ang = m * thetas
    cos, sin = np.cos(ang), np.sin(ang)
    R = np.zeros((d, d))
    R[0::2, 0::2] = np.diag(cos)               # 块内 (1,1)
    R[0::2, 1::2] = np.diag(-sin)              # 块内 (1,2)
    R[1::2, 0::2] = np.diag(sin)               # 块内 (2,1)
    R[1::2, 1::2] = np.diag(cos)               # 块内 (2,2)
    return R


def rope_rotate_matrix(x, m, base=10000.0):
    """f(x, m) = R^m_Θ x(显式矩阵路径)。"""
    return rotation_matrix(m, np.asarray(x).shape[-1], base) @ np.asarray(x, dtype=np.float64)


def rope_rotate_complex(x, m, base=10000.0):
    """§3.2.1 复数表述:每块 (a,b) 视作 a+bi,乘 e^{imθ_i}。"""
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    half = d // 2
    thetas = base ** (-2.0 * np.arange(half) / d)   # Eq.(15)
    ang = m * thetas
    out = np.empty_like(x)
    for i in range(half):
        a, b = x[..., 2 * i], x[..., 2 * i + 1]
        z = (a + 1j * b) * np.exp(1j * ang[i])      # q·e^{imθ}(§3.2.1)
        out[..., 2 * i] = z.real
        out[..., 2 * i + 1] = z.imag
    return out
