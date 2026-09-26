"""RoPE —— 按论文 Eq.(13)-(15) 的按块旋转实现(公式版)。

来源: Su et al., RoFormer, arXiv:2104.09864, §3.2.2 Eq.(13)(14)(15)。
numpy-only, float64。索引约定:论文 θ_i 用 1-based i=1..d/2(Eq.15 的 i-1),
本实现代码内 0-based i 对应论文 i-1,两者一致(常见 off-by-one 见 REPORT 发现 1)。
"""

import numpy as np


def theta_seq(d, base=10000.0):
    """Eq.(15): θ_i = base^{-2(i-1)/d}, i=1..d/2。返回 (d/2,)。"""
    i = np.arange(d // 2)                      # 代码 0-based ↔ 论文 i-1
    return base ** (-2.0 * i / d)              # Eq.(15)


def rope_rotate(x, m, base=10000.0):
    """Eq.(13)(14): x 按 2 维块旋转 m 步。x 形状 (..., d),d 为偶数。"""
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    ang = m * theta_seq(d, base)               # 每块的旋转角 m·θ_i
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]        # 每块的 (a_i, b_i)
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin       # Eq.(14) 块内第一行
    out[..., 1::2] = x1 * sin + x2 * cos       # Eq.(14) 块内第二行
    return out


def rope_score(q, k, m, n, base=10000.0):
    """注意力打分 ⟨f(q,m), f(k,n)⟩。"""
    return float(rope_rotate(q, m, base) @ rope_rotate(k, n, base))
