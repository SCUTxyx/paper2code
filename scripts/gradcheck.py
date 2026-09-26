"""通用有限差分梯度检查工具 —— 所有 repro 与考卷复用(验证阶梯 L2 的核心件)。

数值规范(references/verification.md §数值规范):
- float64、中心差分;
- 标量损失 L: R^n -> R 的解析梯度与数值梯度逐分量比对;
- 相对误差阈值默认 1e-6;
- 步长 h_i = cbrt(float64 eps) * max(1, |x_i|) ≈ 6e-6·max(1,|x_i|)
  —— 中心差分的最优步长在「截断误差 O(h²)」与「舍入误差 O(ε/h)」的
  平衡点 cbrt(ε) 处,不是前向差分的 sqrt(ε)。
"""

from __future__ import annotations

import numpy as np

__all__ = ["central_diff_grad", "max_rel_err", "assert_grad_close"]


def central_diff_grad(f, x, h=None):
    """f: 接受与 x 同形状的 ndarray、返回标量;返回中心差分梯度(与 x 同形状)。"""
    x = np.asarray(x, dtype=np.float64)
    if h is None:
        h = np.cbrt(np.finfo(np.float64).eps) * np.maximum(1.0, np.abs(x))
    flat = x.ravel()
    g = np.zeros_like(flat)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + h.ravel()[i]
        fp = f(flat.reshape(x.shape))
        flat[i] = orig - h.ravel()[i]
        fm = f(flat.reshape(x.shape))
        flat[i] = orig
        g[i] = (fp - fm) / (2.0 * h.ravel()[i])
    return g.reshape(x.shape)


def max_rel_err(numeric, analytic, atol=1e-12):
    """逐分量相对误差,分母带下限,避免纯零比较放大噪声。"""
    numeric = np.asarray(numeric, dtype=np.float64)
    analytic = np.asarray(analytic, dtype=np.float64)
    denom = np.maximum(np.maximum(np.abs(numeric), np.abs(analytic)), atol)
    return float(np.max(np.abs(numeric - analytic) / denom))


def assert_grad_close(f, x, analytic, rtol=1e-6, name="grad"):
    """解析梯度 analytic 与中心差分比对,超过 rtol 抛 AssertionError。

    通过时返回最大相对误差(供 REPORT 的结果矩阵引用)。
    """
    numeric = central_diff_grad(f, x)
    analytic = np.asarray(analytic, dtype=np.float64)
    if numeric.shape != analytic.shape and numeric.size == analytic.size:
        numeric = numeric.reshape(analytic.shape)  # 允许调用方给矩阵形解析梯度
    err = max_rel_err(numeric, analytic)
    if err > rtol:
        numeric = np.asarray(numeric)
        analytic = np.asarray(analytic, dtype=np.float64)
        idx = np.unravel_index(
            np.argmax(np.abs(numeric - analytic)
                      / np.maximum(np.abs(numeric) + np.abs(analytic), 1e-12)),
            np.asarray(analytic).shape)
        raise AssertionError(
            f"{name}: max rel err {err:.3e} > rtol {rtol:g} at {idx} "
            f"(analytic={analytic[idx]:.6e}, numeric={numeric[idx]:.6e})")
    return err
