"""gradcheck.py 自身的回归测试 —— 验证工具必须先验证自己。

包括反向测试:喂一个错误梯度,assert_grad_close 必须抛 AssertionError
(一个从不报错的梯度检查器等于没有检查器)。
"""

import numpy as np
import pytest

from gradcheck import assert_grad_close, central_diff_grad, max_rel_err


def test_linear_function_exact():
    """f(x) = a·x 的梯度恒为 a。注意:中心差分并非机器精确——f 本身的舍入
    (|f|·eps/(2h))给出 ~1e-10 的相对误差地板,这是有限差分的真实水位。"""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(6)
    err = assert_grad_close(lambda x: float(a @ x), rng.standard_normal(6),
                            a, rtol=1e-8, name="linear")
    assert err < 1e-9


def test_quadratic_function():
    """f(x) = xᵀAx 的梯度 = (A + Aᵀ)x。"""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4))
    x = rng.standard_normal(4)
    analytic = (A + A.T) @ x
    err = assert_grad_close(lambda v: float(v @ A @ v), x, analytic, rtol=1e-6)
    assert err < 1e-9


def test_catches_wrong_gradient():
    """反向测试:梯度被扰动 1% 的分量必须被抓出来(检查器不能是哑炮)。"""
    rng = np.random.default_rng(2)
    a = rng.standard_normal(5)
    x = rng.standard_normal(5)
    wrong = a.copy()
    wrong[2] *= 1.01  # 1% 扰动,远大于 1e-6
    with pytest.raises(AssertionError, match="max rel err"):
        assert_grad_close(lambda v: float(a @ v), x, wrong, rtol=1e-6)


def test_catches_missing_component():
    """反向测试:梯度漏掉一个非零分量(常见实现 bug)必须被抓出来。"""
    rng = np.random.default_rng(3)
    a = rng.standard_normal(4)
    wrong = a.copy()
    wrong[1] = 0.0
    with pytest.raises(AssertionError):
        assert_grad_close(lambda v: float(a @ v), rng.standard_normal(4),
                          wrong, rtol=1e-6)


def test_step_size_is_cbrt_eps():
    """中心差分最优步长 = cbrt(eps)·max(1,|x|),不是前向差分的 sqrt(eps)。
    用同量级的 x:若混有大量级分量(如 [1,100]),小分量方向的差分会被
    大分量的 ulp 淹没,误差地板升到 ~1e-8 —— 这正是混合量级问题要分开检的教训。"""
    x = np.array([1.0, 2.0])
    g = central_diff_grad(lambda v: float(np.sum(v * v)), x)
    expected = 2.0 * x
    assert max_rel_err(g, expected) < 1e-9
    # 显式验证 h 公式:与 sqrt(eps) 相差 cbrt(eps)/sqrt(eps) ≈ 400 倍
    h = np.cbrt(np.finfo(np.float64).eps) * np.maximum(1.0, np.abs(x))
    assert h[0] / (np.sqrt(np.finfo(np.float64).eps)) > 100.0
