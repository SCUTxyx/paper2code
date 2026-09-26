"""Adam 性质测试(claim A1 / A2 / A4)。"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]  # 同名 impl 模块跨 repro 冲突防护
import numpy as np

import core_eq


def test_first_step_size_independent_of_scale():
    """A1: 首步 m̂_1=g、v̂_1=g²,更新 = α·g/(|g|+ε) ≈ α,与 |g| 无关。"""
    g = np.array([3.7, -2.1, 0.9])
    out = core_eq.adam_run(g[None, :], theta0=np.zeros(3), lr=0.1)
    step = out["thetas"][1] - out["thetas"][0]
    assert np.allclose(np.abs(step), 0.1, rtol=1e-6), f"|step|={np.abs(step)}"
    assert np.all(np.sign(step) == np.sign(-g))


def test_beta_to_zero_degenerates():
    """A2: β→0 时矩退化为当前梯度,更新应为 α·g/(|g|+ε)(手写参照)。"""
    g = np.array([0.5, -1.3, 2.0])
    eps = 1e-8
    out = core_eq.adam_run(g[None, :], theta0=np.zeros(3), lr=0.1,
                           beta1=1e-12, beta2=1e-12, eps=eps)
    step = out["thetas"][1] - out["thetas"][0]
    expected = -0.1 * g / (np.abs(g) + eps)  # 手写参照,非实现复述
    assert np.allclose(step, expected, rtol=1e-12, atol=1e-15)
    # 与符号型更新的偏差有解析上界 α·ε/(|g|+ε) ≤ α·ε/0.5
    assert np.max(np.abs(step - (-0.1 * np.sign(g)))) < 0.1 * eps / 0.5


def test_constant_gradient_exact():
    """A4: 常梯度下 m̂_t=g、v̂_t=g² 对任意 t 精确成立(几何级数与偏差修正相消)。"""
    c = np.array([0.7, -0.4])
    T = 200
    seq = np.tile(c, (T, 1))
    out = core_eq.adam_run(seq, theta0=np.zeros(2), lr=0.1)
    assert np.allclose(out["m_hats"], c[None, :], rtol=0, atol=1e-12)
    assert np.allclose(out["v_hats"], (c * c)[None, :], rtol=0, atol=1e-12)
    # 每步更新恒为 α·c/(|c|+ε) → θ_t 线性
    per_step = -0.1 * c / (np.abs(c) + 1e-8)
    expect = np.array([per_step * t for t in range(T + 1)])
    assert np.allclose(out["thetas"], expect, rtol=0, atol=1e-12)


def test_zero_gradient_stability():
    """零梯度下更新恰为零(分母由 ε 兜底,不得产生 NaN)。"""
    out = core_eq.adam_run(np.zeros((5, 3)), theta0=np.array([1.0, 2.0, 3.0]))
    assert np.allclose(out["thetas"], np.array([[1.0, 2.0, 3.0]] * 6), rtol=0, atol=0)
