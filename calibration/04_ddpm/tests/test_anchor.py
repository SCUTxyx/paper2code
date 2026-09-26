"""DDPM 锚点测试(D1):闭式边缘 vs 逐步加噪的蒙特卡洛统计。

容差按解析公式导出(规范 §2.4,禁拍脑袋):
- 均值:每维 std = √((1-ᾱ_t)/N) → 容差 5σ/√N;
- 方差:高斯样本方差的相对标准误 ≈ √(2/N) → 容差 5·√(2/N)(相对)。
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo

T, D, T_T = 1000, 3, 400   # 调度长度 / 数据维 / 校准时刻
N = 8000                   # 蒙特卡洛轨迹数


def _setup():
    betas = core_eq.linear_beta_schedule(T)
    abars = core_eq.alpha_bar_seq(betas)
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-1.0, 1.0, size=D)   # 论文设定 x0 ∈ [-1,1](§4)
    return betas, abars, x0


def test_closed_form_vs_iterative_monte_carlo():
    """D1: t=400 处,闭式采样与逐步加噪的均值/方差都应落在解析容差内。"""
    betas, abars, x0 = _setup()
    t = T_T
    ab = abars[t - 1]

    rng = np.random.default_rng(42)
    xs_closed = core_pseudo.q_sample_batch(x0, t, rng, abars, N)
    rng = np.random.default_rng(43)
    xs_iter = core_pseudo.q_sample_iterative_batch(x0, t, rng, betas, N)

    tol_mean = 5.0 * np.sqrt((1.0 - ab) / N)          # 每维均值容差
    tol_var = 5.0 * np.sqrt(2.0 / N)                  # 方差相对容差
    for name, xs in (("closed", xs_closed), ("iterative", xs_iter)):
        mean_err = np.abs(xs.mean(axis=0) - np.sqrt(ab) * x0)
        var = xs.var(axis=0)
        assert np.all(mean_err < tol_mean), f"{name} mean err {mean_err} ≥ {tol_mean}"
        assert np.all(np.abs(var - (1.0 - ab)) / (1.0 - ab) < tol_var), (
            f"{name} var rel err {np.abs(var - (1-ab))/(1-ab)} ≥ {tol_var}")


def test_two_paths_agree_in_distribution():
    """D1 强化:两条路径的均值差也应远小于各自容差之和(同分布的交叉印证)。"""
    betas, abars, x0 = _setup()
    t = T_T
    rng = np.random.default_rng(7)
    xs_closed = core_pseudo.q_sample_batch(x0, t, rng, abars, N)
    rng = np.random.default_rng(8)
    xs_iter = core_pseudo.q_sample_iterative_batch(x0, t, rng, betas, N)
    tol = 5.0 * np.sqrt((1.0 - abars[t - 1]) / N)
    diff = np.abs(xs_closed.mean(axis=0) - xs_iter.mean(axis=0))
    assert np.all(diff < np.sqrt(2) * tol), f"两条路径均值差 {diff} 过大"
