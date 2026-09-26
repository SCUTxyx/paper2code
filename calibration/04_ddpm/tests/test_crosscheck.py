"""DDPM 双实现互对拍(D2):同噪声序列下,递推 = 未展开加权和 = 闭式(合成噪声)。

D2 是精确恒等(推导见 METHOD_CARD),容差 1e-10;
闭式路径需先把逐步噪声按权重合成 ε_comb,再走 Eq.(4),同样应精确一致。
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

T, D = 1000, 4


def _combined_eps(eps_seq, t, betas):
    """ε_comb = Σ_s √(ᾱ_t/ᾱ_s)·√(1-α_s)·ε_s / √(1-ᾱ_t)。"""
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    acc = np.zeros(D)
    for s in range(1, t + 1):
        coef = np.sqrt(abar[t - 1] / abar[s - 1]) * np.sqrt(1.0 - alphas[s - 1])
        acc += coef * eps_seq[s - 1]
    return acc / np.sqrt(1.0 - abar[t - 1])


def test_iterative_equals_unrolled():
    """D2: 同一 ε_s 前缀下两条路径精确相等。"""
    betas = core_eq.linear_beta_schedule(T)
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-1, 1, D)
    eps_seq = rng.standard_normal((T, D))
    for t in (1, 17, 500, 1000):
        a = core_pseudo.q_sample_iterative(x0, t, eps_seq, betas)
        b = core_pseudo.q_sample_unrolled(x0, t, eps_seq, betas)
        assert np.allclose(a, b, rtol=0, atol=1e-10), f"t={t}: {a} vs {b}"


def test_closed_form_with_combined_eps():
    """闭式 Eq.(4) 用合成噪声 ε_comb 应精确复现逐步递推结果。"""
    betas = core_eq.linear_beta_schedule(T)
    abars = core_eq.alpha_bar_seq(betas)
    rng = np.random.default_rng(1)
    x0 = rng.uniform(-1, 1, D)
    eps_seq = rng.standard_normal((T, D))
    for t in (1, 17, 500, 1000):
        a = core_pseudo.q_sample_iterative(x0, t, eps_seq, betas)
        b = core_eq.q_sample(x0, t, _combined_eps(eps_seq, t, betas), abars)
        assert np.allclose(a, b, rtol=0, atol=1e-10), f"t={t}: {a} vs {b}"
