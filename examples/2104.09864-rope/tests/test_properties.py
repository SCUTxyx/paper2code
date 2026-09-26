"""RoPE 性质测试(R1 正交 / R2 相对位置不变性 / R4 复合律)。"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def _vec(seed, d=8):
    return np.random.default_rng(seed).standard_normal(d)


def test_rotation_preserves_norm():
    """R1: ‖f(x,m)‖ = ‖x‖ 精确(旋转是正交变换)。"""
    x = _vec(0)
    for m in (0, 1, 17, 1000):
        assert np.allclose(np.linalg.norm(core_eq.rope_rotate(x, m)),
                           np.linalg.norm(x), rtol=0, atol=1e-12), f"m={m}"


def test_relative_position_invariance():
    """R2(论文核心声明): ⟨f(q,m),f(k,n)⟩ 只依赖 m−n。
    对多组 (m,n,c) 与多个随机向量对验证,1e-12。"""
    rng = np.random.default_rng(1)
    for _ in range(5):
        q, k = rng.standard_normal(8), rng.standard_normal(8)
        for (m, n) in ((3, 1), (0, 7), (100, 98), (512, 5)):
            c = 37
            s1 = core_eq.rope_score(q, k, m, n)
            s2 = core_eq.rope_score(q, k, m + c, n + c)
            assert abs(s1 - s2) < 1e-12, f"(m,n)=({m},{n}): {s1} vs {s2}"


def test_composition():
    """R4: 先转 m 再转 n = 直接转 m+n。"""
    x = _vec(2)
    m, n = 13, 29
    a = core_eq.rope_rotate(core_eq.rope_rotate(x, m), n)
    b = core_eq.rope_rotate(x, m + n)
    assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_identity_rotation_at_zero():
    """m=0 时 f(x,0) = x 精确(θ 任意,cos0=1, sin0=0)。"""
    x = _vec(3, d=6)
    assert np.allclose(core_eq.rope_rotate(x, 0), x, rtol=0, atol=0)
