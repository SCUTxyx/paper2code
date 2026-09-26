"""RoPE 双实现互对拍(R3):按块切片 vs 显式矩阵 vs 复数形式。"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def test_slice_vs_matrix_vs_complex():
    """三条独立计算路径对同一 (x, m) 给出相同结果(1e-12)。"""
    rng = np.random.default_rng(0)
    for d in (2, 4, 8, 16):
        x = rng.standard_normal(d)
        for m in (0, 1, 7, 128, -5):
            a = core_eq.rope_rotate(x, m)
            b = core_pseudo.rope_rotate_matrix(x, m)
            c = core_pseudo.rope_rotate_complex(x, m)
            assert np.allclose(a, b, rtol=0, atol=1e-12), f"d={d}, m={m}"
            assert np.allclose(a, c, rtol=0, atol=1e-12), f"d={d}, m={m}"


def test_batched_rotate_matches_matrix():
    """批量(序列维)形式:core_eq 支持 (..., d) 广播,应与逐向量一致。"""
    rng = np.random.default_rng(1)
    xs = rng.standard_normal((6, 8))       # 6 个 token
    ms = np.array([0, 1, 2, 3, 4, 5])
    batch = core_eq.rope_rotate(xs[:, None, :], ms[:, None, None]).squeeze(1)
    for i in range(6):
        assert np.allclose(batch[i], core_eq.rope_rotate(xs[i], int(ms[i])),
                           rtol=0, atol=1e-12)
