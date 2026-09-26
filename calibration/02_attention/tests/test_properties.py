"""Attention 性质测试(T1 行和 / T2 因果掩码 / T3 缩放方差 / T4 置换等变)。"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def _data(seed=0, n=8, d_k=6, d_v=4):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d_k)), rng.standard_normal((n, d_k)),
            rng.standard_normal((n, d_v)))


def test_rows_sum_to_one():
    """T1: softmax 行和 = 1,精度 1e-12。"""
    Q, K, V = _data()
    for causal in (False, True):
        _, A = core_eq.attention(Q, K, V, causal=causal)
        assert np.allclose(A.sum(axis=-1), 1.0, rtol=0, atol=1e-12)
        assert np.all(A >= 0.0)


def test_causal_mask_zero():
    """T2: 掩码未来位置权重精确为 0;∂out_i/∂V_j = 0 对 j>i
    (扰动 V_j 不影响任何 i<j 的输出行,精确成立)。"""
    Q, K, V = _data()
    out, A = core_eq.attention(Q, K, V, causal=True)
    assert np.all(A[np.triu_indices(8, k=1)] == 0.0)  # exp(-inf)=0,精确
    V2 = V.copy()
    V2[3] += 1.7  # 扰动一个「未来」值
    out2, _ = core_eq.attention(Q, K, V2, causal=True)
    assert np.all(out2[:3] == out[:3])  # 前 3 行完全不变
    assert not np.allclose(out2[3:], out[3:])  # 自身及之后会变(性质的方向性)


def test_scaling_variance():
    """T3: q·k ~ N(0, d_k) → 原始 logit std ≈ √d_k,缩放后 ≈ 1。
    容差按解析 3σ 导出:N = n² 个样本,std 估计的相对波动 ≈ 1/√(2N),放宽到 0.5。"""
    rng = np.random.default_rng(2)
    n, d_k = 64, 64
    Q = rng.standard_normal((n, d_k))
    K = rng.standard_normal((n, d_k))
    raw = (Q @ K.T).ravel()
    scaled = raw / np.sqrt(d_k)  # Eq.(1) 的缩放
    assert abs(raw.std() - np.sqrt(d_k)) < 0.5, f"raw std={raw.std():.3f}, 期望≈8"
    assert abs(scaled.std() - 1.0) < 0.5 / np.sqrt(d_k) * np.sqrt(d_k), (
        f"scaled std={scaled.std():.3f}, 期望≈1")


def test_query_permutation_equivariance():
    """T4: 无掩码时置换查询行,输出行做同样置换。"""
    Q, K, V = _data(seed=3)
    rng = np.random.default_rng(4)
    perm = rng.permutation(8)
    out_perm, _ = core_eq.attention(Q[perm], K, V, causal=False)
    out_ref, _ = core_eq.attention(Q, K, V, causal=False)
    assert np.allclose(out_perm, out_ref[perm], rtol=0, atol=1e-12)
