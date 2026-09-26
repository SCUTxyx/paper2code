"""FlashAttention cross-checks: tiled (both forms) vs direct attention, and the
two tiled formulations against each other, over a grid of shapes and partitions."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def test_tiled_equals_direct():
    """F1 (the paper's core claim): tiled attention is EXACT attention.
    1e-12 across shapes and both causal settings."""
    for seed, (n_q, n_k, d, d_v) in enumerate([(8, 8, 4, 3), (1, 7, 3, 2), (16, 4, 2, 5)]):
        r = np.random.default_rng(seed)
        Q = r.standard_normal((n_q, d))
        K = r.standard_normal((n_k, d))
        V = r.standard_normal((n_k, d_v))
        for causal in (False, True):
            direct = core_eq.attention_direct(Q, K, V, causal=causal)  # reference per setting
            a = core_eq.flash_attention_eq(Q, K, V, block_q=2, block_k=3, causal=causal)
            b = core_pseudo.flash_attention_stream(Q, K, V, block_q=3, block_k=2, causal=causal)
            assert np.allclose(a, direct, rtol=0, atol=1e-12), (seed, causal, "eq")
            assert np.allclose(b, direct, rtol=0, atol=1e-12), (seed, causal, "stream")


def test_two_tiled_forms_match_each_other():
    """Normalized-every-step (diag-matrix Algorithm 1 form) vs streaming
    (unnormalized accumulator): same recurrence, different path, 1e-12 equal."""
    rng = np.random.default_rng(1)
    for _ in range(3):
        Q = rng.standard_normal((11, 5))
        K = rng.standard_normal((13, 5))
        V = rng.standard_normal((13, 4))
        bq, bk = rng.integers(1, 12), rng.integers(1, 14)
        causal = bool(rng.integers(0, 2))
        a = core_eq.flash_attention_eq(Q, K, V, block_q=int(bq), block_k=int(bk), causal=causal)
        b = core_pseudo.flash_attention_stream(Q, K, V, block_q=int(bq), block_k=int(bk), causal=causal)
        assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_rectangular_cross_attention():
    """n_q ≠ n_k (cross-attention) tiles correctly — a common real-world shape."""
    rng = np.random.default_rng(2)
    Q = rng.standard_normal((4, 6))
    K = rng.standard_normal((9, 6))
    V = rng.standard_normal((9, 2))
    assert np.allclose(core_eq.flash_attention_eq(Q, K, V, block_q=3, block_k=4),
                       core_eq.attention_direct(Q, K, V), rtol=0, atol=1e-12)
