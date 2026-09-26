"""Attention dual-implementation cross-check: vectorized -inf masking vs
per-query subset normalization."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def test_same_output_with_and_without_mask():
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((9, 5))
    K = rng.standard_normal((9, 5))
    V = rng.standard_normal((9, 3))
    for causal in (False, True):
        out_a, A_a = core_eq.attention(Q, K, V, causal=causal)
        out_b, A_b = core_pseudo.attention_loop(Q, K, V, causal=causal)
        assert np.allclose(out_a, out_b, rtol=0, atol=1e-12)
        assert np.allclose(A_a, A_b, rtol=0, atol=1e-12)


def test_rectangular_kv():
    """Cross-attention (different query/key lengths) agrees too."""
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((4, 6))
    K = rng.standard_normal((7, 6))
    V = rng.standard_normal((7, 2))
    out_a, _ = core_eq.attention(Q, K, V, causal=False)
    out_b, _ = core_pseudo.attention_loop(Q, K, V, causal=False)
    assert np.allclose(out_a, out_b, rtol=0, atol=1e-12)
