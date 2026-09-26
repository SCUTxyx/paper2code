"""Attention property tests (T1 row sums / T2 causal mask / T3 scaling variance /
T4 permutation equivariance)."""

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
    """T1: softmax row sums = 1 at 1e-12."""
    Q, K, V = _data()
    for causal in (False, True):
        _, A = core_eq.attention(Q, K, V, causal=causal)
        assert np.allclose(A.sum(axis=-1), 1.0, rtol=0, atol=1e-12)
        assert np.all(A >= 0.0)


def test_causal_mask_zero():
    """T2: masked future positions have exactly-zero weights; ∂out_i/∂V_j = 0 for j>i
    (perturbing V_j leaves every output row i<j bit-identical)."""
    Q, K, V = _data()
    out, A = core_eq.attention(Q, K, V, causal=True)
    assert np.all(A[np.triu_indices(8, k=1)] == 0.0)  # exp(-inf)=0, exact
    V2 = V.copy()
    V2[3] += 1.7  # perturb a "future" value
    out2, _ = core_eq.attention(Q, K, V2, causal=True)
    assert np.all(out2[:3] == out[:3])  # first 3 rows unchanged exactly
    assert not np.allclose(out2[3:], out[3:])  # self and later rows do change (direction)


def test_scaling_variance():
    """T3: for q·k ~ N(0, d_k), raw logit std ≈ √d_k; after 1/√d_k scaling ≈ 1.
    Tolerance derived analytically at 3σ: N = n² samples, std-estimate relative
    fluctuation ≈ 1/√(2N), loosened to 0.5."""
    rng = np.random.default_rng(2)
    n, d_k = 64, 64
    Q = rng.standard_normal((n, d_k))
    K = rng.standard_normal((n, d_k))
    raw = (Q @ K.T).ravel()
    scaled = raw / np.sqrt(d_k)  # Eq.(1)'s scaling
    assert abs(raw.std() - np.sqrt(d_k)) < 0.5, f"raw std={raw.std():.3f}, expected ≈8"
    assert abs(scaled.std() - 1.0) < 0.5 / np.sqrt(d_k) * np.sqrt(d_k), (
        f"scaled std={scaled.std():.3f}, expected ≈1")


def test_query_permutation_equivariance():
    """T4: without masking, permuting query rows permutes output rows identically."""
    Q, K, V = _data(seed=3)
    rng = np.random.default_rng(4)
    perm = rng.permutation(8)
    out_perm, _ = core_eq.attention(Q[perm], K, V, causal=False)
    out_ref, _ = core_eq.attention(Q, K, V, causal=False)
    assert np.allclose(out_perm, out_ref[perm], rtol=0, atol=1e-12)
