"""Attention anchor tests: d=1 two-key hand case via an exact softmax identity.

Hand computation: d=1 so the scale factor √1=1. q=1, k₁=1, k₂=-1 → logits [1, -1].
softmax([1,-1]) = [e²/(1+e²), 1/(1+e²)] = [σ(2), σ(-2)], σ(2)=0.8807970779778823.
V=[2,-1] (d_v=1) → out = 2σ(2) - σ(-2) = 3σ(2) - 1 = 1.642391233933647.
Causal single position: weights [1,0] (exact), out = v₁ = 2.
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq

SIGMA2 = 0.8807970779778823  # σ(2), hand constant


def test_two_keys_sigmoid_identity():
    Q = np.array([[1.0]])
    K = np.array([[1.0], [-1.0]])
    V = np.array([[2.0], [-1.0]])
    out, A = core_eq.attention(Q, K, V, causal=False)
    assert A.shape == (1, 2)
    assert np.allclose(A[0], [SIGMA2, 1 - SIGMA2], rtol=1e-12)
    assert np.allclose(out[0], 3 * SIGMA2 - 1, rtol=1e-12)


def test_causal_single_position():
    Q = np.array([[1.0], [1.0]])
    K = np.array([[1.0], [-1.0]])
    V = np.array([[2.0], [-1.0]])
    out, A = core_eq.attention(Q, K, V, causal=True)
    # Position 0 sees only key 0: weights [1,0]; position 1 sees both: [σ(2), σ(-2)]
    assert np.allclose(A[0], [1.0, 0.0], rtol=0, atol=0)
    assert np.allclose(out[0], [2.0], rtol=0, atol=1e-15)
    assert np.allclose(A[1], [SIGMA2, 1 - SIGMA2], rtol=1e-12)
