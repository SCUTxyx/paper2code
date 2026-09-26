"""FlashAttention anchor tests: single-query two-tile hand computations.

Hand derivation (d=1 ⇒ scale factor √1=1; one query, two keys tiled with b_k=1):
  logits s = [0, 2], V = [1, 3].
  Tile 1: m=0, l=1, U = e^{0−0}·v₁ = 1.
  Tile 2: m_new = max(0, 2) = 2, rescale = e^{0−2} = e^{−2};
          U = e^{−2}·1 + e^{2−2}·3 = e^{−2} + 3;  l = e^{−2}·1 + 1.
  out = (e^{−2} + 3)/(e^{−2} + 1) = (3 + e^{−2})·σ(2) = 3σ(2) + (1−σ(2)) = 1 + 2σ(2)
      = 2.7615941559557646.  (softmax([0,2])·[1,3] = σ(−2)·1 + σ(2)·3 — same ✓)
  This case FORCES the running max to increase — the rescale step is exercised.

Second case: s = [1, −1], V = [2, −1], max never increases:
  out = 3σ(2) − 1 = 1.642391233933647 (same constants as calibration/02_attention).
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

SIGMA2 = 0.8807970779778823  # σ(2)


def test_rescale_when_max_increases():
    """F3 through the growing-max path: literal 1 + 2σ(2), both implementations."""
    Q = np.array([[1.0]])
    K = np.array([[0.0], [2.0]])
    V = np.array([[1.0], [3.0]])
    for fn in (core_eq.flash_attention_eq, core_pseudo.flash_attention_stream):
        out = fn(Q, K, V, block_q=1, block_k=1)
        assert np.allclose(out, 1.0 + 2.0 * SIGMA2, rtol=1e-12), fn.__name__
    assert abs(1.0 + 2.0 * SIGMA2 - 2.7615941559557646) < 1e-15


def test_first_block_frozen_start():
    """F3 initial-state semantics: first tile must start from m=−∞, l=0, O=0 and
    the rescale factor must be exactly 0 (not NaN) — a classic off-by-init bug.
    Case with non-increasing max: out = 3σ(2) − 1 = 1.642391233933647."""
    Q = np.array([[1.0]])
    K = np.array([[1.0], [-1.0]])
    V = np.array([[2.0], [-1.0]])
    for fn in (core_eq.flash_attention_eq, core_pseudo.flash_attention_stream):
        out = fn(Q, K, V, block_q=1, block_k=1)
        assert np.allclose(out, 3.0 * SIGMA2 - 1.0, rtol=1e-12), fn.__name__


def test_single_tile_equals_direct():
    """Degenerate tiling (one block = no tiling) must equal the direct result."""
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((5, 4))
    K = rng.standard_normal((6, 4))
    V = rng.standard_normal((6, 2))
    a = core_eq.flash_attention_eq(Q, K, V, block_q=5, block_k=6)
    b = core_eq.attention_direct(Q, K, V)
    assert np.allclose(a, b, rtol=0, atol=1e-12)
