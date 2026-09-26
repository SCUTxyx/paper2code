"""FlashAttention property tests (F2 stability / F4 partition invariance / F5 causal)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def _data(seed=0, n_q=8, n_k=8, d=4, d_v=3):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_q, d)), rng.standard_normal((n_k, d)),
            rng.standard_normal((n_k, d_v)))


def test_softmax_shift_invariance():
    """F2 basis: softmax(S) == softmax(S + c·1) — the identity the running max
    applies per tile. Exact to 1e-12."""
    rng = np.random.default_rng(0)
    S = rng.standard_normal((6, 5))
    for c in (0.5, 100.0, -333.0):
        P1 = np.exp(S - S.max(-1, keepdims=True))
        P1 /= P1.sum(-1, keepdims=True)
        P2 = np.exp((S + c) - (S + c).max(-1, keepdims=True))
        P2 /= P2.sum(-1, keepdims=True)
        assert np.allclose(P1, P2, rtol=0, atol=1e-12), f"c={c}"


def test_never_overflows():
    """F2: with logits ~±10⁴, naive exp overflows (inf/inf → NaN) while the
    tiled path stays finite and equals the stable direct result."""
    Q, K, V = _data(seed=1)
    Q_big, K_big = 40.0 * Q, 40.0 * K     # |logit| ~ 40·40·√4 ≈ 3×10³ ≫ 709
    with np.errstate(over="ignore", invalid="ignore"):
        naive = core_pseudo.attention_naive_unstable(Q_big, K_big, V)
    assert np.isnan(naive).any(), "naive softmax should overflow here (documented)"
    tiled = core_eq.flash_attention_eq(Q_big, K_big, V, block_q=2, block_k=2)
    direct = core_eq.attention_direct(Q_big, K_big, V)
    assert np.all(np.isfinite(tiled))
    assert np.allclose(tiled, direct, rtol=0, atol=1e-12)


def test_block_partition_invariance():
    """F4: output identical for any block grid (bq, bk), including b=1 —
    tiling is an algebraic identity, not an approximation."""
    Q, K, V = _data(seed=2, n_q=9, n_k=7)
    ref = core_eq.flash_attention_eq(Q, K, V, block_q=9, block_k=7)
    for bq in (1, 2, 4, 9):
        for bk in (1, 3, 7):
            out = core_pseudo.flash_attention_stream(Q, K, V, block_q=bq, block_k=bk)
            assert np.allclose(out, ref, rtol=0, atol=1e-12), f"(bq,bk)=({bq},{bk})"


def test_causal_masked_blocks_zero():
    """F5: with causal masking, perturbing a strictly-future V row leaves all
    earlier output rows bit-identical (masked tiles contribute exactly 0)."""
    Q, K, V = _data(seed=3, n_q=8, n_k=8)
    out = core_eq.flash_attention_eq(Q, K, V, block_q=3, block_k=3, causal=True)
    V2 = V.copy()
    V2[5] += 1.7                                   # a strictly-future value
    out2 = core_eq.flash_attention_eq(Q, K, V2, block_q=3, block_k=3, causal=True)
    assert np.all(out2[:5] == out[:5])
