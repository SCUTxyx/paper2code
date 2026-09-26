"""LoRA cross-checks: ΔW-formed forward vs streaming forward; merge/unmerge
equivalence; the merged weight produces identical outputs."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def test_two_paths_match():
    """L4 (forward form): W0x + (α/r)(BA)x == W0x + (α/r)B(Ax) at 1e-12 — the
    matrix product's associativity, over several shapes and scalings."""
    rng = np.random.default_rng(0)
    for seed, (d, k, r) in enumerate([(6, 5, 2), (1, 7, 1), (8, 3, 3)]):
        r_ = core_eq.init_lora(d, k, r, np.random.default_rng(seed))
        B, A = r_[0], r_[1]
        B = np.abs(B) + 0.5                       # keep full rank
        W0 = np.random.default_rng(seed + 10).standard_normal((d, k))
        x = np.random.default_rng(seed + 20).standard_normal(k)
        for alpha in (1.0, 3.0):
            a, _ = core_eq.lora_forward_eq(W0, B, A, x, alpha=alpha)
            b = core_pseudo.lora_forward_stream(W0, B, A, x, alpha=alpha)
            assert np.allclose(a, b, rtol=0, atol=1e-12), (d, k, r, alpha)


def test_merge_equivalence():
    """L4 (merged weight): applying W0 + (α/r)BA must give the same output as the
    adapter path, and unmerge must recover the base exactly."""
    rng = np.random.default_rng(1)
    d, k, r = 7, 6, 2
    B = rng.standard_normal((d, r))
    A = rng.standard_normal((r, k))
    W0 = rng.standard_normal((d, k))
    alpha = 2.0
    W = core_pseudo.merge(W0, B, A, alpha=alpha)
    for _ in range(5):
        x = rng.standard_normal(k)
        direct = W @ x
        adapted = core_eq.lora_forward_eq(W0, B, A, x, alpha=alpha)[0]
        assert np.allclose(direct, adapted, rtol=0, atol=1e-12)
    assert np.allclose(core_pseudo.unmerge(W, B, A, alpha=alpha), W0, rtol=0, atol=1e-12)
