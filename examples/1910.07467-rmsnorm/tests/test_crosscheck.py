"""RMSNorm dual-implementation cross-check: vectorized mean vs per-token loop."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def test_vectorized_equals_loop():
    rng = np.random.default_rng(0)
    for eps in (0.0, 1e-8, 1.0):
        x = rng.standard_normal((5, 7)) * (1.0 + eps)
        g = rng.standard_normal(7)
        a = core_eq.rmsnorm(x, g, eps=eps)
        b = core_pseudo.rmsnorm_loop(x, g, eps=eps)
        assert np.allclose(a, b, rtol=0, atol=1e-12), f"ε={eps}"


def test_three_dimensional_batch():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 4, 16))
    g = rng.standard_normal(16)
    assert np.allclose(core_eq.rmsnorm(x, g), core_pseudo.rmsnorm_loop(x, g),
                       rtol=0, atol=1e-12)
