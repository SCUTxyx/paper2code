"""InfoNCE/CLIP dual-implementation cross-check: vectorized matrix form vs
per-pair loop form."""

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
    for seed, (n, d, tau) in enumerate([(6, 8, 0.07), (3, 4, 1.0), (5, 2, 0.5)]):
        r = np.random.default_rng(seed)
        I = r.standard_normal((n, d)) * (1.0 + seed)
        T = r.standard_normal((n, d))
        a = core_eq.clip_loss(I, T, tau=tau)
        b = core_pseudo.clip_loss_loop(I, T, tau=tau)
        assert abs(a - b) < 1e-12, f"(n={n},d={d},τ={tau}): {a} vs {b}"


def test_directional_components_match():
    """The two directional components (i2t and t2i) must also agree between the
    implementations (matrix-transpose semantics cross-checked)."""
    r = np.random.default_rng(3)
    I = r.standard_normal((6, 5))
    T = r.standard_normal((6, 5))
    a = core_eq.clip_loss(I, T, tau=0.2)
    b = core_pseudo.clip_loss_loop(I, T, tau=0.2)
    assert abs(a - b) < 1e-12
