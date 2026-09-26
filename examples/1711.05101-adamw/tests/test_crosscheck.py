"""AdamW dual-implementation cross-check: fused Algorithm 2 update vs the decoupled
composition (Adam step + separate shrink); plus the λ=0 → Adam degeneracy."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def test_fused_equals_decoupled_composition():
    """W5: the fused update and the two-step composition agree exactly (1e-12) —
    the additivity that makes the decay 'decoupled'."""
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((40, 4))
    theta0 = rng.standard_normal(4)
    for wd in (0.0, 1e-3, 0.1, 0.5):
        a = core_eq.adamw_run(seq, theta0, weight_decay=wd)
        b = core_pseudo.adamw_run_decoupled(seq, theta0, weight_decay=wd)
        for key in ("thetas", "m_hats", "v_hats"):
            assert np.allclose(a[key], b[key], rtol=0, atol=1e-12), (wd, key)


def test_lambda_zero_is_adam():
    """W2: with λ=0, AdamW must equal a hand-rolled Adam (a few literal lines in
    this test, independent of both implementations)."""
    rng = np.random.default_rng(1)
    seq = rng.standard_normal((30, 3))
    theta0 = rng.standard_normal(3)
    lr, b1, b2, eps = 0.07, 0.85, 0.99, 1e-8

    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    expected = [theta.copy()]
    for t in range(1, 31):
        g = seq[t - 1]
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta = theta - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        expected.append(theta.copy())
    expected = np.array(expected)

    out = core_eq.adamw_run(seq, theta0, lr=lr, beta1=b1, beta2=b2, eps=eps,
                            weight_decay=0.0)
    assert np.allclose(out["thetas"], expected, rtol=0, atol=1e-12)
