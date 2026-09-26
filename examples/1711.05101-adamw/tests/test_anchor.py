"""AdamW anchor tests: hand-computed closed forms.

Hand computation (θ0=1, g1=1, g2=2, β1=0.9, β2=0.999, lr=0.1, λ=0.1, ε=1e-8):
  t=1: adaptive part = 0.1·1/(1+1e-8) ≈ 0.1 (as in the Adam exam); decay = 0.1·0.1·1
       θ₁ = 1 − 0.1 − 0.01 = 0.89 (exact up to the ε correction)
  t=2: adaptive = 0.1·1.5263158/√2.5007504 = 0.0965182 (same as Adam exam's m̂₂/v̂₂);
       decay = 0.1·0.1·0.89 = 0.0089
       θ₂ = 0.89 − 0.0965182 − 0.0089 = 0.7845818
Zero-gradient closed form: θ_t = θ_0·(1−lr·λ)^t. For lr=0.05, λ=0.2: factor 0.99,
θ₅ = 0.99⁵ = 0.9509900499 (hand-computable geometric series).
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def test_two_steps_hand_computed():
    g = np.array([[1.0], [2.0]])
    out = core_eq.adamw_run(g, theta0=np.ones(1), lr=0.1, weight_decay=0.1)
    assert np.allclose(out["thetas"][1], 0.89, rtol=1e-6)
    assert np.allclose(out["thetas"][2], 0.7845818, rtol=1e-5)


def test_zero_gradient_shrink():
    """W3: g=0 → θ_t = θ_0·(1−lr·λ)^t exactly — a pure geometric shrink,
    independent of β1/β2 and of any gradient history."""
    for theta0 in (np.array([2.0, -3.0]), np.array([1.0])):
        out = core_eq.adamw_run(np.zeros((5, theta0.size)), theta0,
                                lr=0.05, weight_decay=0.2)
        for t in range(6):
            expect = theta0 * (0.99 ** t)
            assert np.allclose(out["thetas"][t], expect, rtol=0, atol=1e-15), f"t={t}"
    # the hand constant: 0.99⁵
    assert abs(0.99 ** 5 - 0.9509900499) < 1e-12
