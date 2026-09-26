"""Adam anchor tests: the first updates hand-computed (derivation in comments,
constants as literals).

Hand computation (θ0=0, g1=1, g2=2, β1=0.9, β2=0.999, lr=0.1, ε=1e-8):
  t=1: m₁=0.1, v₁=0.001; m̂₁=m₁/(1-0.9)=1, v̂₁=v₁/(1-0.999)=1
       θ₁ = -0.1·1/(1+1e-8) = -0.099999999
  t=2: m₂=0.9·0.1+0.1·2=0.29; v₂=0.999·0.001+0.001·4=0.004999
       m̂₂=0.29/0.19=1.5263158; v̂₂=0.004999/0.001999=2.5007504
       θ₂ = -0.1 - 0.1·1.5263158/√2.5007504 = -0.1 - 0.0965182 = -0.1965182
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def test_first_two_steps_hand_computed():
    g = np.array([[1.0], [2.0]])
    out = core_eq.adam_run(g, theta0=np.zeros(1), lr=0.1,
                           beta1=0.9, beta2=0.999, eps=1e-8)
    # Mechanical anchor for the bias correction: first step has m̂₁ = v̂₁ = 1 exactly
    assert np.allclose(out["m_hats"][0], 1.0, rtol=0, atol=1e-15)
    assert np.allclose(out["v_hats"][0], 1.0, rtol=0, atol=1e-15)
    assert np.allclose(out["thetas"][1], -0.099999999, rtol=1e-9)
    assert np.allclose(out["thetas"][2], -0.1965182, rtol=1e-5)


def test_five_steps_constant_gradient_hand_computed():
    """Constant gradient c=[0.7,-0.4] for five steps: each update is α·c/(|c|+ε) ≈
    α·sign(c) — indeed 0.1·0.7/0.7 = 0.1, which is exactly A1's "step ≈ lr,
    scale-independent". θ₅ = -5·0.1·(1 - ε/|c|) = -0.4999999929 / +0.4999999875."""
    c = np.array([0.7, -0.4])
    out = core_eq.adam_run(np.tile(c, (5, 1)), theta0=np.zeros(2), lr=0.1)
    assert np.allclose(out["thetas"][5], [-0.49999999, 0.49999999], rtol=1e-7)


def test_epsilon_placement_is_discriminating():
    """Mutation-audit anchor: with a LARGE ε the three plausible readings of
    Algorithm 1's '√v̂ + ε' differ hugely, so the placement is verified exactly:
      correct  m̂/(√v̂+ε): update = 0.1·1/(1+1) = 0.05      → θ₁ = -0.05
      wrong    m̂/√(v̂+ε): update = 0.1/√2 ≈ 0.0707         → θ₁ ≈ -0.0707
      wrong    m̂/√v̂+ε: update = 0.1/1 + 1 = 1.1            → θ₁ = -1.1
    A single assertion at rtol 1e-9 kills all wrong placements (with the default
    ε=1e-8 the readings differ by only ~1e-8 and slip under ordinary tolerances —
    this is the 'self-consistent misreading' the verification ladder warns about)."""
    out = core_eq.adam_run(np.array([[1.0]]), theta0=np.zeros(1), lr=0.1, eps=1.0)
    assert np.allclose(out["thetas"][1], -0.05, rtol=1e-9)
