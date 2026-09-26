"""RoPE anchor tests: d=2 hand cases with standard trigonometric constants.

Hand computation: with d=2, Eq.(15) has a single frequency θ_1 = 10000^0 = 1
(one radian per step).
1) f([1,0], 3) = [cos3, sin3] = [-0.9899924966004454, 0.1411200080598672];
2) score: ⟨f(q,3), f(k,1)⟩ = qᵀR^{n-m}k = qᵀR^{-2}k; q=[1,0], k=[0,1]:
   R^{-2}k = [sin2, cos2] → score = sin(2) = 0.9092974268256817.
   (complex-form check: Re[q·k̄·e^{i(m-n)θ}] = Re[-i·e^{2i}] = sin2 ✓)
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def test_single_frequency_theta():
    """With d=2 the only θ equals 1 (the i=1 term of Eq.15)."""
    assert np.allclose(core_eq.theta_seq(2), [1.0], rtol=0, atol=0)


def test_rotate_three_steps():
    q = np.array([1.0, 0.0])
    out = core_eq.rope_rotate(q, 3)
    assert np.allclose(out, [np.cos(3.0), np.sin(3.0)], rtol=1e-12)
    assert np.allclose(out, [-0.9899924966004454, 0.1411200080598672], rtol=1e-9)


def test_score_equals_sin2():
    s = core_eq.rope_score(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 3, 1)
    assert abs(s - np.sin(2.0)) < 1e-12
    assert abs(s - 0.9092974268256817) < 1e-9


def test_multi_frequency_anchor():
    """d=4, two frequencies θ = [1, 10000^{-1/2}] (Eq.15 i=1,2), m=5 — the
    rotation angle is 5θ, hand-checkable per block."""
    thetas = core_eq.theta_seq(4)
    assert np.allclose(thetas, [1.0, 10000.0 ** (-0.5)], rtol=1e-15)
    x = np.array([1.0, 0.0, 0.0, 1.0])
    out = core_eq.rope_rotate(x, 5)
    t1, t2 = thetas
    expect = np.array([np.cos(5 * t1), np.sin(5 * t1),
                       -np.sin(5 * t2), np.cos(5 * t2)])
    assert np.allclose(out, expect, rtol=1e-12)
