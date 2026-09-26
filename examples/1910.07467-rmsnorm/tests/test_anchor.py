"""RMSNorm anchor tests: exact hand constants + the ε-placement discriminator.

Hand computation (d=2, g=1, ε=0): x = [3, 4] → m = (9+16)/2 = 25/2,
r = √(25/2) = 5/√2 → y = [3√2/5, 4√2/5] = [0.8485281374238570, 1.1313708498984760].
With g = [2, 0.5]: y = [6√2/5, 2√2/5] = [1.697056274847714, 0.565685424949238].

ε-placement discriminator (d=1, x=[2], ε=1): the three plausible readings of
"x / sqrt(mean(x²) + ε)":
  correct  x/√(x²+ε) = 2/√5 = 0.8944271909999159
  wrong    x/(√x²+ε) = 2/3   = 0.6666666666666666
  wrong    x/√x² + ε  = 1+1  = 2
One assertion at rtol 1e-12 kills both wrong placements (the classic shared-misreading
case: a dual implementation that reads ε the same wrong way would pass cross-checks).
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def test_d2_hand_case():
    x = np.array([3.0, 4.0])
    out = core_eq.rmsnorm(x, np.ones(2), eps=0.0)
    assert np.allclose(out, [3 * np.sqrt(2) / 5, 4 * np.sqrt(2) / 5], rtol=1e-15)
    assert np.allclose(out, [0.8485281374238570, 1.1313708498984760], rtol=1e-12)
    out_g = core_eq.rmsnorm(x, np.array([2.0, 0.5]), eps=0.0)
    assert np.allclose(out_g, [1.697056274847714, 0.565685424949238], rtol=1e-12)


def test_epsilon_placement():
    out = core_eq.rmsnorm(np.array([2.0]), np.ones(1), eps=1.0)
    assert np.allclose(out, 2 / np.sqrt(5), rtol=1e-12)
    assert abs(out[0] - 0.8944271909999159) < 1e-12
    # the two wrong readings, for the record:
    assert 0.6666666666666666 != out[0] != 2.0
