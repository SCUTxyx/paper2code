"""LoRA anchor tests: integer hand case (d=2, r=1, k=1).

Hand computation: A = [[2]] (1×1), B = [[3], [5]] (2×1), W0 = [[0.5], [-1.0]] (2×1),
x = [1.0], α/r = 1:
  ΔW = B@A = [[6], [10]] (2×1)
  h = W0 x + ΔW x = [0.5, -1.0] + [6, 10] = [6.5, 9.0]
  with α/r = 2: h = [0.5, -1.0] + 2·[6, 10] = [12.5, 19.0]
  merged W = W0 + ΔW = [[6.5], [9.0]] (2×1, same shape as W0 — no broadcasting).
All hand integers / exact halves.
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

W0 = np.array([[0.5], [-1.0]])
B = np.array([[3.0], [5.0]])
A = np.array([[2.0]])
X = np.array([1.0])


def test_rank_one_hand_case():
    h, delta_W = core_eq.lora_forward_eq(W0, B, A, X, alpha=1.0)
    assert np.all(delta_W == [[6.0], [10.0]])
    assert np.all(h == [6.5, 9.0])           # exact hand values

    h2, _ = core_eq.lora_forward_eq(W0, B, A, X, alpha=2.0)
    assert np.all(h2 == [12.5, 19.0])        # α/r = 2


def test_streaming_path_same_hand_case():
    h = core_pseudo.lora_forward_stream(W0, B, A, X, alpha=1.0)
    assert np.all(h == [6.5, 9.0])


def test_merge_hand_case():
    W = core_pseudo.merge(W0, B, A, alpha=1.0)
    assert np.all(W == [[6.5], [9.0]])       # W0 + ΔW, shape (2,1) preserved
    # unmerge recovers the frozen base exactly
    W0_back = core_pseudo.unmerge(W, B, A, alpha=1.0)
    assert np.all(W0_back == W0)


def test_scale_discrimination_with_r_greater_than_one():
    """Mutation-audit anchor: the paper's scaling is (α/r) — with r=1 the readings
    α/r and α·r coincide, so a real discriminator needs r>1.
    B=[[1,2],[3,4]], A=[[5,6],[7,8]], α=1, r=2 → scale 0.5:
      BA = [[19,22],[43,50]] → ΔW = [[9.5,11],[21.5,25]]  (α·r would give ×2)"""
    B = np.array([[1.0, 2.0], [3.0, 4.0]])
    A = np.array([[5.0, 6.0], [7.0, 8.0]])
    W0 = np.zeros((2, 2))
    _, delta_W = core_eq.lora_forward_eq(W0, B, A, np.ones(2), alpha=1.0)
    assert np.all(delta_W == [[9.5, 11.0], [21.5, 25.0]])
    # α=2 with r=2 → scale 1: ΔW must be exactly BA
    _, delta_W2 = core_eq.lora_forward_eq(W0, B, A, np.ones(2), alpha=2.0)
    assert np.all(delta_W2 == [[19.0, 22.0], [43.0, 50.0]])
