"""InfoNCE/CLIP property tests (N1 symmetry / N2 temperature limits / N3 scale /
N4 permutation)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def _data(seed=0, n=6, d=8):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)), rng.standard_normal((n, d))


def test_tower_swap_symmetry():
    """N1: the bidirectional average is invariant under swapping towers,
    L(I,T) = L(T,I), at 1e-12."""
    I, T = _data()
    assert abs(core_eq.clip_loss(I, T, tau=0.1) - core_eq.clip_loss(T, I, tau=0.1)) < 1e-12


def test_temperature_limits():
    """N2: τ→∞ sends the loss to log N (softmax → uniform); with strictly dominant
    positives, τ→0 sends it to 0."""
    I, T = _data(seed=1)
    n = I.shape[0]
    # τ→∞: logits ~ 1/τ → 1e-8; CE deviates from log N by ~1e-8
    assert abs(core_eq.clip_loss(I, T, tau=1e8) - np.log(n)) < 1e-7
    # τ→0: build strictly dominant positives (I=T: self-similarity 1 strictly
    # exceeds all cross-similarities < 1)
    assert core_eq.clip_loss(I, I, tau=1e-5) < 1e-6
    # monotone direction: smaller τ sharpens correct pairings → lower loss
    assert core_eq.clip_loss(I, I, tau=1e-3) < core_eq.clip_loss(I, I, tau=1.0)


def test_scale_invariance():
    """N3: internal L2 normalization → the loss is invariant to input scale."""
    I, T = _data(seed=2)
    a = core_eq.clip_loss(I, T, tau=0.2)
    b = core_eq.clip_loss(5.3 * I, 0.2 * T, tau=0.2)
    assert abs(a - b) < 1e-12


def test_batch_permutation_invariance():
    """N4: jointly permuting batch rows leaves the loss unchanged."""
    I, T = _data(seed=3)
    rng = np.random.default_rng(4)
    perm = rng.permutation(I.shape[0])
    a = core_eq.clip_loss(I, T, tau=0.07)
    b = core_eq.clip_loss(I[perm], T[perm], tau=0.07)
    assert abs(a - b) < 1e-12
