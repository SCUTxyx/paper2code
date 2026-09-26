"""LoRA property tests (L1 init semantics / L2 gradient asymmetry / L3 rank /
L6 both-zero fixed point)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def _fixture(seed=0, d=6, k=5, r=2):
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((d, k))
    B, A = core_eq.init_lora(d, k, r, rng)
    x = rng.standard_normal(k)
    w = rng.standard_normal(d)
    return W0, B, A, x, w


def test_init_output_equals_base():
    """L1: with B = 0, the adapter contributes exactly nothing — h == W0 x
    bit-for-bit for any A."""
    W0, B, A, x, _ = _fixture()
    h, delta_W = core_eq.lora_forward_eq(W0, B, A, x)
    assert np.all(delta_W == 0.0)
    assert np.all(h == W0 @ x)          # exact equality, not rtol


def test_gradient_asymmetry_at_init():
    """L2: at the paper's initialization, ∂L/∂A is EXACTLY zero in every component
    (the loss is independent of A when B=0) while ∂L/∂B is nonzero — training
    starts by moving B only. Note: swapping the zeros (A=0, B random) also trains;
    it is zeroing BOTH factors that freezes training (L6)."""
    W0, B, A, x, w = _fixture()
    grad_B, grad_A = core_eq.lora_gradients(B, A, x, w)
    assert np.all(grad_A == 0.0)         # exact zeros, structural not approximate
    assert np.all(grad_B != 0.0)
    # swapped init also starts training (A's gradient becomes the nonzero one)
    Z = np.zeros_like(A)
    gB2, gA2 = core_eq.lora_gradients(B + 1.0, Z, x, w)  # A=0, B nonzero
    assert np.all(gB2 == 0.0)
    assert np.all(gA2 != 0.0)


def test_rank_constraint():
    """L3: rank(BA) ≤ r, and = r for full-rank factors — the singular-value
    spectrum of ΔW has exactly r entries above the float noise floor."""
    d, k, r = 8, 6, 2
    rng = np.random.default_rng(1)
    B = rng.standard_normal((d, r))
    A = rng.standard_normal((r, k))
    _, delta_W = core_eq.lora_forward_eq(np.zeros((d, k)), B, A, np.ones(k), alpha=1.0)
    sv = np.linalg.svd(delta_W, compute_uv=False)
    tol = sv[0] * 1e-12
    assert np.sum(sv > tol) == r, f"spectrum above tol: {sv}"
    assert np.all(sv[r:] < tol), "rank must be truncated exactly at r"


def test_both_zero_init_is_fixed_point():
    """L6: zeroing BOTH factors freezes training forever — both gradients are
    exactly zero and plain SGD never moves ΔW. The classic silent LoRA bug,
    demonstrated rather than asserted."""
    d, k, r = 4, 4, 2
    rng = np.random.default_rng(2)
    W0 = rng.standard_normal((d, k))
    A = np.zeros((r, k))
    B = np.zeros((d, r))
    x = rng.standard_normal(k)
    w = rng.standard_normal(d)
    lr = 0.1
    for _ in range(20):
        gB, gA = core_eq.lora_gradients(B, A, x, w)
        assert np.all(gB == 0.0) and np.all(gA == 0.0)
        B, A = B - lr * gB, A - lr * gA       # plain SGD step
    _, delta_W = core_eq.lora_forward_eq(W0, B, A, x)
    assert np.all(delta_W == 0.0)             # the adapter never learned anything
