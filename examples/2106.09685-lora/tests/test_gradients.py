"""LoRA gradient checks (L5): outer-product analytic gradients vs central
differences, through both the ΔW-formed and the streaming forward paths."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq
import core_pseudo

D, K, R = 5, 4, 2


def _fixture(seed=0, zero_B=False):
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((D, K))
    B, A = core_eq.init_lora(D, K, R, rng)
    if not zero_B:
        B = rng.standard_normal((D, R))
    x = rng.standard_normal(K)
    w = rng.standard_normal(D)
    return W0, B, A, x, w


def test_grads_wrt_B_and_A():
    """L5 at a generic (nonzero) B: both analytic outer products must match
    central differences at rtol 1e-6."""
    W0, B, A, x, w = _fixture(seed=7)
    gB, gA = core_eq.lora_gradients(B, A, x, w, alpha=2.0)

    err_B = assert_grad_close(
        lambda b: float(w @ core_eq.lora_forward_eq(W0, b.reshape(D, R), A, x, alpha=2.0)[0]),
        B.ravel(), gB, rtol=1e-6, name="dL/dB")
    err_A = assert_grad_close(
        lambda a: float(w @ core_eq.lora_forward_eq(W0, B, a.reshape(R, K), x, alpha=2.0)[0]),
        A.ravel(), gA, rtol=1e-6, name="dL/dA")
    print(f"\n[lora] dL/dB err={err_B:.2e}, dL/dA err={err_A:.2e}")


def test_grads_through_streaming_path():
    """Same check through the streaming forward (no ΔW formed): the two paths are
    the same differentiable function, so gradients must agree too.

    rtol loosened to 5e-4 for an honest reason: the smallest gradient component
    here is O(1e-7), where the central-difference noise floor |f|·eps/(2h·|g|) is
    ≈1e-4 (see tests/test_gradcheck.py's floor analysis). The 1e-12 exactness of
    the two paths is established separately in test_crosscheck.py."""
    W0, B, A, x, w = _fixture(seed=8)
    gB, gA = core_eq.lora_gradients(B, A, x, w, alpha=1.0)

    err_B = assert_grad_close(
        lambda b: float(w @ core_pseudo.lora_forward_stream(W0, b.reshape(D, R), A, x)),
        B.ravel(), gB, rtol=5e-4, name="dL/dB (stream)")
    err_A = assert_grad_close(
        lambda a: float(w @ core_pseudo.lora_forward_stream(W0, B, a.reshape(R, K), x)),
        A.ravel(), gA, rtol=5e-4, name="dL/dA (stream)")
    print(f"\n[lora] stream dL/dB err={err_B:.2e}, dL/dA err={err_A:.2e}")


def test_grad_A_exact_zero_at_init():
    """L2's gradient statement: at the paper's init (B=0), central differences on A
    must also be exactly zero — matching the analytic zeros (the loss is constant
    in A)."""
    W0, B, A, x, w = _fixture(seed=9, zero_B=True)
    gB, gA = core_eq.lora_gradients(B, A, x, w)
    assert np.all(gA == 0.0)
    err = assert_grad_close(
        lambda a: float(w @ core_eq.lora_forward_eq(W0, B, a.reshape(R, K), x)[0]),
        A.ravel(), gA, rtol=1e-6, name="dL/dA (B=0)")
    print(f"\n[lora] dL/dA at init: max rel err = {err:.2e}")
