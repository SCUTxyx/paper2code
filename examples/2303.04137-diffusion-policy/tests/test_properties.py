"""Diffusion Policy property tests (P-2 negative / P-3 window & semantics)."""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq

T, D_A = 60, 4
T_P, T_A, H = 16, 8, 50


def _fixture(seed=0):
    betas = core_eq.linear_beta_schedule(T)
    abars = core_eq.alpha_bar_seq(betas)
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((T_P, D_A))
    return betas, abars, x0, rng


def test_window_property():
    """P-3: receding-horizon execution uses only each chunk's first T_a rows
    (and the final chunk's remainder) — mutating a chunk BEYOND its executed
    window leaves the executed sequence bit-identical."""
    rng = np.random.default_rng(0)
    chunks = [rng.standard_normal((T_P, D_A)) for _ in range(7)]
    out = core_eq.execute_receding(chunks, T_A, H)
    chunks2 = [c.copy() for c in chunks]
    chunks2[0][T_A:] += 10.0    # rows beyond the executed window of chunk 0
    chunks2[3][T_A:] += 10.0    # a middle chunk's tail (never executed)
    out2 = core_eq.execute_receding(chunks2, T_A, H)
    assert np.all(out == out2)


def test_execution_semantics():
    """P-3: output has exactly `horizon` rows; rows come from the current replan's
    chunk head; the final chunk runs to min(T_p, remaining)."""
    rng = np.random.default_rng(1)
    chunks = [rng.standard_normal((T_P, D_A)) for _ in range(7)]
    out = core_eq.execute_receding(chunks, T_A, H)
    assert out.shape == (H, D_A)
    # first T_a rows equal chunk 0's head
    assert np.all(out[:T_A] == chunks[0][:T_A])
    # rows T_A..2T_A equal chunk 1's head, etc.
    for k in range(1, H // T_A):
        assert np.all(out[k * T_A:(k + 1) * T_A] == chunks[k][:T_A])
    # the tail beyond the last full replan comes from the final chunk's remainder
    tail_start = (H // T_A) * T_A
    last_i = H // T_A
    assert np.all(out[tail_start:] == chunks[last_i][:H - tail_start])


def test_ddim_biased_prediction_deviates():
    """P-2 negative: a biased predictor (ε̂ = ε + δ) does NOT land on x_0 —
    quantifies why predictor quality matters; deviation must be order-1 here."""
    betas, abars, x0, rng = _fixture(seed=2)
    eps = rng.standard_normal((T_P, D_A))
    x_T = core_eq.q_sample(x0, T, eps, abars)
    bias = 0.05 * np.ones_like(eps)
    eps_hat_seq = [eps + bias] * T
    out = core_eq.run_ddim(x_T, eps_hat_seq, abars)
    dev = np.max(np.abs(out - x0))
    assert dev > 1e-3, f"biased prediction deviated only {dev}"
    # and the deviation grows with the bias (direction sanity)
    out2 = core_eq.run_ddim(x_T, [eps + 0.1 * np.ones_like(eps)] * T, abars)
    assert np.max(np.abs(out2 - x0)) > dev


def test_ddim_perfect_prediction_is_exact():
    """P-2 positive (placed here as the property counterpart): ε̂ = ε keeps the
    closed-form trajectory invariant — exact to 1e-10 (hand derivation in
    test_anchor.py)."""
    betas, abars, x0, rng = _fixture(seed=3)
    eps = rng.standard_normal((T_P, D_A))
    x_T = core_eq.q_sample(x0, T, eps, abars)
    out = core_eq.run_ddim(x_T, [eps] * T, abars)
    assert np.allclose(out, x0, rtol=0, atol=1e-10)
