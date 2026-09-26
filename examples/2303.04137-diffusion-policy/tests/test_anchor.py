"""Diffusion Policy anchor tests: DDIM trajectory invariance + hand-literal
receding-horizon stitching.

Hand derivation of P-2 (DDIM, Song et al. Eq.12, η=0):
  x_t = √ᾱ_t x0 + √(1-ᾱ_t) ε   (closed-form trajectory with the SAME ε)
  x_{t-1} = √ᾱ_{t-1}·(x_t − √(1-ᾱ_t)ε̂)/√ᾱ_t + √(1-ᾱ_{t-1})·ε̂
  with ε̂ = ε:  (x_t − √(1-ᾱ_t)ε)/√ᾱ_t = x0  ⇒  x_{t-1} = √ᾱ_{t-1} x0 + √(1-ᾱ_{t-1}) ε = x_{t-1}^{closed}
  ⇒ the closed-form trajectory is an invariant of the DDIM map when ε̂ = ε,
  and at t=0 (ᾱ_0 = 1) the chain lands exactly on x_0.

Hand computation of P-3 stitching (T_p=4, T_a=2, 3 chunks of rows 10·i + [0,1,2,3],
horizon 7): replans at t=0,2,4; executed = [0,1, 10,11, 20,21, 22] — the final
chunk runs min(4, 7-4) = 3 rows at episode end.
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

T, D_A, T_P, T_A = 60, 4, 16, 8


def test_ddim_chain_recovers_x0():
    """P-2: DDIM with perfect ε̂ recovers x_0 exactly from the T-step closed-form
    sample (both implementations, atol 1e-10)."""
    betas = core_eq.linear_beta_schedule(T)
    abars = core_eq.alpha_bar_seq(betas)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((T_P, D_A))
    eps = rng.standard_normal((T_P, D_A))
    x_T = core_eq.q_sample(x0, T, eps, abars)
    out = core_eq.run_ddim(x_T, [eps] * T, abars)
    out_b = core_pseudo.run_ddim_loop(x_T, [eps] * T, abars)
    assert np.allclose(out, x0, rtol=0, atol=1e-10)
    assert np.allclose(out_b, x0, rtol=0, atol=1e-10)


def test_receding_horizon_stitching():
    """P-3: hand-literal stitching (derivation above), both implementations."""
    chunks = [10 * i + np.arange(4)[:, None] * np.ones((1, 2)) for i in range(3)]
    chunks = [np.asarray(c, dtype=np.float64) for c in chunks]
    out = core_eq.execute_receding(chunks, t_a=2, horizon=7)
    expect = np.array([[0, 0], [1, 1], [10, 10], [11, 11],
                       [20, 20], [21, 21], [22, 22]], dtype=np.float64)
    assert out.shape == (7, 2)
    assert np.all(out == expect)
    out_b = core_pseudo.execute_receding_loop(chunks, t_a=2, horizon=7)
    assert np.all(out_b == expect)


def test_chunk_head_execution():
    """P-3 supplementary: with horizon = 2·T_a (no final-chunk remainder), the
    executed rows are exactly chunk heads [0:T_a]."""
    rng = np.random.default_rng(1)
    chunks = [rng.standard_normal((T_P, D_A)) for _ in range(2)]
    out = core_eq.execute_receding(chunks, T_A, 2 * T_A)
    assert np.all(out[:T_A] == chunks[0][:T_A])
    assert np.all(out[T_A:] == chunks[1][:T_A])
