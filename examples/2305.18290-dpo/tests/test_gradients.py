"""DPO 梯度检查(P5):四路 log-prob 的解析梯度 vs 中心差分。"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq

N_BATCH, BETA = 5, 0.3


def test_grads_four_paths():
    rng = np.random.default_rng(0)
    lw, ll = rng.standard_normal(N_BATCH), rng.standard_normal(N_BATCH)
    rw, rl = rng.standard_normal(N_BATCH), rng.standard_normal(N_BATCH)
    L, g = core_eq.dpo_loss_and_grad(lw, ll, rw, rl, beta=BETA)
    vec = np.concatenate([lw, ll, rw, rl])
    analytic = np.concatenate([g["logp_w"], g["logp_l"],
                               g["logp_w_ref"], g["logp_l_ref"]])

    def f(x):
        a, b, c, d = np.split(x, 4)
        return core_eq.dpo_loss(a, b, c, d, beta=BETA)

    err = assert_grad_close(f, vec, analytic, rtol=1e-6, name="dL/d(logp)")
    print(f"\n[dpo] grad max rel err = {err:.2e}")


def test_grad_structure_at_reference_point():
    """π_θ = π_ref 处 σ(−z)=½:批均值损失下四路梯度量级恰为 β/(2N)。"""
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(4), rng.standard_normal(4)
    _, g = core_eq.dpo_loss_and_grad(a, b, a, b, beta=BETA)
    for key in ("logp_w", "logp_w_ref"):
        assert np.allclose(np.abs(g[key]), BETA / 2 / a.size, rtol=1e-12), key
