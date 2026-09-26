"""InfoNCE/CLIP 梯度检查(N5):解析梯度(含归一化投影)vs 中心差分。"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq

N_BATCH, D = 4, 5


def test_grads_wrt_I_and_T():
    rng = np.random.default_rng(0)
    I = rng.standard_normal((N_BATCH, D))
    T = rng.standard_normal((N_BATCH, D))
    L, dI, dT = core_eq.clip_loss(I, T, tau=0.3, return_grads=True)

    err_I = assert_grad_close(
        lambda x: core_eq.clip_loss(x.reshape(N_BATCH, D), T, tau=0.3),
        I, dI, rtol=1e-6, name="dL/dI")
    err_T = assert_grad_close(
        lambda x: core_eq.clip_loss(I, x.reshape(N_BATCH, D), tau=0.3),
        T, dT, rtol=1e-6, name="dL/dT")
    print(f"\n[infonce] dL/dI err={err_I:.2e}, dL/dT err={err_T:.2e}")


def test_grad_zero_for_identical_normalized_inputs():
    """I = T(单位向量)时,损失对每行输入的梯度应与该行正交(归一化投影),
    表现为:沿行方向的分量精确为 0。"""
    rng = np.random.default_rng(1)
    V = rng.standard_normal((N_BATCH, D))
    _, dI, dT = core_eq.clip_loss(V, V, tau=0.1, return_grads=True)
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    proj_I = np.sum(dI * Vn, axis=1)
    proj_T = np.sum(dT * Vn, axis=1)
    assert np.allclose(proj_I, 0.0, atol=1e-12), proj_I
    assert np.allclose(proj_T, 0.0, atol=1e-12), proj_T
