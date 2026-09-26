"""Adam 梯度检查:m̂_t、v̂_t 对梯度历史的解析敏感系数 vs 中心差分。

手写推导(由 A3 闭式展开式求导):
  ∂m̂_t/∂g_k = (1-β1)·β1^{t-k} / (1-β1^t)                    (k=1..t)
  ∂v̂_t/∂g_k = 2(1-β2)·β2^{t-k}·g_k / (1-β2^t)
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq

T, N = 6, 3
BETA1, BETA2 = 0.9, 0.999


def _run(seq_flat):
    return core_eq.adam_run(seq_flat.reshape(T, N), theta0=np.zeros(N))


def test_grad_m_hat_wrt_history():
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((T, N))
    analytic = np.zeros(T * N)
    for k in range(T):  # g_k 的 0-based 下标 k ↔ 论文记号 t-k = T-1-k
        analytic[k * N] = (1 - BETA1) * BETA1 ** (T - 1 - k) / (1 - BETA1 ** T)
    err = assert_grad_close(
        lambda x: _run(x)["m_hats"][T - 1, 0], seq.ravel(), analytic,
        rtol=1e-6, name="d m̂_T / d g_k")
    print(f"\n[adam] m̂ max rel err = {err:.2e}")


def test_grad_v_hat_wrt_history():
    rng = np.random.default_rng(1)
    seq = rng.standard_normal((T, N))
    analytic = np.zeros(T * N)
    for k in range(T):
        analytic[k * N] = 2 * (1 - BETA2) * BETA2 ** (T - 1 - k) * seq[k, 0] / (1 - BETA2 ** T)
    err = assert_grad_close(
        lambda x: _run(x)["v_hats"][T - 1, 0], seq.ravel(), analytic,
        rtol=1e-6, name="d v̂_T / d g_k")
    print(f"\n[adam] v̂ max rel err = {err:.2e}")
