"""AdamW gradient check (W4): the decay-path gradient has a closed form.

Hand derivation: under any fixed gradient sequence the recursion is linear in θ_0,
θ_t = (1−lr·λ)^t·θ_0 + (terms independent of θ_0) ⇒ ∂θ_t/∂θ_0 = (1−lr·λ)^t exactly.
(The gradient w.r.t. g is O(ε) and numerically meaningless — see TEST_PLAN W4 note.)
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


def test_grad_theta_t_wrt_theta0():
    for wd, seq_kind in ((0.2, "zero"), (0.1, "constant")):
        lr = 0.05
        seq = np.zeros((T, N)) if seq_kind == "zero" else np.tile([0.7, -0.4, 1.1], (T, 1))
        factor = (1 - lr * wd) ** T

        rng = np.random.default_rng(0)
        theta0 = rng.standard_normal(N)
        w = rng.standard_normal(N)

        def f(x, seq=seq, w=w, lr=lr, wd=wd):
            out = core_eq.adamw_run(seq, x, lr=lr, weight_decay=wd)
            return float(w @ out["thetas"][T])

        err = assert_grad_close(f, theta0, factor * w, rtol=1e-6,
                                name=f"d θ_T/d θ_0 (λ={wd}, {seq_kind} g)")
        print(f"\n[adamw] λ={wd}, {seq_kind} gradient: max rel err = {err:.2e}")
