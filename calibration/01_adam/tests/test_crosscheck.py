"""Adam 双实现互对拍:递推版(Algorithm 1 直译) vs 历史展开和版(独立表述)。"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def _compare(cfg):
    rng = np.random.default_rng(0)
    seq = rng.standard_normal((50, 4))
    theta0 = rng.standard_normal(4)
    a = core_eq.adam_run(seq, theta0, **cfg)
    b = core_pseudo.adam_run_history(seq, theta0, **cfg)
    for key in ("thetas", "ms", "vs", "m_hats", "v_hats"):
        assert np.allclose(a[key], b[key], rtol=1e-9, atol=1e-14), key


def test_default_hyperparams():
    _compare({})


def test_other_hyperparams():
    _compare({"lr": 0.03, "beta1": 0.5, "beta2": 0.99, "eps": 1e-6})
