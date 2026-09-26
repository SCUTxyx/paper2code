"""DPO 锚点测试:σ/softplus 精确常数手算。

手算:L = −log σ(z) = log(1+e^{−z})。
- z = 0   → L = log 2 = 0.6931471805599453;
- z = ln3 → σ(z) = 3/4 → L = log(4/3) = 0.28768207245178085;
- z = −ln3 → L = log(1 + e^{ln3}) = log 4 = 1.3862943611198906。
构造:β = 0.1,(logπ_w−logπ_ref,w) − (logπ_l−logπ_ref,l) = 10·ln3 ⇒ z = ln3。
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq

LN3 = 1.0986122886681098
BETA = 0.1


def _config(z):
    """构造四路 log-prob,使 β[(w−w_ref)−(l−l_ref)] = z。"""
    diff = z / BETA
    return (np.array([diff]), np.zeros(1), np.zeros(1), np.zeros(1))


def test_z_zero_gives_log2():
    lw, ll, rw, rl = _config(0.0)
    assert abs(core_eq.dpo_loss(lw, ll, rw, rl, beta=BETA) - np.log(2.0)) < 1e-15


def test_z_ln3_gives_log_4_3():
    lw, ll, rw, rl = _config(LN3)
    L = core_eq.dpo_loss(lw, ll, rw, rl, beta=BETA)
    assert abs(L - np.log(4.0 / 3.0)) < 1e-12
    assert abs(L - 0.28768207245178085) < 1e-12


def test_z_neg_ln3_gives_log4():
    lw, ll, rw, rl = _config(-LN3)
    L = core_eq.dpo_loss(lw, ll, rw, rl, beta=BETA)
    assert abs(L - np.log(4.0)) < 1e-12
