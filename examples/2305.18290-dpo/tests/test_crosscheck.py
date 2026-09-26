"""DPO 双实现互对拍:Eq.(7) 直写 vs Bradley–Terry 路径;Eq.(4)(5) 往返恒等。"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq
import core_pseudo


def test_eq7_vs_bradley_terry():
    """两条推导路径(Eq.7 直写 / Eq.6 先算 p 再取 −log)一致到 1e-12。"""
    rng = np.random.default_rng(0)
    for seed, beta in ((0, 0.1), (1, 0.5), (2, 2.0)):
        r = np.random.default_rng(seed)
        lw, ll = r.standard_normal(8), r.standard_normal(8)
        rw, rl = r.standard_normal(8), r.standard_normal(8)
        a = core_eq.dpo_loss(lw, ll, rw, rl, beta=beta)
        b = core_pseudo.dpo_loss_bt(lw, ll, rw, rl, beta=beta)
        assert abs(a - b) < 1e-12, f"β={beta}: {a} vs {b}"


def test_reward_roundtrip():
    """P3: r → π*(Eq.4) → r̂(Eq.5) 精确还原 r(验证 β·logZ 项不能丢)。"""
    rng = np.random.default_rng(1)
    K, beta = 6, 0.2
    r = rng.uniform(-1.0, 1.0, K)
    logp_ref = rng.standard_normal(K) - 1.5
    logp_ref = logp_ref - logp_ref.max()      # 任意未归一的 log π_ref
    pi_star, logZ = core_pseudo.optimal_policy(r, logp_ref, beta=beta)
    r_hat = core_pseudo.reward_from_policy(np.log(pi_star), logp_ref,
                                           beta=beta, logZ=logZ)
    assert np.allclose(r_hat, r, rtol=0, atol=1e-12)
    # 反例对照:丢掉 β·logZ 项则不能还原(恒等式的组成部分是必要的)
    r_noZ = core_pseudo.reward_from_policy(np.log(pi_star), logp_ref, beta=beta, logZ=0.0)
    assert not np.allclose(r_noZ, r, atol=1e-3)


def test_optimal_policy_normalized():
    """P4: Σ_y π*(y|x) = 1。"""
    rng = np.random.default_rng(2)
    r = rng.uniform(-2.0, 2.0, 7)
    logp_ref = rng.standard_normal(7)
    pi_star, _ = core_pseudo.optimal_policy(r, logp_ref, beta=0.1)
    assert abs(pi_star.sum() - 1.0) < 1e-12
