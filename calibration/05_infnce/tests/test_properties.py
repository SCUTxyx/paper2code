"""InfoNCE/CLIP 性质测试(N1 对称 / N2 温度极限 / N3 尺度 / N4 置换)。"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def _data(seed=0, n=6, d=8):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)), rng.standard_normal((n, d))


def test_tower_swap_symmetry():
    """N1: 双向平均对塔交换不变 L(I,T) = L(T,I),精度 1e-12。"""
    I, T = _data()
    assert abs(core_eq.clip_loss(I, T, tau=0.1) - core_eq.clip_loss(T, I, tau=0.1)) < 1e-12


def test_temperature_limits():
    """N2: τ→∞ 损失 → log N(softmax→均匀);正确配对严格占优时 τ→0 损失 → 0。"""
    I, T = _data(seed=1)
    n = I.shape[0]
    # τ→∞:logits ~ 1/τ → 1e-8,CE 偏离 log N 的量级 ~ 1e-8
    assert abs(core_eq.clip_loss(I, T, tau=1e8) - np.log(n)) < 1e-7
    # τ→0:把正样本构造成严格占优(I=T,自相似=1 严格大于互相似 < 1)
    assert core_eq.clip_loss(I, I, tau=1e-5) < 1e-6
    # 单调性方向:τ 越小,正确配对越突出,损失越低
    assert core_eq.clip_loss(I, I, tau=1e-3) < core_eq.clip_loss(I, I, tau=1.0)


def test_scale_invariance():
    """N3: 内部 L2 归一化 → 损失对输入尺度不变。"""
    I, T = _data(seed=2)
    a = core_eq.clip_loss(I, T, tau=0.2)
    b = core_eq.clip_loss(5.3 * I, 0.2 * T, tau=0.2)
    assert abs(a - b) < 1e-12


def test_batch_permutation_invariance():
    """N4: 联合置换 batch 行,损失不变。"""
    I, T = _data(seed=3)
    rng = np.random.default_rng(4)
    perm = rng.permutation(I.shape[0])
    a = core_eq.clip_loss(I, T, tau=0.07)
    b = core_eq.clip_loss(I[perm], T[perm], tau=0.07)
    assert abs(a - b) < 1e-12
