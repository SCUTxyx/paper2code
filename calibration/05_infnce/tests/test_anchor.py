"""InfoNCE/CLIP 锚点测试:2×2 旋转构造手算 + N=1 退化。

手算(I = 单位阵两行,T = 旋转 φ=30° 的两行,均为单位向量,τ=1):
  c = cos30° = 0.8660254, s = 0.5
  S = [[c, -s], [s, c]](S[i,i] = c 为正样本)
  e^c = 2.3774431, e^{±s} = 1.6487213 / 0.6065307
  L_i2t = -½[(c - ln(e^c+e^{-s})) + (c - ln(e^s+e^c))]
        = -½[(0.8660254-1.0932560) + (0.8660254-1.3928141)] = 0.3770096
  L_t2i = -½[(c - ln(e^c+e^s)) + (c - ln(e^{-s}+e^c))] = 0.3770096(本构造两向相等)
  L = 0.3770096
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def test_two_by_two_rotation_hand_computed():
    c, s = np.cos(np.pi / 6), 0.5
    I = np.eye(2)
    T = np.array([[c, -s], [s, c]])  # 旋转 30°,行已单位化
    L = core_eq.clip_loss(I, T, tau=1.0)
    assert abs(L - 0.3770096) < 1e-5  # 值对拍规范 rtol=1e-5


def test_single_pair_degenerates_to_zero():
    """N=1 时 softmax 只有一个元素 → 损失精确为 0(任意温度)。"""
    I = np.array([[1.0, 2.0]])
    T = np.array([[3.0, -1.0]])
    for tau in (0.07, 1.0, 100.0):
        assert core_eq.clip_loss(I, T, tau=tau) == 0.0
