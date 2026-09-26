"""Adam 锚点测试:前几步更新值闭式手算(推导见注释,常数落字面量)。

手算(θ0=0, g1=1, g2=2, β1=0.9, β2=0.999, lr=0.1, ε=1e-8):
  t=1: m₁=0.1, v₁=0.001; m̂₁=m₁/(1-0.9)=1, v̂₁=v₁/(1-0.999)=1
       θ₁ = -0.1·1/(1+1e-8) = -0.099999999
  t=2: m₂=0.9·0.1+0.1·2=0.29; v₂=0.999·0.001+0.001·4=0.004999
       m̂₂=0.29/0.19=1.5263158; v̂₂=0.004999/0.001999=2.5007504
       θ₂ = -0.1 - 0.1·1.5263158/√2.5007504 = -0.1 - 0.0965182 = -0.1965182
"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

import core_eq


def test_first_two_steps_hand_computed():
    g = np.array([[1.0], [2.0]])
    out = core_eq.adam_run(g, theta0=np.zeros(1), lr=0.1,
                           beta1=0.9, beta2=0.999, eps=1e-8)
    # 偏差修正的机制性锚点:首步 m̂₁ = v̂₁ = 1 精确成立
    assert np.allclose(out["m_hats"][0], 1.0, rtol=0, atol=1e-15)
    assert np.allclose(out["v_hats"][0], 1.0, rtol=0, atol=1e-15)
    assert np.allclose(out["thetas"][1], -0.099999999, rtol=1e-9)
    assert np.allclose(out["thetas"][2], -0.1965182, rtol=1e-5)


def test_five_steps_constant_gradient_hand_computed():
    """常梯度 c=[0.7,-0.4] 五步:每步更新 = α·c/(|c|+ε) ≈ α·sign(c)
    (0.1·0.7/0.7 = 0.1,正是 A1 的「步长≈lr 与尺度无关」)。
    θ₅ = -5·0.1·(1 - ε/|c|) = -0.4999999929 / +0.4999999875(手算)。"""
    c = np.array([0.7, -0.4])
    out = core_eq.adam_run(np.tile(c, (5, 1)), theta0=np.zeros(2), lr=0.1)
    assert np.allclose(out["thetas"][5], [-0.49999999, 0.49999999], rtol=1e-7)
