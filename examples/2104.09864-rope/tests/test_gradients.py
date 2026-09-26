"""RoPE 梯度检查(R5):解析梯度(旋转转置)vs 中心差分。"""

import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1] / "impl"
sys.path.insert(0, str(IMPL))
for _m in [m for m in sys.modules if m.startswith("core_")]:
    del sys.modules[_m]
import numpy as np

from gradcheck import assert_grad_close

import core_eq
import core_pseudo

D, M = 8, 17


def test_grad_rotate_wrt_x():
    """∂ wᵀf(x,m)/∂x = R^{m⊤}w = f(w,−m)(旋转矩阵的转置即逆旋转)。"""
    x = np.random.default_rng(0).standard_normal(D)
    w = np.random.default_rng(1).standard_normal(D)
    analytic = core_eq.rope_rotate(w, -M)          # R^{m⊤} = R^{−m}
    err = assert_grad_close(
        lambda v: float(w @ core_eq.rope_rotate(v, M)),
        x, analytic, rtol=1e-6, name="d w·R^m x / dx")
    print(f"\n[rope] rotate grad err = {err:.2e}")


def test_grad_score_wrt_q():
    """∂ ⟨f(q,m),f(k,n)⟩/∂q = R^{m⊤}R^n k = f(k, n−m)(相对位置结构的直接推论)。"""
    rng = np.random.default_rng(2)
    q, k = rng.standard_normal(D), rng.standard_normal(D)
    m, n = 5, 41
    analytic = core_eq.rope_rotate(k, n - m)
    err = assert_grad_close(
        lambda v: core_eq.rope_score(v, k, m, n),
        q, analytic, rtol=1e-6, name="d score/dq")
    print(f"\n[rope] score grad err = {err:.2e}")


def test_grad_via_matrix_form():
    """显式矩阵路径(core_pseudo)的梯度同样应与中心差分一致。"""
    x = np.random.default_rng(3).standard_normal(D)
    w = np.random.default_rng(4).standard_normal(D)
    analytic = core_pseudo.rope_rotate_matrix(w, -M)
    err = assert_grad_close(
        lambda v: float(w @ core_pseudo.rope_rotate_matrix(v, M)),
        x, analytic, rtol=1e-6, name="d (matrix path)/dx")
    print(f"\n[rope] matrix-path grad err = {err:.2e}")
