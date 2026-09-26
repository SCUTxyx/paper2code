"""LoRA — independent formulation: two-stage application without forming ΔW
(the memory-efficient path used in practice).

Difference vs the formula version: never materializes the d×k matrix BA; computes
h = W0 x + (α/r)·B·(A x), contracting over the rank dimension first. Different
multiplication order and a different failure surface (no d×k allocation). For the
cross-check in test_crosscheck.py. Also provides the merge/unmerge utilities.
"""

import numpy as np


def lora_forward_stream(W0, B, A, x, alpha=1.0):
    """Two-stage path: h = W0 x + (α/r)·B·(A x). No d×k matrix is formed."""
    W0 = np.asarray(W0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    r = A.shape[0]
    return W0 @ x + (alpha / r) * (B @ (A @ x))


def merge(W0, B, A, alpha=1.0):
    """Merged inference weight: W = W0 + (α/r)·B A (§4.4 deployment identity)."""
    r = A.shape[0]
    return np.asarray(W0, dtype=np.float64) + (alpha / r) * (B @ A)


def unmerge(W, B, A, alpha=1.0):
    """Recover the frozen base weight from a merged weight: W0 = W − (α/r)·BA."""
    r = A.shape[0]
    return np.asarray(W, dtype=np.float64) - (alpha / r) * (B @ A)
