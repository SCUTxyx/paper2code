"""DPO — independent formulations: the Bradley–Terry path (the Eq.6→7 derivation
order) and the optimal-policy utilities (Eq.4)(5).

The formula version writes Eq.(7)'s z and softplus directly; this module follows
the paper's derivation order: first the preference probability p = σ(z), loss =
−log p (per sample); plus the Eq.(4) optimal-policy construction and the Eq.(5)
reward inversion, for the cross-check in test_crosscheck.py.
"""

import numpy as np


def dpo_loss_bt(logp_w, logp_l, logp_w_ref, logp_l_ref, beta=0.1):
    """Eq.(6)→(7) path: p = σ(z) per sample, L = mean of −log p."""
    logp_w = np.asarray(logp_w, dtype=np.float64)
    logp_l = np.asarray(logp_l, dtype=np.float64)
    logp_w_ref = np.asarray(logp_w_ref, dtype=np.float64)
    logp_l_ref = np.asarray(logp_l_ref, dtype=np.float64)
    total = 0.0
    for i in range(logp_w.size):
        z = beta * ((logp_w.ravel()[i] - logp_w_ref.ravel()[i])
                    - (logp_l.ravel()[i] - logp_l_ref.ravel()[i]))
        p = 1.0 / (1.0 + np.exp(-z))             # σ(z), Bradley–Terry Eq.(6)
        total += -np.log(p)
    return float(total / logp_w.size)


def optimal_policy(r, logp_ref, beta=0.1):
    """Eq.(4): π*(y|x) = π_ref(y|x)·exp(r/β) / Z(x).
    r: (K,) rewards per candidate response; logp_ref: (K,) log π_ref.
    Returns (π* (K,), logZ)."""
    r = np.asarray(r, dtype=np.float64)
    logp_ref = np.asarray(logp_ref, dtype=np.float64)
    logits = logp_ref + r / beta                 # log π_ref + r/β
    m = logits.max()                             # numerically stable logZ
    logZ = m + np.log(np.exp(logits - m).sum())
    return np.exp(logits - logZ), float(logZ)


def reward_from_policy(logp, logp_ref, beta=0.1, logZ=0.0):
    """Eq.(5): r = β·log(π/π_ref) + β·logZ. Pointwise reward recovery."""
    logp = np.asarray(logp, dtype=np.float64)
    logp_ref = np.asarray(logp_ref, dtype=np.float64)
    return beta * (logp - logp_ref) + beta * logZ
