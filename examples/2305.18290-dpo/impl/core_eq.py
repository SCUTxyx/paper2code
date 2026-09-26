"""DPO loss — literal implementation of paper Eq.(7) (formula version), with
analytic gradients.

Source: Rafailov et al., arXiv:2305.18290, §4 Eq.(7).
Implementation note: the inputs are **log-prob differences** (log π − log π_ref),
never probability ratios (ratios underflow and lose precision; mathematically
equivalent, numerically not — see REPORT finding 1).
numpy-only, float64.
"""

import numpy as np


def _softplus(x):
    """log(1+e^x), numerically stable."""
    return np.logaddexp(0.0, x)


def _sigmoid(x):
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.clip(x, None, 700))),
                    np.exp(np.clip(x, -700, None)) / (1.0 + np.exp(np.clip(x, -700, None))))


def dpo_loss(logp_w, logp_l, logp_w_ref, logp_l_ref, beta=0.1):
    """Eq.(7): z = β[(logπ_w − logπ_ref,w) − (logπ_l − logπ_ref,l)],
    L = −log σ(z) = softplus(−z). Scalars or same-shaped arrays (batch mean)."""
    logp_w = np.asarray(logp_w, dtype=np.float64)
    logp_l = np.asarray(logp_l, dtype=np.float64)
    logp_w_ref = np.asarray(logp_w_ref, dtype=np.float64)
    logp_l_ref = np.asarray(logp_l_ref, dtype=np.float64)
    z = beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref))   # Eq.(7) σ argument
    return float(np.mean(_softplus(-z)))                         # −log σ(z)


def dpo_loss_and_grad(logp_w, logp_l, logp_w_ref, logp_l_ref, beta=0.1):
    """Returns (L, analytic gradients for each input). ∂L/∂z = −σ(−z); the chain
    rule splits into four paths; averaged over the batch."""
    logp_w = np.asarray(logp_w, dtype=np.float64)
    logp_l = np.asarray(logp_l, dtype=np.float64)
    logp_w_ref = np.asarray(logp_w_ref, dtype=np.float64)
    logp_l_ref = np.asarray(logp_l_ref, dtype=np.float64)
    z = beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref))
    L = float(np.mean(_softplus(-z)))
    s = _sigmoid(-z) / np.asarray(logp_w).size      # ∂L/∂z (batch mean)
    g = {}
    g["logp_w"] = -beta * s                          # ∂z/∂logp_w = +β
    g["logp_l"] = +beta * s                          # ∂z/∂logp_l = −β
    g["logp_w_ref"] = +beta * s
    g["logp_l_ref"] = -beta * s
    return L, g
