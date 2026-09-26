"""AdamW — literal implementation of paper Algorithm 2 (formula version).

Source: Loshchilov & Hutter, arXiv:1711.05101, Algorithm 2, with the schedule
multiplier η_t fixed to 1 (documented in METHOD_CARD). numpy-only, float64.
"""

import numpy as np


def adamw_run(grad_seq, theta0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    """Run Algorithm 2 over a gradient sequence.

    grad_seq: (T, n); theta0: (n,). Returns dict with thetas (T+1, n) and the
    moment trajectories m_hats/v_hats (T, n) — the trajectories are the object of
    the decoupling test W1.
    """
    g_seq = np.asarray(grad_seq, dtype=np.float64)
    theta = np.array(theta0, dtype=np.float64)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    thetas, m_hats, v_hats = [theta.copy()], [], []
    for t in range(1, g_seq.shape[0] + 1):
        g = g_seq[t - 1]
        m = beta1 * m + (1 - beta1) * g                        # Algorithm 2: m recursions
        v = beta2 * v + (1 - beta2) * g * g                    #   (decay plays no role)
        m_hat = m / (1 - beta1 ** t)                           # Algorithm 2: bias correction
        v_hat = v / (1 - beta2 ** t)
        # Algorithm 2 fused update: adaptive step + decay on the PREVIOUS parameters
        theta = theta - lr * (m_hat / (np.sqrt(v_hat) + eps)) - lr * weight_decay * theta
        thetas.append(theta.copy())
        m_hats.append(m_hat.copy())
        v_hats.append(v_hat.copy())
    return {"thetas": np.array(thetas), "m_hats": np.array(m_hats), "v_hats": np.array(v_hats)}
