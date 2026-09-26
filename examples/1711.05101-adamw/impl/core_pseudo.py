"""AdamW — independent formulation: the decoupled composition view (claim W5).

Difference vs the formula version: instead of one fused update line, this path
computes a plain Adam step first and applies the decay shrink as a *separate*
operation on the previous parameters — θ_t = adam_step(θ_{t-1}, g) − lr·λ·θ_{t-1}.
Additivity of the two paths is the mathematical content of "decoupled". For the
cross-check in test_crosscheck.py.
"""

import numpy as np


def adam_step(theta, g, m, v, t, lr, beta1, beta2, eps):
    """One standard Adam step (no decay). Returns (theta_next, m, v)."""
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return theta - lr * m_hat / (np.sqrt(v_hat) + eps), m, v


def adamw_run_decoupled(grad_seq, theta0, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8,
                        weight_decay=0.01):
    """Same interface as core_eq.adamw_run; decay applied separately (W5)."""
    g_seq = np.asarray(grad_seq, dtype=np.float64)
    theta = np.array(theta0, dtype=np.float64)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    thetas, m_hats, v_hats = [theta.copy()], [], []
    for t in range(1, g_seq.shape[0] + 1):
        g = g_seq[t - 1]
        prev = theta.copy()                                    # decay acts on θ_{t-1}
        theta, m, v = adam_step(theta, g, m, v, t, lr, beta1, beta2, eps)
        theta = theta - lr * weight_decay * prev               # separate shrink step
        thetas.append(theta.copy())
        m_hats.append(m / (1 - beta1 ** t))
        v_hats.append(v / (1 - beta2 ** t))
    return {"thetas": np.array(thetas), "m_hats": np.array(m_hats), "v_hats": np.array(v_hats)}
