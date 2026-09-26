"""Diffusion Policy math core — independent formulation: per-step loops.

The formula version uses closed-form coefficients and vectorized slicing; this
module (a) noises step-by-step per DDPM Eq.(2) instead of the closed form, (b)
runs the DDIM chain in the predicted-x_0 form written per step, (c) executes the
receding horizon with an explicit clock loop. For the cross-checks.
"""

import numpy as np


def q_sample_iterative(x0, t, eps_seq, betas):
    """DDPM Eq.(2) step-by-step noising."""
    x = np.array(x0, dtype=np.float64)
    for s in range(1, t + 1):
        e = np.asarray(eps_seq[s - 1], dtype=np.float64)
        x = np.sqrt(1.0 - betas[s - 1]) * x + np.sqrt(betas[s - 1]) * e
    return x


def run_ddim_loop(x_T, eps_hat_seq, abars):
    """DDIM chain in the predicted-x_0 form, step by step:
    x0_pred = (x_t − √(1-ᾱ_t) ε̂)/√ᾱ_t;  x_{t-1} = √ᾱ_{t-1} x0_pred + √(1-ᾱ_{t-1}) ε̂."""
    x = np.asarray(x_T, dtype=np.float64)
    for t in range(len(eps_hat_seq), 0, -1):
        ab_t, ab_prev = abars[t], abars[t - 1]
        e = np.asarray(eps_hat_seq[t - 1], dtype=np.float64)
        x0_pred = (x - np.sqrt(1.0 - ab_t) * e) / np.sqrt(ab_t)
        x = np.sqrt(ab_prev) * x0_pred + np.sqrt(1.0 - ab_prev) * e
    return x


def execute_receding_loop(chunks, t_a, horizon):
    """Explicit-clock receding-horizon executor: replans at t = 0, T_a, 2T_a, …;
    executes chunk[0:T_a] except the final chunk runs to min(T_p, remaining)."""
    chunks = [np.asarray(c, dtype=np.float64) for c in chunks]
    out = []
    t, i = 0, 0
    while t < horizon:
        chunk = chunks[min(i, len(chunks) - 1)]
        last = i >= len(chunks) - 1
        take = min(len(chunk), horizon - t) if last else min(t_a, horizon - t)
        for j in range(take):
            out.append(chunk[j])
            t += 1
        i += 1
    return np.array(out[:horizon])
