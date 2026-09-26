"""DDPM forward process — independent formulations: step-by-step recursion (Eq.2)
and the unrolled weighted sum (derivation in METHOD_CARD).

Neither uses the closed-form √ᾱ_t coefficients directly; one noises step by step,
the other sums weighted historical noises — both for the cross-check against the
formula version in test_crosscheck.py (D2 exact identity).
"""

import numpy as np


def q_sample_iterative(x0, t, eps_seq, betas):
    """Eq.(2) step-by-step noising: x_s = √(1-β_s) x_{s-1} + √β_s ε_s,
    using the first t entries of eps_seq."""
    x = np.array(x0, dtype=np.float64)
    for s in range(1, t + 1):
        e = np.asarray(eps_seq[s - 1], dtype=np.float64)
        x = np.sqrt(1.0 - betas[s - 1]) * x + np.sqrt(betas[s - 1]) * e
    return x


def q_sample_unrolled(x0, t, eps_seq, betas):
    """Unrolled identity: x_t = √ᾱ_t x0 + Σ_s √(ᾱ_t/ᾱ_s)·√(1-α_s)·ε_s."""
    betas = np.asarray(betas, dtype=np.float64)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    out = np.sqrt(abar[t - 1]) * np.array(x0, dtype=np.float64)
    for s in range(1, t + 1):
        coef = np.sqrt(abar[t - 1] / abar[s - 1]) * np.sqrt(1.0 - alphas[s - 1])
        out = out + coef * np.asarray(eps_seq[s - 1], dtype=np.float64)
    return out


def q_sample_batch(x0, t, rng, abars, n):
    """Monte Carlo of the closed-form marginal: x0 broadcast over n trajectories
    (Eq.4 sampling form, vectorized)."""
    x0 = np.asarray(x0, dtype=np.float64)
    ab = abars[t - 1]
    eps = rng.standard_normal((n, x0.shape[-1]))
    return np.sqrt(ab) * x0[None, :] + np.sqrt(1.0 - ab) * eps


def q_sample_iterative_batch(x0, t, rng, betas, n):
    """Monte Carlo of step-by-step noising (trajectories vectorized, time looped)."""
    x = np.tile(np.asarray(x0, dtype=np.float64), (n, 1))
    for s in range(1, t + 1):
        eps = rng.standard_normal(x.shape)
        x = np.sqrt(1.0 - betas[s - 1]) * x + np.sqrt(betas[s - 1]) * eps
    return x
