"""DDPM forward process — closed-form marginal per paper Eq.(4) (formula version).

Source: Ho et al., arXiv:2006.11239, §2 Eq.(1)(2)(4), §4 linear schedule.
numpy-only, float64.
"""

import numpy as np


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """§4: β_t grows linearly from 1e-4 to 0.02 with T=1000. Returns (T,);
    index t-1 ↔ the paper's β_t."""
    return np.linspace(beta_start, beta_end, T)


def alpha_bar_seq(betas):
    """ᾱ_t = Π_{s≤t} α_s with α_t = 1-β_t (notation below Eq.2)."""
    return np.cumprod(1.0 - np.asarray(betas, dtype=np.float64))


def q_sample(x0, t, eps, abars):
    """Eq.(4): x_t = √ᾱ_t x0 + √(1-ᾱ_t) ε. t is 1-based."""
    x0 = np.asarray(x0, dtype=np.float64)
    ab = abars[t - 1]
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * np.asarray(eps, dtype=np.float64)
