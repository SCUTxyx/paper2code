"""Diffusion Policy action-diffusion math core — formula version.

Sources: Chi et al., arXiv:2303.04137 §3 (training objective, receding-horizon
execution); DDIM sampler per Song et al., arXiv:2010.02502 Eq.(12) with η=0
(DP's real-world inference path); DDPM forward process per Ho et al.,
arXiv:2006.11239 Eq.(2)(4) (DP trains with the DDPM objective).
numpy-only, float64.
"""

import numpy as np


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear schedule for this minimal repro (DP-sim uses squared-cosine —
    see REPORT limitations). Index t-1 ↔ paper's β_t."""
    return np.linspace(beta_start, beta_end, T)


def alpha_bar_seq(betas):
    """ᾱ_t = Π_{s≤t} (1-β_s), with the convention ᾱ_0 = 1 (t=0 is the data step)."""
    betas = np.asarray(betas, dtype=np.float64)
    return np.concatenate([[1.0], np.cumprod(1.0 - betas)])


def q_sample(x0, t, eps, abars):
    """DDPM closed-form marginal (Ho et al. Eq.4): x_t = √ᾱ_t x0 + √(1-ᾱ_t) ε.
    t is 1-based for the noised steps; ᾱ index is offset by the t=0 entry."""
    x0 = np.asarray(x0, dtype=np.float64)
    ab = abars[t]
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * np.asarray(eps, dtype=np.float64)


def q_sample_iterative(x0, t, eps_seq, betas):
    """DDPM forward recursion (Ho et al. Eq.2): x_s = √(1-β_s) x_{s-1} + √β_s ε_s."""
    x = np.array(x0, dtype=np.float64)
    for s in range(1, t + 1):
        e = np.asarray(eps_seq[s - 1], dtype=np.float64)
        x = np.sqrt(1.0 - betas[s - 1]) * x + np.sqrt(betas[s - 1]) * e
    return x


def ddim_step(x_t, t, eps_hat, abars):
    """DDIM update (Song et al. Eq.12, η=0): one step τ → τ-1.

    x_{τ-1} = √ᾱ_{τ-1}·(x_τ − √(1-ᾱ_τ) ε̂)/√ᾱ_τ + √(1-ᾱ_{τ-1})·ε̂
    t is the current 1-based diffusion step; ᾱ_0 = 1 lands on the data step.
    """
    x_t = np.asarray(x_t, dtype=np.float64)
    ab_t, ab_prev = abars[t], abars[t - 1]
    x0_pred = (x_t - np.sqrt(1.0 - ab_t) * eps_hat) / np.sqrt(ab_t)   # predicted x_0
    return np.sqrt(ab_prev) * x0_pred + np.sqrt(1.0 - ab_prev) * eps_hat


def run_ddim(x_T, eps_hat_seq, abars):
    """Full DDIM chain T → 0, consuming per-step predictions eps_hat_seq[T-1..0]
    (eps_hat_seq[t-1] is used at step t)."""
    x = np.asarray(x_T, dtype=np.float64)
    T = len(eps_hat_seq)
    for t in range(T, 0, -1):
        x = ddim_step(x, t, np.asarray(eps_hat_seq[t - 1], dtype=np.float64), abars)
    return x


def training_loss(eps, eps_hat):
    """§3 Eq.(5): L = mean((ε − ε̂)²). Returns (L, ∂L/∂ε̂)."""
    eps = np.asarray(eps, dtype=np.float64)
    eps_hat = np.asarray(eps_hat, dtype=np.float64)
    diff = eps_hat - eps
    n = diff.size
    return float(np.mean(diff * diff)), 2.0 * diff / n


def execute_receding(chunks, t_a, horizon):
    """Receding-horizon execution (§"receding horizon control"): chunks are emitted
    every T_a steps; each executes its first T_a rows; the FINAL chunk runs to
    min(T_p, remaining) at episode end. Returns the executed sequence with
    exactly `horizon` rows.

    chunks: list of (T_p, d_a) arrays, len = ceil(horizon / T_a).
    """
    rows = []
    t = 0
    chunks = np.asarray(chunks, dtype=np.float64)
    for i, chunk in enumerate(chunks):
        last = i == len(chunks) - 1
        take = min(chunk.shape[0], horizon - t) if last else min(t_a, horizon - t)
        for j in range(max(take, 0)):
            rows.append(chunk[j])
        t += max(take, 0)
        if t >= horizon:
            break
    # chunks exhausted before the horizon: keep executing the final chunk
    # (cyclically from its head) — same convention as the loop executor
    while t < horizon:
        chunk = chunks[-1]
        take = min(chunk.shape[0], horizon - t)
        for j in range(take):
            rows.append(chunk[j])
        t += take
    return np.array(rows[:horizon])
