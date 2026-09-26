# Method card: AdamW decoupled weight decay (Loshchilov & Hutter 2019)

- Paper: *Decoupled Weight Decay Regularization*, arXiv:1711.05101 (ICLR 2019)
- Scope statement: only the Algorithm 2 update rule (AdamW) and its decoupling property;
  the schedule multiplier η_t is fixed to 1, the ICLR reviews' generalization claims and
  the hyperparameter-transfer grid search are not covered.

## Problem formalization

Standard Adam couples L2 regularization into the gradient, so the adaptive scaling
$m̂/\sqrt{v̂}$ rescales the decay too. AdamW removes the coupling:

$$\theta_t = \theta_{t-1} - \mathrm{lr}\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon} + \lambda\,\theta_{t-1}\Big)$$

## Symbol table

| Symbol | Meaning | Source |
|---|---|---|
| $\theta_t$ | parameters | Algorithm 2 |
| $g_t$ | (stochastic) gradient | Algorithm 2 |
| $m_t, v_t, \hat m_t, \hat v_t$ | Adam moment recursions (identical to Adam) | Algorithm 2 |
| $\lambda$ | weight decay coefficient | Algorithm 2 |
| $\eta_t, \alpha$ | schedule multiplier and base lr (η_t := 1 here) | Algorithm 2 |

## Core formula (Algorithm 2, with η_t = 1)

- $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$;  $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
- $\hat m_t = m_t/(1-\beta_1^t)$;  $\hat v_t = v_t/(1-\beta_2^t)$
- $\theta_t = \theta_{t-1} - \mathrm{lr}\,(\hat m_t/(\sqrt{\hat v_t}+\varepsilon)) - \mathrm{lr}\cdot\lambda\cdot\theta_{t-1}$

## Algorithm box (transcription of Algorithm 2, η_t = 1)

1. Update the moment recursions exactly as Adam (the decay plays no role here);
2. Bias-correct as in Adam;
3. Apply the adaptive step **plus** a decay term proportional to the *previous*
   parameter value: θ_t = θ_{t-1} − lr·(m̂/(√v̂+ε)) − lr·λ·θ_{t-1}.

## Claims list

- **W1** (decoupling, the paper's core claim): the decay path never passes through the
  adaptive scaling — the moment trajectories $(\hat m_t, \hat v_t)$ are **identical for
  every λ**, and the decay contribution is exactly lr·λ·θ_{t-1} independent of the
  gradient history. (Adam-with-L2 violates both.)
- **W2** (λ=0 degeneracy): λ=0 reduces AdamW exactly to Adam (same recursions, same
  update).
- **W3** (zero-gradient closed form): with g=0, θ_t = θ_0·(1−lr·λ)^t — a pure exponential
  shrink with no adaptive scaling, exactly.
- **W4** (decay-path gradient): under g=0, ∂θ_t/∂θ_0 = (1−lr·λ)^t (closed form;
  central-difference checkable). The gradient w.r.t. g is numerically meaningless at the
  ε scale and deliberately not used as a test.
- **W5** (composition view): AdamW(θ, g, λ) == Adam_step(θ, g) − lr·λ·θ — the update is
  the Adam step plus a separate shrink; two independent formulations must agree exactly.
