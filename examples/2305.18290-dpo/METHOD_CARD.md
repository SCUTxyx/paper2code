# Method card: DPO direct preference optimization (Rafailov et al. 2023)

- Paper: *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*,
  arXiv:2305.18290 (NeurIPS 2023)
- Scope statement: only the §4 derivation chain (Eq.3–7) and the loss function itself;
  training dynamics, the RM stage, and sampling strategies are not covered.

## Problem formalization

Learn a policy from preference pairs $(y_w \succ y_l \mid x)$ while bypassing explicit
reward modeling:
$$\max_\pi\ \mathbb E_{x\sim\mathcal D,\,y\sim\pi}\big[r(x,y)\big] - \beta\,\mathbb{KL}\big(\pi \| \pi_{\mathrm{ref}}\big)\quad(\text{Eq.3})$$

## Symbol table

| Symbol | Meaning | Source |
|---|---|---|
| $\pi_\theta, \pi_{\mathrm{ref}}$ | policy / frozen reference policy | §4 |
| $\beta$ | KL strength (inversely proportional to deviation freedom) | Eq.(3) |
| $y_w, y_l$ | chosen / rejected responses in a preference pair | §4 |
| $z$ | the argument of $\sigma$ in Eq.(7) (the implicit reward difference) | Eq.(6)(7) |
| $Z(x)$ | partition function (cancels in pairwise differences) | Eq.(4)(5) |

## Core formulas (§4 derivation chain)

- Eq.(4):closed-form optimum of KL-constrained reward maximization
  $\pi^*(y|x) = \frac{1}{Z(x)}\pi_{\mathrm{ref}}(y|x)\exp\big(\frac{1}{\beta}r(x,y)\big)$
- Eq.(5) (reward reparameterization):$r(x,y) = \beta\log\frac{\pi(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta\log Z(x)$
- Eq.(6):Bradley–Terry preference model $p(y_w \succ y_l) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$
- Eq.(7) (DPO loss):$L_{\mathrm{DPO}} = -\mathbb E_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\Big)\Big]$

## Claims list

- **P1** (direct corollary of Eq.7):at $\pi_\theta = \pi_{\mathrm{ref}}$, $z=0$ and the loss
  is exactly $\log 2$ (the first smoke test of any DPO implementation).
- **P2** (σ monotonicity + pairwise difference):the loss is strictly decreasing in $z$;
  swapping $y_w/y_l$ flips the sign of $z$ and the identity
  $L(y_l,y_w) - L(y_w,y_l) = z$ holds (exact).
- **P3** (Eq.5 round trip):given $r$ and $\pi_{\mathrm{ref}}$, construct $\pi^*$ per Eq.(4),
  invert per Eq.(5), and $r$ is recovered exactly (the $\beta\log Z$ term cannot be dropped).
- **P4** (Eq.4 normalization):$\sum_y \pi^*(y|x) = 1$ (by construction of $Z$).
- **P5**:the analytic gradient $\partial L/\partial(\log p)$ has the structure
  $\mp\beta\,\sigma(-z)$ across four paths, matching finite differences.
- **Not testable** (→ GAP_LIST):the expectation over preference data, comparison against
  the RLHF baseline, β sensitivity — all need real data and training.
