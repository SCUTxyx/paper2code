# Method card: RMSNorm (Zhang & Sennrich 2019)

- Paper: *Root Mean Square Layer Normalization*, arXiv:1910.07467 (NeurIPS 2019)
- Scope statement: only the §3 RMSNorm formulation (Eq.4 with the common ε stabilization)
  and its invariance properties; the LLT kernel-fusion analysis, pRMSNorm, and all
  benchmarks are out of scope.

## Problem formalization

Replace LayerNorm's mean/variance centering with a pure root-mean-square scaling:
$$y = \frac{x}{\mathrm{RMS}(x)} \odot g,\qquad \mathrm{RMS}(x) = \sqrt{\tfrac{1}{d}\textstyle\sum_i x_i^2 + \varepsilon}$$

## Symbol table

| Symbol | Meaning | Source |
|---|---|---|
| $x \in \mathbb{R}^d$ | one token's features | §3 |
| $g \in \mathbb{R}^d$ | learned per-feature gain | Eq.(4) |
| $\varepsilon$ | stabilization inside the sqrt (implementation convention) | §3/practice |
| $\mathrm{RMS}(x)$ | root mean square | Eq.(3) |

## Core formula (Eq.4 + ε convention)

- $y_i = g_i\, x_i \,/\, \sqrt{\tfrac{1}{d}\sum_j x_j^2 + \varepsilon}$

## Claims list

- **C1** (definition): with $g=\mathbf 1$, the output has mean-square exactly
  $m/(m+\varepsilon)$ where $m = \tfrac1d\sum x_j^2$ — i.e. RMS 1 up to the analytic
  $\varepsilon/(m+\varepsilon)$ deficit (a bound, testable exactly).
- **C2** (scale invariance, §3's motivation vs LayerNorm): at $\varepsilon = 0$,
  $\mathrm{RMSNorm}(c\,x) = \mathrm{RMSNorm}(x)$ for $c>0$ (exact) and $= -\mathrm{RMSNorm}(x)$
  for $c<0$; with $\varepsilon>0$ the invariance is approximate with an analytically
  bounded deviation (the ε breaks exact homogeneity — worth pinning).
- **C3** (contrast with LayerNorm): RMSNorm is **not** shift-invariant —
  $\mathrm{RMSNorm}(x + c\mathbf 1) \ne \mathrm{RMSNorm}(x)$ (negative property, documents
  the design difference; LayerNorm's mean-centering is what buys shift invariance).
- **C4** (gradient): for $L = w^\top y$,
  $\partial L/\partial x_k = w_k g_k / r - x_k \sum_i w_i g_i x_i / (d\, r^3)$ with
  $r = \sqrt{m + \varepsilon}$ (standard RMSNorm backward, central-difference checkable).
- **C5** (ε placement discriminator): at large ε the three plausible readings of the
  formula differ hugely; a d=1 hand case separates them exactly.
- **Not testable** (→ GAP_LIST): kernel fusion / wall-clock claims (§4.1.1), benchmark
  tables, fp16 behavior.
