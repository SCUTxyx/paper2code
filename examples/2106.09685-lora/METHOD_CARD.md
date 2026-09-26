# Method card: LoRA low-rank adaptation (Hu et al. 2021)

- Paper: *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685 (ICLR 2022)
- Scope statement: only the adaptation math (§4.1-4.2): ΔW = BA, the B=0 / A~Gaussian
  initialization and its gradient asymmetry, the α/r scaling, and merge/unmerge
  equivalence. Full fine-tuning comparisons, task tables, multi-adapter serving are
  not covered.

## Problem formalization

Freeze a pretrained weight $W_0 \in \mathbb{R}^{d\times k}$ and learn a low-rank update
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\, B A\, x,\qquad
B \in \mathbb{R}^{d\times r},\ A \in \mathbb{R}^{r\times k}$$
with $r \ll \min(d, k)$; trainable parameters drop from $d \times k$ to $r(d+k)$.

## Symbol table

| Symbol | Meaning | Source |
|---|---|---|
| $W_0$ | frozen pretrained weight | §4.1 |
| $A, B$ | low-rank factors | §4.1 |
| $r$ | rank | §4.1 |
| $\alpha/r$ | scaling of $\Delta W$ | §4.1 |
| $\Delta W = BA$ | the learned update | §4.1 Eq.(4) |

## Core formulas (§4.1)

- Eq.(4):$h = W_0 x + \frac{\alpha}{r} B A x$
- Initialization (§4.1):$A \sim \mathcal N(0, \sigma^2)$ (Gaussian), $B = 0$ — so
  $\Delta W = 0$ at the start of training.
- Merged inference:$W = W_0 + \frac{\alpha}{r} B A$ (§4.4 discusses deployment; the
  identity is mathematical).

## Claims list

- **L1** (init semantics): at initialization $h = W_0 x$ **exactly** — the adapter
  contributes zero, bit-for-bit.
- **L2** (gradient asymmetry at init): with $B = 0$, the loss is *independent of A
  entirely*, so $\partial L/\partial A \equiv 0$ (exactly zero, every component) while
  $\partial L/\partial B = \frac{\alpha}{r}\,w\,(Ax)^\top \neq 0$ — training starts by
  moving B only. (This asymmetry is the design; swapping the zeros also works, but
  zeroing BOTH factors is a training fixed point — see finding 2.)
- **L3** (rank constraint): $\mathrm{rank}(BA) \le r$; with full-rank factors it equals
  $r$ — the singular-value spectrum of $\Delta W$ has exactly r nonzero entries.
- **L4** (merge equivalence): $W_0 + \frac{\alpha}{r}BA$ applied to x gives the same
  output as the two-stage path $W_0 x + \frac{\alpha}{r}B(Ax)$ (atol 1e-12; float
  associativity differs in op order only).
- **L5** (gradients): $\partial L/\partial B = \frac{\alpha}{r}\,w\,(Ax)^\top$ and
  $\partial L/\partial A = \frac{\alpha}{r}\,B^\top w\, x^\top$ for a linear loss
  $L = w^\top h$ — outer products, central-difference checkable.
- **L6** (both-zero fixed point): if $A = 0$ AND $B = 0$, both gradients are exactly
  zero and plain SGD never moves the adapter — a training fixed point (the classic
  silent LoRA bug; demonstrated, not just asserted).
