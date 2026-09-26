# Method Card: FlashAttention tiled softmax (Dao et al. 2022)

- Paper: *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv:2205.14135
- Scope statement: only the **tiled (online) softmax** math piece — Algorithm 1's forward recurrence
  and its exactness/stability properties (the online-softmax trick originates in Milakov & Gimelshein,
  arXiv:1805.02867). IO-complexity analysis, SRAM/HBM tiling sizes, kernel implementation and
  wall-clock speedups are systems claims → GAP_LIST (see applicability precheck in
  references/verification.md §4: this is a *systems* paper; only the math core is testable on CPU).

## Problem formalization

Compute exact attention $\mathrm{softmax}(QK^\top/\sqrt{d})\,V$ **without ever materializing the
$n\times n$ score matrix**, by streaming over $K/V$ blocks and maintaining per-query running
statistics $(m, l, O)$: running max, running normalizer, running (rescaled) output.

## Symbol table

| Symbol | Meaning | Source |
|---|---|---|
| $Q, K, V$ | query / key / value matrices ($n_q \times d$, $n_k \times d$, $n_k \times d_v$) | §3.1 |
| $S^{(j)}$ | score block $Q_i K_j^\top / \sqrt{d}$ | Algorithm 1 |
| $m_i$ | running row-max of logits for query block $i$ | Algorithm 1 |
| $l_i$ | running row-sum of exponentials (normalizer) | Algorithm 1 |
| $\tilde P$ | $\exp(S - m^{\mathrm{new}})$, the locally re-scaled probability block | Algorithm 1 |
| $O_i$ | running output accumulator for query block $i$ | Algorithm 1 |

## Core formulas (§3.1, Algorithm 1 forward)

- Attention definition (Vaswani 2017 Eq.1, the *target* being tiled):
  $\mathrm{Attn}(Q,K,V) = \mathrm{softmax}(QK^\top/\sqrt{d})\,V$
- Per-tile update (Algorithm 1):
  $m^{\mathrm{new}} = \max(m^{\mathrm{old}},\ \mathrm{rowmax}(S))$
  $\tilde P = \exp(S - m^{\mathrm{new}})$
  $\ell^{\mathrm{new}} = e^{\,m^{\mathrm{old}}-m^{\mathrm{new}}}\,\ell^{\mathrm{old}} + \mathrm{rowsum}(\tilde P)$
  $O^{\mathrm{new}} = \mathrm{diag}(\ell^{\mathrm{new}})^{-1}\big(\mathrm{diag}(e^{\,m^{\mathrm{old}}-m^{\mathrm{new}}})\,O^{\mathrm{old}} + \tilde P^\top V^{(j)}\big)$
- Final: $O = \mathrm{diag}(\ell)^{-1} O$ where the accumulation is kept unnormalized
  (streaming variant; both forms are exact).

## Algorithm box (transcription of Algorithm 1, forward)

1. Initialize per-query-row $m = -\infty$, $\ell = 0$, $O = 0$.
2. For each $K/V$ block $j$: load $K^{(j)}, V^{(j)}$; for each $Q$ block $i$:
   compute score block $S = Q_i K_j^\top / \sqrt d$; apply causal mask if needed;
   update $(m, \ell, O)$ with the recurrence above.
3. Return $O$.

## Claims list

- **F1** (Algorithm 1, "exact attention"): the tiled computation returns *exactly* the same
  output as direct softmax attention — tiling is an algebraic identity, not an approximation
  (verified to 1e-12 across block-size grids, causal and not).
- **F2** (§3.1 stability device, softmax shift invariance): $\mathrm{softmax}(S) =
  \mathrm{softmax}(S + c\mathbf 1)$; the running-max recurrence is this identity applied
  per tile. Consequence: the tiled path can never exponentiate a positive overflow —
  naive softmax overflows at $|{\rm logit}| \gtrsim 709$ in float64, the tiled path does not.
- **F3** (Algorithm 1 rescale step): the update is consistent iff previous partial results are
  rescaled by $e^{m^{\rm old}-m^{\rm new}}$ whenever the running max increases; initial state
  $m=-\infty$ must make the rescale factor exactly $0$ (not NaN).
- **F4** (corollary of F1): the output is invariant to the block partition — any $(b_q, b_k)$
  grid (including $b{=}1$) yields identical results.
- **F5** (§3.2): causal masking composes with tiling — masked blocks contribute exactly 0.
- **F6** (corollary of F1): gradients through the tiled forward equal the analytic gradients
  of direct attention (already finite-difference-verified in calibration/02_attention).
- **Not testable** (→ GAP_LIST): IO-complexity and wall-clock speedup (needs GPU + HBM model),
  SRAM tile sizes, fp16/bf16 kernel behavior, backward-pass IO.
