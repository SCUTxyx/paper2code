# Reproduction report: FlashAttention tiled softmax (arXiv:2205.14135)

## Scope statement

Only the tiled (online) softmax math piece — Algorithm 1's forward recurrence, its
exactness, and its numerical-stability properties. FlashAttention is a *systems* paper:
the IO-complexity analysis and kernel-level claims are out of reach on CPU (applicability
precheck, references/verification.md §4) and are honestly listed in GAP_LIST.

## Results matrix

| Test file | Cases | Pass | Fail | Notes |
|---|---|---|---|---|
| test_properties.py | 4 | 4 | 0 | shift invariance (1e-12); overflow impossibility; partition invariance over a (bq,bk) grid; causal zero-contribution |
| test_anchor.py | 3 | 3 | 0 | hand constants 1+2σ(2) (growing max, rescale exercised) and 3σ(2)−1; degenerate single-tile case |
| test_gradients.py | 2 | 2 | 0 | gradients through both tiled forms == analytic direct grads, rtol 1e-6, causal included |
| test_crosscheck.py | 3 | 3 | 0 | tiled == direct to 1e-12 across shapes/causal; both tiled forms agree; rectangular cross-attention |

Run: `python -m pytest examples/2205.14135-flashattention/tests -q` (float64, CPU, < 1 s).

## Findings

1. **The `l_old` factor is *the* FlashAttention implementation trap — and our first
   attempt walked straight into it.** The printed Algorithm 1 keeps the output
   accumulator unnormalized and divides by `l` once at the end. A natural
   "normalized-at-every-step" transliteration must rescale the previous output by
   `e^{Δm} · l_old` — rescaling by `e^{Δm}` alone mixes a normalized quantity with an
   unnormalized one. Our first `core_eq` did exactly that; the F1 exactness test
   (tiled == direct, 1e-12) caught it immediately, the gradient check would also have
   caught it (errors ~O(1), not ~1e-6). This is a *self-consistent* error class:
   a weaker suite that only checks shapes or soft properties would have passed it.
2. **Tiling is an algebraic identity, not an approximation** — exact to 1e-12 across
   every block partition tested, including b=1, rectangular shapes, and causal masks.
   This is the paper's "exact attention" claim, confirmed at the math level.
3. **Numerical stability is structural, not incidental**: the running-max recurrence is
   softmax shift invariance applied per tile, so the tiled path can never exponentiate
   a positive overflow. Demonstrated: naive softmax produces NaN at logits ~±10³ where
   the tiled path is exact. The `−∞`-initialized first block (rescale factor exactly 0,
   not NaN) is the second classic off-by-init bug — pinned by an anchor test.
4. **Gradients through the tiled path equal direct-attention gradients** (rtol 1e-6,
   both tiled forms, causal included) — verifying that the tiling changes the
   computation schedule, not the differentiable function.

## Known limitations

- No backward-pass implementation (FlashAttention's tiled backward with recomputation
  is its own algorithm); gradients here are checked *through* the tiled forward.
- float64 only; fp16/bf16 accumulation order effects untested (GAP_LIST).
- No IO model, no kernels, no wall-clock measurements — deliberately (scope statement).
