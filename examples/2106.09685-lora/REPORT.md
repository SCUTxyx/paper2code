# Reproduction report: LoRA (arXiv:2106.09685, §4.1-4.2 adaptation math)

## Scope statement

Only the low-rank adaptation math: ΔW = BA (Eq.4), the B=0 / A-Gaussian initialization
and its gradient asymmetry, the α/r scaling, and merge/unmerge equivalence. Task-quality
tables, multi-adapter serving, and quantized-base variants (QLoRA) are out of scope
(GAP_LIST).

## Results matrix

| Test file | Cases | Pass | Fail | Notes |
|---|---|---|---|---|
| test_properties.py | 4 | 4 | 0 | L1 init exactness (bit-for-bit); L2 gradient asymmetry with EXACT zeros (+swapped-init control); L3 rank spectrum truncated at r; L6 both-zero fixed point demonstrated over 20 SGD steps |
| test_anchor.py | 4 | 4 | 0 | integer hand case h=[6.5, 9.0], α/r=2 variant, merge/unmerge exact; r=2 scale discriminator (α/r vs α·r) |
| test_gradients.py | 3 | 3 | 0 | outer-product grads rtol 1e-6 (ΔW path), 5e-4 (streaming path — small-component floor, see finding 3); grad_A ≡ 0 at init matched by central differences |
| test_crosscheck.py | 2 | 2 | 0 | ΔW-formed vs streaming forward 1e-12; merged weight produces identical outputs; unmerge exact |

Run: `python -m pytest examples/2106.09685-lora/tests -q` (float64, CPU, < 1 s).

## Findings

1. **The gradient asymmetry at init is structural, not approximate.** With B=0 the loss
   is *independent of A entirely*, so ∂L/∂A ≡ 0 in every component — central differences
   return exact zeros, matching the analytic outer product. Training starts by moving B
   only. Swapping the zeros (A=0, B random) also trains (A moves first); what freezes
   training permanently is zeroing BOTH factors — demonstrated as a fixed point over 20
   SGD steps (L6). This is the highest-frequency silent LoRA bug, and it is exactly
   testable.
2. **ΔW = BA has rank exactly r** (for full-rank factors): the singular-value spectrum
   of the formed update is truncated to r entries above the float noise floor — a cheap,
   direct check that an implementation hasn't silently lost the low-rank structure.
3. **Finite-difference floor, encountered again**: a streaming-path gradient component
   of O(1e-7) carries ~1e-4 relative central-difference noise (|f|·eps/(2h·|g|)), so that
   check uses rtol 5e-4 with the derivation in the test docstring — the second repro in
   this repo to hit the floor documented in tests/test_gradcheck.py. The two forward
   paths' exact equivalence is established separately at 1e-12.

## Known limitations

- No training runs: convergence, task quality, and rank-choice ablations (paper §5
  tables) are empirical and out of scope.
- Only linear-loss gradients; the composition with dropout/scaling in real trainers is
  not modeled.
- fp16/bf16 and quantized-base (QLoRA) behavior untested.
