# Reproduction report: RMSNorm (arXiv:1910.07467, §3)

## Scope statement

Only the §3 formulation (Eq.4 with the ε-inside-sqrt convention) and its invariance
properties. Kernel fusion (§4.1.1), pRMSNorm, benchmarks, and fp16 behavior are
out of scope (GAP_LIST). This folder is a **dry run of SKILL.md itself** — produced
by following the six-stage pipeline literally, into the user run area (repros/).

## Results matrix

| Test file | Cases | Pass | Fail | Notes |
|---|---|---|---|---|
| test_properties.py | 4 | 4 | 0 | C1 exact ε-deficit bound; C2 scale homogeneity exact at ε=0, closed-form at ε>0; C3 negative (shift) property; broadcasting |
| test_anchor.py | 2 | 2 | 0 | d=2 hand case [3√2/5, 4√2/5]; ε=1 discriminator (2/√5 vs 2/3 vs 2) |
| test_gradients.py | 2 | 2 | 0 | closed-form backward < 1e-6, incl. small-x edge (see finding 3) |
| test_crosscheck.py | 2 | 2 | 0 | vectorized vs per-token loop, 1e-12, three ε values |

Run: `python -m pytest repros/1910.07467-rmsnorm/tests -q` (float64, CPU, < 1 s).

## Findings

1. **Scale "invariance" is exact only at ε=0** — with ε>0, RMSNorm(cx) ≠ RMSNorm(x)
   by a small closed-form amount (the stabilization breaks homogeneity). Implementations
   and papers usually gloss over this; here it is pinned with the exact closed form
   (test compares against the definition, not against another code path).
2. **The ε-placement trap generalizes** (Adam's ε, FlashAttention's scaling, now RMSNorm's):
   d=1 with ε=1 separates the three readings of the formula by 2.3-3× per unit — one
   assertion kills the whole class. Added to the anchor suite.
3. **Checker-vs-code error, promoted to the spec**: at input scale 1e-3, RMSNorm's
   third derivative ~1/r³ makes central-difference truncation breach rtol — an artifact
   of the CHECKER, not the implementation. Now verification.md §2.6: when testing
   ε-stabilized norms at small scales, document the truncation estimate or loosen the
   tolerance with derivation.
4. **Negative property with design content**: RMSNorm is deliberately NOT
   shift-invariant (that is what dropping mean-centering buys in speed); the test
   asserts the asymmetry rather than pretending all invariances hold.

## Known limitations

- No kernel/benchmark verification (GAP_LIST); float64 only; gain-before/after-norm
  conventions vary across real codebases — EQ_MAP pins this reproduction's convention.
