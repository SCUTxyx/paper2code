# Reproduction report: RoPE (arXiv:2104.09864, §3.2 rotary position embedding)

## Scope statement

Only the §3.2 rotary position embedding math piece (formulas + two equivalent
formulations). The overall RoFormer architecture, long-context extrapolation
experiments, and the decay upper bound's asymptotic proof are not covered (GAP_LIST).

## Results matrix

| Test file | Cases | Pass | Fail | Notes |
|---|---|---|---|---|
| test_properties.py | 4 | 4 | 0 | R1 orthogonality / R2 relative-position invariance / R4 composition / m=0 identity |
| test_anchor.py | 4 | 4 | 0 | θ sequence, d=2 hand case [cos3, sin3], score = sin2, d=4 dual-frequency |
| test_gradients.py | 3 | 3 | 0 | max gradient rel error < 1e-10 (rotate / score / matrix path) |
| test_crosscheck.py | 2 | 2 | 0 | slicing vs explicit matrix vs complex form, 1e-12 (d=2..16, m∈{0,1,7,128,−5}) |

Run: `python -m pytest examples/2104.09864-rope/tests -q` (float64, CPU, < 1 s).

## Findings

1. **Off-by-one in the θ indexing is the top real-implementation risk.** The paper's
   Eq.(15) is 1-based ($\theta_i = 10000^{-2(i-1)/d}$, $i = 1..d/2$); mainstream open-source
   code uses 0-based `inv_freq = base^(-2i/d)`. The two agree — but shifting a dimension by
   one or mispairing blocks yields a *self-consistent* implementation whose property tests
   (relative-position invariance) still pass. Only the EQ_MAP human review catches that
   class. Both index conventions are written down here and cross-checked.
2. **Three formulations agree at 1e-12**: block slicing, explicit block-diagonal matrix,
   and complex multiplication — the paper's §3.2.1 ↔ §3.2.2 equivalence confirmed
   bit-for-bit.
3. **Positions extend to any integer (including negative)** because the group structure
   $R^mR^n = R^{m+n}$ relies only on integer addition; this does NOT constitute a claim
   about long-context generalization (that is an empirical matter).

## Known limitations

- Only the math piece is verified; decay-style asymptotic claims and model quality are
  entirely unverified (GAP_LIST).
- float64 and an even-dimension assumption; odd $d$ and half precision (fp16/bf16)
  behavior untested.
- Later variants (NTK-aware scaling, YaRN, etc.) not covered — only the original
  Eq.(15) with base 10000.
