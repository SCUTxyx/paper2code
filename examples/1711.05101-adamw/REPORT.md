# Reproduction report: AdamW (arXiv:1711.05101, Algorithm 2)

## Scope statement

Only the Algorithm 2 update rule and its decoupling property (schedule multiplier η_t
fixed to 1). The paper's generalization claims, the η_t schedule interaction, and the
hyperparameter-transfer grid are empirical matters → GAP_LIST.

## Results matrix

| Test file | Cases | Pass | Fail | Notes |
|---|---|---|---|---|
| test_properties.py | 2 | 2 | 0 | W1: moment trajectories identical for all λ (atol=0); gap follows the closed-form linear recursion (1e-12) |
| test_anchor.py | 3 | 3 | 0 | two hand-computed steps with decay (θ₂=0.7845818); g=0 geometric shrink; ε=1 discriminating anchor (wrong ε placements differ 8-26×) |
| test_gradients.py | 1 | 1 | 0 | ∂θ_T/∂θ_0 = (1−lr·λ)^T exact, closed form vs central differences, two λ values |
| test_crosscheck.py | 2 | 2 | 0 | fused == decoupled composition (1e-12); λ=0 == hand-rolled Adam (1e-12) |

Run: `python -m pytest examples/1711.05101-adamw/tests -q` (float64, CPU, < 1 s).

## Findings

1. **The adaptive step is gradient-only — the technical heart of decoupling.** Because
   Adam's preconditioner m̂/(√v̂+ε) depends solely on the gradient history, the two runs
   that differ only in λ share identical moment trajectories (verified with atol=0), and
   the parameter gap obeys the linear recursion d_t = (1−lr·λ)d_{t-1} − lr·λ·θ_{t-1}^{base}.
   A coupled implementation (L2 folded into g) violates both — its moments would shrink
   with λ. This gives a property test that *discriminates* the two designs, which loss
   curves cannot do.
2. **The gradient w.r.t. g is numerically meaningless as a test target**: the update's
   sensitivity to g is O(ε) (= lr·ε/(|g|+ε)²), far below the finite-difference noise
   floor. We deliberately test only the decay-path gradient ∂θ_t/∂θ_0 = (1−lr·λ)^t,
   which is O(1) and exact. Choosing *which* partial derivative to check is part of
   honest testability planning.
3. The composition view (Adam step + separate shrink) is not just an implementation
   convenience: it is the mathematical content of "decoupled", and the two formulations
   agree bit-for-bit here.

## Known limitations

- No η_t schedule; no comparison of generalization vs Adam (needs training).
- The infamous ICLR-2019 reviews' disagreement about when decoupling matters is an
  empirical question outside this repo's scope.
