# Testability plan: RMSNorm

| Claim | Content | Test type | Location | Notes |
|---|---|---|---|---|
| C1 | mean(y²) = m/(m+ε), deficit ≤ ε/m (analytic bound) | property | test_properties.py::test_output_rms | exact bound, not statistical |
| C2 | exact scale invariance at ε=0; bounded at ε>0 | property | test_properties.py::test_scale_invariance | ε=0 exact 1e-12; ε>0 analytic bound |
| C3 | NOT shift-invariant (vs LayerNorm) | negative property | test_properties.py::test_not_shift_invariant | asserts difference > 0 |
| C4 | closed-form backward | gradient check | test_gradients.py | rtol 1e-6 |
| C5 | ε=1 discriminator (d=1) | anchor | test_anchor.py::test_epsilon_placement | 2/√5 vs 2/3 vs 2 |
| — | d=2 hand case: x=[3,4] → [3√2/5, 4√2/5] | anchor | test_anchor.py | exact constants |
| — | vectorized vs per-token loop | cross-check | test_crosscheck.py | |
| Kernel / benchmarks | systems claims | **not automatable** | → GAP_LIST #1-2 | needs GPU |
