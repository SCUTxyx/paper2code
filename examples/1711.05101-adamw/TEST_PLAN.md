# Testability plan: AdamW

| Claim | Content | Test type | Location | Notes |
|---|---|---|---|---|
| W1 | Moments independent of λ; decay = lr·λ·θ_{t-1} exactly | property-invariance | test_properties.py::test_moments_independent_of_decay | THE decoupling test: coupled Adam-with-L2 fails it |
| W2 | λ=0 == Adam exactly | cross-check (degeneracy) | test_crosscheck.py::test_lambda_zero_is_adam | inline hand-rolled Adam reference |
| W3 | g=0 → θ_t = θ_0(1−lr·λ)^t exact | anchor | test_anchor.py::test_zero_gradient_shrink | hand constants |
| W4 | ∂θ_t/∂θ_0 = (1−lr·λ)^t | gradient check | test_gradients.py | closed form incl. nonzero constant g |
| W5 | fused update == Adam step + separate shrink | cross-check | test_crosscheck.py::test_fused_equals_decoupled_composition | dual formulations |
| — | two-step hand computation with decay | anchor | test_anchor.py::test_two_steps_hand_computed | derivation in comments |
| Generalization / η_t schedule | empirical claims | **not automatable** | → GAP_LIST #1-2 | needs training |
