# Testability plan: DPO

| Claim | Content | Test type | Location | Notes |
|---|---|---|---|---|
| P1 | π=π_ref gives L = log 2 (exact) | property-degeneracy | test_properties.py::test_reference_policy_gives_log2 | |
| P2 | strictly decreasing in z; swap identity L(l,w)−L(w,l)=z | property-monotone/invariance | test_properties.py::test_monotone_decreasing_in_z + ::test_swap_identity | swap identity at 1e-12 |
| P3 | Eq.(5) reward reparameterization round trip | cross-check (closed-form identity) | test_crosscheck.py::test_reward_roundtrip | where the β·logZ term matters |
| P4 | π* normalization | property-invariance | test_crosscheck.py::test_optimal_policy_normalized | |
| P5 | analytic gradient | gradient check | test_gradients.py | four log-prob inputs |
| — | z=ln3 → L=ln(4/3);z=0 → log 2;z=−ln3 → ln 4 | anchor | test_anchor.py | exact hand constants, derivation in comments |
| — | Eq.(7) literal vs Eq.(6) Bradley–Terry path | cross-check | test_crosscheck.py::test_eq7_vs_bradley_terry | dual formulations |
| Data expectation | −E[·] over the preference distribution | **not automatable** | → GAP_LIST #1 | needs real preference data |
