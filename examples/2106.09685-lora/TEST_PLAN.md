# Testability plan: LoRA

| Claim | Content | Test type | Location | Notes |
|---|---|---|---|---|
| L1 | h = W₀x exactly at init | anchor/property | test_properties.py::test_init_output_equals_base | bit-for-bit |
| L2 | grad_A ≡ 0 (exact), grad_B ≠ 0 at init | property + gradient | test_properties.py::test_gradient_asymmetry_at_init + test_gradients.py | exact zeros matter |
| L3 | rank(ΔW) = r, spectrum truncated | property | test_properties.py::test_rank_constraint | SVD spectrum check |
| L4 | merge == two-stage | cross-check | test_crosscheck.py::test_merge_equivalence | 1e-12 |
| L5 | outer-product gradients | gradient check | test_gradients.py | incl. through two-stage path |
| L6 | both-zero init is a training fixed point | property (demonstrated) | test_properties.py::test_both_zero_init_is_fixed_point | the classic bug |
| — | d=2, r=1 hand case | anchor | test_anchor.py | hand integers |
| Task tables / serving | empirical claims | **not automatable** | → GAP_LIST #1-2 | needs training / infra |
