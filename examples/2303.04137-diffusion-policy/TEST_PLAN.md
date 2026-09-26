# Testability plan: Diffusion Policy math core

| Claim | Content | Test type | Location | Notes |
|---|---|---|---|---|
| P-1 | Closed-form marginal == iterative noising (same ε, exact) | cross-check | test_crosscheck.py::test_closed_form_vs_iterative | re-derived identity, mirrors calibration/04 |
| P-2 | DDIM chain: perfect ε̂ ⇒ exact trajectory invariance; biased ε̂ ⇒ deviation | anchor + negative property | test_anchor.py::test_ddim_chain_recovers_x0 + test_properties.py::test_ddim_biased_prediction_deviates | the marquee inference-path test |
| P-3 | Receding-horizon stitching; window property | anchor + property | test_anchor.py::test_receding_horizon_stitching + test_properties.py::test_window_property | hand-literal stitching case |
| P-4 | Loss gradient; forward reparameterization gradient | gradient check | test_gradients.py | |
| — | Real-robot success rates, encoder, training | **not automatable** | → GAP_LIST #1-3 | feasibility 🟡 PARTIAL |
