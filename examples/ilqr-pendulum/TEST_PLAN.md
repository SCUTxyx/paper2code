# Testability plan: iLQR pendulum

| Claim | Content | Test type | Location | Notes |
|---|---|---|---|---|
| I-1 | Analytic Jacobians == central differences | gradient check | test_gradients.py::test_dynamics_jacobians | classic bug class |
| I-2 | Backward-pass Q_u == rolled-out cost gradient | gradient check | test_gradients.py::test_backward_qu_vs_rollout | first-order, rtol 1e-3 justified |
| I-3 | Cost monotone; convergence | property | test_properties.py::test_cost_monotone | |
| I-4 | KKT stationarity under box constraints | property | test_properties.py::test_constrained_stationarity | interior ≈0 + sign condition at bounds |
| I-5 | 1-step LQR closed form | anchor | test_anchor.py::test_one_step_lqr_closed_form | exact 1e-12 |
| I-6 | Swing-up reach, box respected | anchor + property | test_anchor.py::test_swing_up + test_properties.py::test_box_respected | deterministic config |
| — | Analytic vs FD jacobians; eq vs pseudo full runs | cross-check | test_crosscheck.py | |
| Hardware / MPC deployment | OUT | **not automatable** | → GAP_LIST | no robot |
