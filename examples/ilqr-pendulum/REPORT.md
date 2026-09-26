# Reproduction report: iLQR on a torque-limited pendulum

## FEASIBILITY verdict

**✅ REPRODUCIBLE-MATH** — model-based control is the embodied class whose full
contribution is verifiable on CPU: dynamics, derivatives, optimizer, and the resulting
trajectory are all checked against implementation-independent oracles (finite
differences, the 1-step LQR closed form, KKT sign conditions). Hardware-in-the-loop and
sim-to-real behavior are out of reach (GAP_LIST).

## Scope statement

The iLQR algorithm itself (linearization, Bellman backward pass, line-searched forward
pass, box clamping) on an inverted-pendulum swing-up. No robot, no learned dynamics,
no MPC deployment loop.

## Results matrix

| Test file | Cases | Pass | Fail | Notes |
|---|---|---|---|---|
| test_properties.py | 5 | 5 | 0 | cost monotone; box exact; clean-interior stationarity (< 1e-3); clamp residual documented & bounded; robustness over 3 inits |
| test_anchor.py | 2 | 2 | 0 | 1-step LQR closed form (u* = −0.78, 1e-9); swing-up reach (θ_T = 0.095 rad, KKT < 1e-6) |
| test_gradients.py | 2 | 2 | 0 | analytic Jacobians & exact discrete adjoint vs central differences (< 1e-6) |
| test_crosscheck.py | 2 | 2 | 0 | analytic vs FD Jacobians (1e-6); analytic vs FD-based solver on full swing-up (cost rtol 1e-5) |

Run: `python -m pytest examples/ilqr-pendulum/tests -q` (float64, CPU, ~3 s).

## Findings

1. **The 1-step LQR closed form is a zero-dependency oracle for the entire backward
   pass.** On a 1-step linear-quadratic problem iLQR must return
   $u^* = -(abq x_0)/(r + b^2 q)$ exactly — it does, to ~1e-12 (asserted at 1e-9 with
   the decaying regularizer documented). This anchors the whole optimizer with a
   hand-derivable constant.
2. **The naive forward-clamp variant leaves O(1) true-gradient residuals at saturated
   controls — measured, not assumed.** With binding torque limits (u_max = 3), the solver
   terminates satisfying its internal criterion (Q_u-based KKT under the box, < 1e-6)
   while the exact adjoint gradient of the *clamped rollout* keeps residuals up to ~0.3
   at saturated controls and ~0.7 at interior controls adjacent to saturated spans.
   The mechanism: the backward pass ignores the clip's zero-derivatives, so its model of
   downstream propagation diverges from the clamped reality. Practitioner guidance: the
   clamp variant is a good fast approximation, but tight constrained optimality needs
   Tassa et al.'s box-aware QP backward pass. Both facts are pinned by tests
   (interior stationarity strict, clamp residual bounded) instead of hidden.
3. **Convergence criteria must be constraint-aware.** Using ‖Q_u‖∞ as the stop condition
   never terminates under binding constraints (saturated controls legitimately have
   Q_u ≠ 0); using "line-search stalled" terminates at non-stationary points. The correct
   measure is the projected KKT violation (interior |Q_u|, clamped signed) — implemented
   in the solver and reused by the tests.
4. A subtle numpy-2 shape trap hit during development: `B.T @ v` on a 1-D B does not
   transpose, silently producing scalar-typed Bellman terms — the column-vector
   convention (B shaped (n,1)) fixed it and is pinned by the tests.

## Known limitations

- Pendulum only; no real hardware, no actuator/sensor models, no MPC receding-horizon
  deployment loop (the plan view is deterministic open-loop optimization).
- float64; no runtime claims (the paper's contribution is optimization quality, not
  wall-clock here).
- The box handling is the clamp variant; a box-aware QP backward pass would remove the
  documented residual (out of scope for this repro, stated in GAP_LIST).
