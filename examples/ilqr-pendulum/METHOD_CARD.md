# Method card: iLQR — iterative Linear Quadratic Regulator on a torque-limited pendulum

- Paper-provenance: Li & Todorov, *Iterative Linear Quadratic Regulator Design for
  Nonlinear Biological Movement Systems* (2004); Tassa, Erez & Todorov,
  *Control-Limited Differential Dynamic Programming* (ICRA 2012) for the box-constraint
  handling. Classical model-based control lineage rather than a single arXiv paper.
- Scope statement: the iLQR algorithm itself (dynamics linearization, Bellman backward
  pass, line-searched forward pass, box clamping) verified on an inverted pendulum
  swing-up. Real hardware, MPC receding-horizon deployment, and learned dynamics are
  out of scope.

## FEASIBILITY (embodied precheck, references/embodied_precheck.md)

- Verdict: **✅ REPRODUCIBLE-MATH** — model-based control is exactly the class where the
  full contribution is verifiable on CPU: dynamics, derivatives, optimizer, and the
  resulting trajectory are all numpy-testable against implementation-independent oracles
  (finite differences, the 1-step LQR closed form, KKT sign conditions).
- Not verifiable here: hardware-in-the-loop behavior, actuator dynamics, sim-to-real gap.
- What is missing: a robot.

## Problem formalization

$$\min_{u_{0:T-1}} \sum_{t=0}^{T-1}\Big[\tfrac12 (x_t - x_g)^\top Q (x_t - x_g) + \tfrac12 r\,u_t^2\Big] + \tfrac12 (x_T - x_g)^\top Q_f (x_T - x_g)$$
s.t. $x_{t+1} = f(x_t, u_t)$ (pendulum, Euler), $u_t \in [-u_{\max}, u_{\max}]$.

## Symbol table

| Symbol | Meaning | Source |
|---|---|---|
| $x = (\theta, \dot\theta)$ | pole angle from upright, angular velocity | §pendulum model |
| $u$ | torque, box-limited | Tassa et al. 2012 |
| $A_t, B_t$ | $\partial f/\partial x, \partial f/\partial u$ at the nominal | iLQR §linearization |
| $Q_{uu}, Q_{ux}, Q_u$ | action-value expansions around the nominal | iLQR §backward pass |
| $k, K$ | feedforward / feedback gains | iLQR §backward pass |

## Core formulas

- Dynamics (Euler, dt):$\theta' = \theta + dt\,\dot\theta$;$\dot\theta' = \dot\theta + dt\,(-g\sin\theta - b\,\dot\theta + u)/I$
- Analytic Jacobians:$A = \begin{pmatrix}1 & dt\\ -dt\,g\cos\theta & 1 - dt\,b/I\end{pmatrix}$,$B = \binom{0}{dt/I}$
- Backward pass (Bellman, first order in the deviations):$Q_u = l_u + B^\top V_x$,$Q_{uu} = l_{uu} + B^\top V_{xx} B + \mu I$,$Q_{ux} = B^\top V_{xx} A$,
  $k = -Q_{uu}^{-1} Q_u$,$K = -Q_{uu}^{-1} Q_{ux}$;$V_x = l_x + A^\top V_x' - A^\top Q_{uu}^{-1} Q_u$… (standard recursion)
- Forward pass:$u_t = \mathrm{clip}\big(\hat u_t + \alpha\,k_t + K_t (x_t - \hat x_t)\big)$ with backtracking line search on $\alpha$.
- 1-step LQR closed form (anchor oracle):$\min \tfrac12 q\,x_f^2 + \tfrac12 r\,u^2$ s.t. $x_f = a x + b u$
  ⇒ $u^* = -\dfrac{a\,b\,q}{r + b^2 q}\,x$.

## Claims list

- **I-1** (linearization): the analytic Jacobians equal central differences (rtol 1e-6)
  — the single most common iLQR implementation bug.
- **I-2** (backward pass): at any nominal trajectory, the backward pass's $Q_u$ equals the
  finite-difference gradient of the *total rolled-out cost* w.r.t. each $u_t$
  (first-order agreement; dynamics are nonlinear, so rtol 1e-3 with derivation).
- **I-3** (descent + convergence): accepted iterations never increase the cost; the run
  converges (cost improvement stalls below tolerance).
- **I-4** (constrained stationarity, KKT): at the returned solution, interior controls have
  ~zero rolled-out cost gradient; controls clamped at a bound have gradient of the
  correct sign (pushing into the bound).
- **I-5** (1-step LQR oracle): on a 1-step linear-quadratic problem iLQR reproduces the
  closed-form $u^*$ exactly.
- **I-6** (task): swing-up from hanging ($\theta=\pi$) to upright ($\theta\approx 0$) within
  $T$ steps, all controls inside the box.
