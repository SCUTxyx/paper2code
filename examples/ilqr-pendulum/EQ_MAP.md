# Formula-to-code map: iLQR pendulum

| Paper location | Formula | Implementation |
|---|---|---|
| Pendulum model | $\ddot\theta = (-g\sin\theta - b\dot\theta + u)/I$ (Euler) | impl/core_eq.py:L19-24 (`pendulum_step`) |
| iLQR §linearization | $A = \partial f/\partial x$, $B = \partial f/\partial u$ (analytic) | impl/core_eq.py:L27-33 |
| iLQR §backward pass | $Q_u = l_u + B^\top V_x$ | impl/core_eq.py:L87 |
| iLQR §backward pass | $Q_{uu} = l_{uu} + B^\top V_{xx} B + \mu I$ | impl/core_eq.py:L89 |
| iLQR §backward pass | $k = -Q_{uu}^{-1}Q_u$, $K = -Q_{uu}^{-1}Q_{ux}$ | impl/core_eq.py:L91-93 |
| Tassa et al. 2012 (box) | $u = \mathrm{clip}(\hat u + \alpha k + K\delta x)$ | impl/core_eq.py:L115 |
| KKT convergence | projected-gradient violation under the box | impl/core_eq.py:L96-104, L130 |
| I-2 exact adjoint | $\lambda_t = l_x + A^\top\lambda_{t+1}$, $g_t = l_u + B^\top\lambda_{t+1}$ | impl/core_eq.py:L148-157 (`rollout_cost_grad`) |
| I-1 independent form | FD Jacobians (central differences) | impl/core_pseudo.py:L24-37 |
