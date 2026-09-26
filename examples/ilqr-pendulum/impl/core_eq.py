"""iLQR on a torque-limited inverted pendulum — analytic-Jacobian implementation
(formula version).

Sources: Li & Todorov (2004); box handling per Tassa, Erez & Todorov (ICRA 2012).
Pendulum: theta measured from UPRIGHT (0 = unstable top, pi = hanging);
theta_dd = (-g·sin(theta) - b·theta_dot + u) / I, Euler-discretized.
The iLQR solver is problem-agnostic (dynamics + Jacobians are callables), which is
what allows the 1-step LQR closed-form anchor to run through the same code path.
Shapes: state (n,), control kept as a column (m,1) so the Bellman recursions are
shape-honest. numpy-only, float64.
"""

import numpy as np

DT, G, B, I = 0.05, 9.81, 0.1, 1.0
U_MAX = 3.0


def pendulum_step(x, u, dt=DT, g=G, b=B, inertia=I):
    """x' = (theta + dt·ω, ω + dt·(-g·sinθ - b·ω + u)/I). u scalar."""
    x = np.asarray(x, dtype=np.float64)
    theta, omega = x
    return np.array([theta + dt * omega,
                     omega + dt * (-g * np.sin(theta) - b * omega + u) / inertia])


def pendulum_jacobians(x, u, dt=DT, g=G, b=B, inertia=I):
    """Analytic A = ∂f/∂x (2,2), B = ∂f/∂u (2,1) — hand-derived, see METHOD_CARD."""
    x = np.asarray(x, dtype=np.float64)
    theta, omega = x
    A = np.array([[1.0, dt],
                  [-dt * g * np.cos(theta), 1.0 - dt * b / inertia]])
    Bc = np.array([[0.0], [dt / inertia]])
    return A, Bc


def ilqr(x0, T, step_fn, jac_fn, x_goal, w_x, w_u, w_f, u_max,
         iters=500, tol=1e-6):
    """Problem-agnostic control-limited iLQR (Tassa et al. 2012 style).

    step_fn(x, u) -> x';  jac_fn(x, u) -> (A, B) with B shaped (n, 1).
    Cost: ½(x−g)ᵀdiag(w_x)(x−g) + ½ w_u u² per step, terminal ½(x_T−g)ᵀdiag(w_f)(x_T−g).
    Convergence: KKT violation under the box (interior |Q_u|, clamped signed),
    NOT a line-search stall. Deterministic.
    Returns dict(controls, trajectory, costs, grad_inf).
    """
    x0 = np.asarray(x0, dtype=np.float64)
    x_goal = np.asarray(x_goal, dtype=np.float64)
    w_x = np.asarray(w_x, dtype=np.float64)
    w_f = np.asarray(w_f, dtype=np.float64)

    def rollout(us):
        xs = [x0.copy()]
        for u in us:
            xs.append(step_fn(xs[-1], u))
        return np.array(xs)

    def total_cost(us, xs):
        J = 0.0
        for t in range(T):
            e = xs[t] - x_goal
            J += 0.5 * w_x @ (e * e) + 0.5 * w_u * us[t] ** 2
        e = xs[-1] - x_goal
        return J + 0.5 * w_f @ (e * e)

    us = np.zeros(T)
    xs = rollout(us)
    J = total_cost(us, xs)
    costs = [J]
    mu = 1.0
    grad_inf = np.inf
    for _ in range(iters):
        # backward pass (Bellman, first-order around the nominal)
        e = xs[-1] - x_goal
        V_x = w_f * e
        V_xx = np.diag(w_f)
        ks, Ks = np.zeros(T), np.zeros((T, len(x0)))
        kkt_viol = 0.0
        for t in range(T - 1, -1, -1):
            e = xs[t] - x_goal
            l_x = w_x * e
            l_u = w_u * float(us[t])
            l_xx = np.diag(w_x)
            l_uu = w_u * np.eye(1)
            A, Bc = jac_fn(xs[t], us[t])
            Q_x = l_x + A.T @ V_x
            Q_u = np.array([l_u + float((Bc.T @ V_x)[0])])
            Q_xx = l_xx + A.T @ V_xx @ A
            Q_uu = l_uu + Bc.T @ V_xx @ Bc + mu * np.eye(1)
            Q_ux = Bc.T @ V_xx @ A
            sol_u = np.linalg.solve(Q_uu, Q_u.reshape(1, 1))
            ks[t] = -sol_u[0, 0]
            Ks[t] = -np.linalg.solve(Q_uu, Q_ux)[0]
            V_x = Q_x - (Q_ux.T @ sol_u).ravel()
            V_xx = Q_xx - Q_ux.T @ np.linalg.solve(Q_uu, Q_ux)
            # KKT violation under the box: interior → |Q_u|; clamped → signed
            at_up = us[t] > u_max - 1e-9
            at_lo = us[t] < -u_max + 1e-9
            if at_up:
                kkt_viol = max(kkt_viol, max(0.0, Q_u[0]))
            elif at_lo:
                kkt_viol = max(kkt_viol, max(0.0, -Q_u[0]))
            else:
                kkt_viol = max(kkt_viol, abs(Q_u[0]))
        grad_inf = kkt_viol
        # forward pass: backtracking line search + box clamp
        improved = False
        alpha = 1.0
        for _ls in range(20):
            us_new = np.empty(T)
            x = x0.copy()
            for t in range(T):
                dev = x - xs[t]
                u = us[t] + alpha * ks[t] + float(Ks[t] @ dev)
                us_new[t] = min(max(u, -u_max), u_max)
                x = step_fn(x, us_new[t])
            J_new = total_cost(us_new, rollout(us_new))
            if J_new < J:
                us, J = us_new, J_new
                xs = rollout(us)
                improved = True
                break
            alpha *= 0.5
        costs.append(J)
        if not improved:
            mu = min(mu * 10.0, 1e8)
        else:
            mu = max(mu / 10.0, 1e-8)
        # convergence = KKT-stationary nominal under the box, NOT a line-search stall
        if grad_inf < tol:
            break
    return {"controls": us, "trajectory": xs, "costs": np.array(costs),
            "grad_inf": grad_inf}


def swing_up(u_max=U_MAX, T=100):
    """The canonical swing-up problem: from hanging (π, 0) to upright (0, 0)."""
    return ilqr(x0=np.array([np.pi, 0.0]), T=T,
                step_fn=pendulum_step, jac_fn=pendulum_jacobians,
                x_goal=np.array([0.0, 0.0]),
                w_x=(1.0, 0.1), w_u=1e-2, w_f=(20.0, 2.0), u_max=u_max)


def rollout_cost_grad(x0, us, x_goal, w_x, w_u, w_f):
    """Exact discrete-adjoint gradient of the rolled-out cost w.r.t. every u_t
    (first-order recursion; assumes controls are interior — clamping is
    nondifferentiable and handled by the KKT logic instead)."""
    x0 = np.asarray(x0, dtype=np.float64)
    x_goal = np.asarray(x_goal, dtype=np.float64)
    w_x = np.asarray(w_x, dtype=np.float64)
    w_f = np.asarray(w_f, dtype=np.float64)
    xs = [x0.copy()]
    for u in us:
        xs.append(pendulum_step(xs[-1], u))
    xs = np.array(xs)
    lam = w_f * (xs[-1] - x_goal)
    grad = np.zeros(len(us))
    for t in range(len(us) - 1, -1, -1):
        e = xs[t] - x_goal
        A, Bc = pendulum_jacobians(xs[t], us[t])
        grad[t] = w_u * us[t] + float((Bc.T @ lam)[0])
        lam = w_x * e + A.T @ lam
    return grad
