"""iLQR on a torque-limited inverted pendulum — independent formulation with
finite-difference Jacobians.

Difference vs the formula version (core_eq): the dynamics Jacobians come from
central differences of the step function instead of the hand-derived closed form;
the solver skeleton is reimplemented here (no import of core_eq). This validates
the analytic derivatives in situ — the single most common iLQR implementation bug.
numpy-only, float64.
"""

import numpy as np

DT, G, B, I = 0.05, 9.81, 0.1, 1.0
U_MAX = 3.0


def pendulum_step(x, u, dt=DT, g=G, b=B, inertia=I):
    x = np.asarray(x, dtype=np.float64)
    theta, omega = x
    return np.array([theta + dt * omega,
                     omega + dt * (-g * np.sin(theta) - b * omega + u) / inertia])


def pendulum_jacobians_fd(x, u, dt=DT, g=G, b=B, inertia=I):
    """Central-difference Jacobians of the step function."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    h = 1e-6
    A = np.zeros((n, n))
    for i in range(n):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        A[:, i] = (pendulum_step(xp, u) - pendulum_step(xm, u)) / (2.0 * h)
    up, um = u + h, u - h
    Bc = ((pendulum_step(x, up) - pendulum_step(x, um)) / (2.0 * h))[:, None]
    return A, Bc


def ilqr_fd(x0, T, x_goal, w_x, w_u, w_f, u_max, iters=500, tol=1e-6):
    """Same solver contract as core_eq.ilqr, with FD Jacobians."""
    x0 = np.asarray(x0, dtype=np.float64)
    x_goal = np.asarray(x_goal, dtype=np.float64)
    w_x = np.asarray(w_x, dtype=np.float64)
    w_f = np.asarray(w_f, dtype=np.float64)

    def rollout(us):
        xs = [x0.copy()]
        for u in us:
            xs.append(pendulum_step(xs[-1], u))
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
            A, Bc = pendulum_jacobians_fd(xs[t], us[t])
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
            at_up = us[t] > u_max - 1e-9
            at_lo = us[t] < -u_max + 1e-9
            if at_up:
                kkt_viol = max(kkt_viol, max(0.0, Q_u[0]))
            elif at_lo:
                kkt_viol = max(kkt_viol, max(0.0, -Q_u[0]))
            else:
                kkt_viol = max(kkt_viol, abs(Q_u[0]))
        grad_inf = kkt_viol
        improved = False
        alpha = 1.0
        for _ls in range(20):
            us_new = np.empty(T)
            x = x0.copy()
            for t in range(T):
                dev = x - xs[t]
                u = us[t] + alpha * ks[t] + float(Ks[t] @ dev)
                us_new[t] = min(max(u, -u_max), u_max)
                x = pendulum_step(x, us_new[t])
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
        if grad_inf < tol:
            break
    return {"controls": us, "trajectory": xs, "costs": np.array(costs),
            "grad_inf": grad_inf}
