from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import ot
from solvers.adapted_ns.suot_l2 import SUOT_L2
from solvers.adapted_ns.uot_l2 import UOT_L2
from utils.utils import get_instance, primal_cost, primal_cost_full_quad

from solvers_reg.cvx import Cvx_ineq
from suot import pdhg_transport_one_relaxed

EPS = 1e-11
n = 10
np.random.seed(1)
a, b, C = get_instance(n)
tau = 1
f, g = np.zeros(n), -b / tau  # 0 * np.ones(n)


Lg = [g.flatten()]
Lf = []
marg_b = []
marg_coord = []


def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


import cvxpy as cp


def proj_simplex_cvx(V, a):
    matrix = V.copy()
    n, m = V.shape
    eps = 1e-15
    P = cp.Variable((n, m))
    u = np.ones((n, 1))
    constraints = [P >= 0, cp.matmul(P, u) <= a[:, None]]
    objective = cp.Minimize(cp.sum_squares(matrix - P))
    prob = cp.Problem(objective, constraints)
    prob.solve(
        solver=cp.CLARABEL,
        max_iter=int(1e4),
        tol_gap_abs=eps,
        tol_feas=eps,
        tol_gap_rel=eps,
        tol_infeas_abs=eps,
        tol_infeas_rel=eps,
        tol_ktratio=eps,
        # verbose=True,
    )
    return P.value


def proj_adapted_err(V, a):
    V_pos = V * (V >= 0)
    mask = V_pos @ np.ones(V.shape[0]) > a
    if np.sum(mask) > 0:
        V_proj = projection_simplex(V_pos.copy()[mask], a[mask], axis=1)
        V_pos[mask] = V_proj
        return V_pos
    return V_pos


def kkt(P, u, v, C, a, b, alpha, tol=1e-13, verbose=False, proj=False):
    # u_exact = (-P @ np.ones(P.shape[1]) + a) / alpha
    v_exact = (-P.T @ np.ones(P.shape[0]) + b) / alpha
    if proj:
        grad = C - v[None, :]
    else:
        grad = C - u[:, None] - v[None, :]
    r_compl = np.linalg.norm(P * grad, ord="fro")
    mask_zero = P <= tol
    r_dual = np.linalg.norm(np.minimum(grad, 0.0) * mask_zero, ord="fro")
    r_primal = np.linalg.norm(np.minimum(P, 0.0), ord="fro")
    r_uv = np.sum(np.abs(v - v_exact))
    # r_uv = np.sum(np.abs(u - v_exact))

    if verbose:
        print(
            f"Condition detail: {r_compl:.2e}  {r_dual:.2e}   {r_primal:.2e}   {r_uv:.2e}"
        )
    if not proj:
        return r_compl + r_dual + r_primal + r_uv
    else:
        return r_uv


def pdhg_transport_one_relaxed_proj(
    C,
    a,
    b,
    alpha,
    tau=0.1,
    sigma=0.1,
    niter=1000,
    verbose=False,
    P0=None,
    v0=None,
    tol=1e-14,
    proj_ineq=False,
):
    """
    PDHG pour :  min_{P>=0, P1=a} <P,C> + (1/(2α)) ||P^T1 - b||^2
    """
    n, m = C.shape
    if P0 is not None:
        P = P0.copy()
    else:
        P = np.zeros((n, m))
    if v0 is not None:
        v = v0.copy()

    else:
        v = np.zeros(n)
    u = np.zeros(n)
    # print(u, v)

    P_prev = np.zeros_like(P)

    obj_values = []
    kkt_conds = []
    # u = None

    for k in range(niter):
        # Extrapolation (optionnelle)
        kkt_cond = kkt(P, None, v, C, a, b, alpha, proj=True)
        kkt_conds.append(kkt_cond)
        # if k % 100 == 0:
        #     print(k)
        if kkt_cond < tol:
            print(k)
            break
        P_bar = 2 * P - P_prev if k > 0 else P.copy()

        # Dual updates
        col_sum = P_bar.T @ np.ones(n)

        # col_sum = P.T @ np.ones(n)
        factor = 1 / (1 + sigma * alpha)
        v = factor * (v + sigma * (b - col_sum))

        # Primal update (avec projection sur P >= 0)
        if not proj_ineq:
            grad = C - v[None, :]
            P_next = projection_simplex(P - tau * grad, a, axis=1)
        else:
            grad = C - v[None, :]
            # print(a)
            # print(P - tau * grad)
            # P_next_prime = proj_simplex_cvx((P - tau * grad).copy(), a)
            P_next = proj_adapted_err((P - tau * grad).copy(), a)
            # print(a)
            # print("P_next")
            # print(P_next)

            # if np.abs(P_next - P_next_prime).max() > 1e-4:
            #     print((P - tau * grad).tolist())
            #     print(np.abs(P_next - P_next_prime).max())
            #     raise ValueError
            # Sauvegarde
        P_prev = P
        P = P_next

        # Évaluation de l'objectif
        obj = np.sum(P * C) + 0.5 / 1 * np.sum((P.T @ np.ones(n) - b) ** 2)
        obj_values.append(obj)

        if verbose and (k % 1000 == 0 or k == niter - 1):
            print(f"Iter {k:4d} | Obj = {obj:.2e} | KKT = {kkt_cond:.2e}")

    print("k", k, f"KKT = {0:.2e}")
    print("FO", np.abs(P.T @ np.ones(n) - (b - alpha * v)).max())

    return P, u, v, np.array(obj_values), kkt_conds


marg_error = []
objs = []
change_support = []
tp_graph = np.zeros((n, n))
g_prev = g.copy()
tp_suivi = [tp_graph.flatten()]
for iteration in range(3000):
    print("iter", iteration)
    # Graph method
    # # with UOT
    # graph = UOT_L2(C - f[:, None] - g[None, :], a, b, n, tau=tau)
    # with SUOT
    # graph = SUOT_L2(C, a, b + tau * g, n, tau=tau)

    # tp_graph = graph.solve(n_max=10000)
    # u = deepcopy(graph.f_array)
    # v = deepcopy(graph.g_array)
    # # Without proj
    # tp_graph, u, v, _, _ = pdhg_transport_one_relaxed(
    #     C,
    #     a,
    #     b + tau * g,
    #     tau,
    #     tau=(2 * n) ** (-0.5),
    #     sigma=(2 * n) ** (-0.5),
    #     P0=tp_graph.copy(),
    #     u0=f.copy(),
    #     v0=g.copy(),
    # )
    beta_k = iteration / (iteration + 3)
    g_bar = g + beta_k * (g - g_prev)
    Lf.append(list(f))
    Lg.append(list(g))
    tp_graph_support = (tp_graph > EPS).copy()
    tp_graph, u_prime, v_prime, _, _ = pdhg_transport_one_relaxed_proj(
        C,
        a,
        b + tau * g_bar,
        tau,
        tau=(2 * n) ** (-0.5),
        sigma=(2 * n) ** (-0.5),
        P0=tp_graph.copy(),
        v0=g.copy(),
        niter=int(2e3),
        tol=1e-13,
        proj_ineq=True,
    )
    marg_coord.append(tp_graph.sum(0) - b)
    change_support_iter = np.all(tp_graph_support == (tp_graph > EPS))
    # (tp_graph_support == (tp_graph > EPS)).sum() == n**2
    change_support.append(change_support_iter)
    tp_suivi.append(tp_graph.flatten())
    # obj_prime = (tp_graph_prime * C).sum() + 0.5 * np.sum(
    #     (b + tau * g - tp_graph_prime.T @ np.ones(n)) ** 2
    # )

    # solver = Cvx_ineq(C, a, b + tau * g, n, tau=tau)
    # tp_graph = solver.solve()
    # obj = (tp_graph * C).sum() + 0.5 * np.sum(
    #     (b + tau * g - tp_graph.T @ np.ones(n)) ** 2
    # )

    # print(obj - obj_prime)

    v = b + g_bar - tp_graph.T @ np.ones(n)

    print(v)
    # f += u
    print("Increasing ? ", np.min(v - g), np.min(v - g) > -1e-10)
    g_prev = g.copy()
    g = v
    if iteration % 10 == 0:
        tau = tau
    objs.append((tp_graph * C).sum())

    # marg_b.append(list(np.abs(tp_graph.sum(0) - b)))
    # print("iter", iteration)

    marg_error.append(
        max(
            np.abs(tp_graph.sum(0) - b).max(), np.abs(tp_graph.sum(1) - a).max()
        )
        # np.abs(tp_graph.sum(0) - b).sum() + np.abs(tp_graph.sum(1) - a).sum()
    )
    print(np.abs(tp_graph.sum(0) - b).max(), np.abs(tp_graph.sum(1) - a).max())
    # print(f)
    # print("g", g)
    # print(b + tau * g)
    if marg_error[-1] < EPS:
        print(marg_error)
        break

tp_ot = ot.emd(a, b, C)

print(
    "Marg POT", np.abs(tp_ot.sum(0) - b).max(), np.abs(tp_ot.sum(1) - a).max()
)
print(np.sum(tp_ot * C), np.sum(tp_graph * C))
# print(np.array(L).shape)
# Lf = np.array(Lf)
# plt.figure()
# for i in range(n):
#     plt.plot(Lf[:, i], label=rf"$f_{str(i)}$")
# plt.legend()
# plt.grid()
# plt.title(r"Evolution of the dual variable $f$")
# plt.show()


# tp_suivi = np.array(tp_suivi)
# print(tp_suivi.shape)
# plt.figure()
# for i in range(n**2):
#     # print(Lg[:, i].shape)
#     plt.plot(tp_suivi[:, i], label=rf"$X_{str(i)}$")
# plt.legend()
# plt.grid()
# plt.title(r"Evolution of the dual variable $g$")
# for i, flag in enumerate(change_support):
#     if not flag:
#         plt.axvline(i, color="k", linestyle="--", alpha=0.3)
# plt.show()

# Lg = np.array(Lg)
# plt.figure()
# for i in range(n):
#     # print(Lg[:, i].shape)
#     plt.plot(Lg[:, i], label=rf"$g_{str(i)}$")
# plt.legend()
# plt.grid()
# plt.title(r"Evolution of the dual variable $g$")
# for i, flag in enumerate(change_support):
#     if not flag:
#         plt.axvline(i, color="k", linestyle="--", alpha=0.3)
# plt.show()
# objs = np.array(objs)

tp_suivi = np.array(tp_suivi)
Lg = np.array(Lg)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- Premier graphique : tp_suivi ---
for i in range(n**2):
    ax1.plot(tp_suivi[:, i], label=rf"$X_{i}$")

ax1.set_title(r"Evolution of the dual variable $X$")
ax1.grid()

for i, flag in enumerate(change_support):
    if not flag:
        ax1.axvline(i, color="k", linestyle="--", alpha=0.8)

ax1.legend(ncol=2, fontsize=8)

# --- Deuxième graphique : Lg ---
for i in range(n):
    ax2.plot(Lg[:, i], label=rf"$g_{i}$")

ax2.set_title(r"Evolution of the dual variable $g$")
ax2.grid()

for i, flag in enumerate(change_support):
    if not flag:
        ax2.axvline(i, color="k", linestyle="--", alpha=0.8)

ax2.legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.show()

marg_coord = np.array(marg_coord)
plt.figure()
for i in range(n):
    plt.plot(-marg_coord[:, i])
plt.yscale("log")
plt.show()

plt.figure()
plt.title(r"$\max(\|X\mathbf{1}-a\|_\infty,\|X^\top\mathbf{1}-b\|_\infty)$")
plt.yscale("log")
plt.grid()
plt.plot(marg_error, label="marg")
plt.plot(np.abs(objs - np.sum(tp_ot * C)), label="obj")
plt.legend()
for i, flag in enumerate(change_support):
    if not flag:
        plt.axvline(i, color="k", linestyle="--", alpha=0.3)
plt.show()


# plt.figure()
# plt.plot(np.array(change_support).astype(int))
# plt.show()
