from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import ot
from solvers.adapted_ns.suot_l2 import SUOT_L2
from solvers.adapted_ns.uot_l2 import UOT_L2
from utils.utils import get_instance, primal_cost, primal_cost_full_quad

from suot import pdhg_transport_one_relaxed, pdhg_transport_one_relaxed_proj


# conditions respect kkt
def kkt(P, u, v, C, a, b, alpha, tol=1e-13, verbose=False):
    u_exact = (-P @ np.ones(P.shape[1]) + a) / alpha
    v_exact = (-P.T @ np.ones(P.shape[0]) + b) / alpha

    grad = C - u[:, None] - v[None, :]
    r_compl = np.linalg.norm(P * grad, ord="fro")
    mask_zero = P <= tol
    r_dual = np.linalg.norm(np.minimum(grad, 0.0) * mask_zero, ord="fro")
    r_primal = np.linalg.norm(np.minimum(P, 0.0), ord="fro")
    r_uv = np.sum(np.abs(u - u_exact)) + np.sum(np.abs(v - v_exact))
    if verbose:
        print(
            f"Condition detail: {r_compl:.2e}  {r_dual:.2e}   {r_primal:.2e}   {r_uv:.2e}"
        )
    return r_compl + r_dual + r_primal + r_uv


# cout primal
def primal_cost(P, C, a, b, alpha):
    row_sum = P @ np.ones(P.shape[1])
    col_sum = P.T @ np.ones(P.shape[0])
    cost = np.sum(P * C)
    cost += 0.5 / alpha * np.sum((row_sum - a) ** 2)
    cost += 0.5 / alpha * np.sum((col_sum - b) ** 2)
    return cost


def pdhg_uot(
    C,
    a,
    b,
    alpha,
    tau=0.1,
    sigma=0.1,
    niter=500,
    verbose=False,
    tol=1e-14,
    P_ref=None,
    k_kkt=100,
    accel=False,
    P0=None,
    v0=None,
):
    n, m = C.shape
    if P0 is not None:
        P = P0.copy()
    else:
        P = np.zeros((n, m))
    if v0 is not None:
        v = v0.copy()
    else:
        v = np.zeros(n)
    P = np.zeros((n, m))
    u = np.zeros(n)
    v = np.zeros(m)
    u_prev = u
    v_prev = v
    obj_values = []
    kkts = []
    tps = []
    if accel:
        theta = 0
    for k in range(niter):
        # on verifie les kkt tous les k_kkt
        if k % k_kkt == 0:
            obj = primal_cost(P, C, a, b, alpha)
            if P_ref is not None:
                error_plan = np.abs(P - P_ref).max()
            kkt_cond = kkt(P, u, v, C, a, b, alpha)
            if kkt_cond < tol:
                print("break")
                break

        P_bar = 2 * P - P_prev if k > 0 else P.copy()

        # dual
        row_sum = P_bar @ np.ones(m)
        col_sum = P_bar.T @ np.ones(n)
        u = (u + sigma * (a - row_sum)) / 1  # (1 + sigma * alpha)
        v = (v + sigma * (b - col_sum)) / (1 + sigma * alpha)

        # primal
        grad = C - u[:, None] - v[None, :]
        P_next = np.maximum(0.0, P - tau * grad)

        # sauvegarde
        P_prev = P.copy()
        P = P_next

        # obj = primal_cost(P, C, a, b, alpha, reg=reg)
        obj_values.append(obj)
        kkts.append(kkt_cond)
        if P_ref is not None:
            tps.append(error_plan)

        if verbose and (k % 1000 == 0 or k == niter - 1):
            print(
                f"Iter {k:4d} | Obj = {obj:.3e} | kkt = {kkt(P, u, v, C, a, b, alpha, verbose=True):.2e}"
            )

    return P, u, v, np.array(obj_values), np.array(tps), np.array(kkts)


EPS = 1e-14
n = 10
np.random.seed(0)
a, b, C = get_instance(n)

f, g = np.zeros(n), np.zeros(n)


tau = 1
Lg = []
Lf = []
marg_b = []


marg_error = []
objs = []
tp_graph = np.zeros((n, n))
for iteration in range(1000):
    print(iteration)
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
    tp_graph, u, v, _, _, _ = pdhg_uot(
        C,
        a,
        b + tau * g,
        tau,
        tau=(2 * n) ** (-0.5),
        sigma=(2 * n) ** (-0.5),
        P0=tp_graph.copy(),
        v0=g.copy(),
        niter=int(1e5),
        tol=1e-13,
    )
    # f += u
    g = v
    if iteration % 10 == 0:
        tau = 1
    Lf.append(list(u))
    Lg.append(list(g))
    objs.append((tp_graph * C).sum())

    # marg_b.append(list(np.abs(tp_graph.sum(0) - b)))

    marg_error.append(
        max((tp_graph.sum(0) - b).max(), np.abs(tp_graph.sum(1) - a).max())
    )
    print(np.abs(tp_graph.sum(0) - b).max(), np.abs(tp_graph.sum(1) - a).max())
    # print(f)
    # print("g", g)
    # print(b + tau * g)
    if marg_error[-1] < EPS:
        break

tp_ot = ot.emd(a, b, C)

print(
    "Marg POT", np.abs(tp_ot.sum(0) - b).max(), np.abs(tp_ot.sum(1) - a).max()
)
print(np.sum(tp_ot * C), np.sum(tp_graph * C))
print(np.array(Lf).shape)
Lf = np.array(Lf)
plt.figure()
for i in range(n):
    plt.plot(Lf[:, i], label=rf"$f_{str(i)}$")
plt.legend()
plt.grid()
plt.title(r"Evolution of the dual variable $f$")
plt.show()

Lg = np.array(Lg)
plt.figure()
for i in range(n):
    # print(Lg[:, i].shape)
    plt.plot(Lg[:, i], label=rf"$g_{str(i)}$")
plt.legend()
plt.grid()
plt.title(r"Evolution of the dual variable $g$")
plt.show()
objs = np.array(objs)

plt.figure()
plt.title(r"$\max(\|X\mathbf{1}-a\|_\infty,\|X^\top\mathbf{1}-b\|_\infty)$")
plt.yscale("log")
plt.grid()
plt.plot(marg_error, label="marg")
plt.plot(np.abs(objs - np.sum(tp_ot * C)), label="obj")
plt.legend()
plt.show()
