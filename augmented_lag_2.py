from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import ot
from solvers.adapted_ns.suot_l2 import SUOT_L2
from solvers.adapted_ns.uot_l2 import UOT_L2
from utils.utils import get_instance, primal_cost, primal_cost_full_quad

EPS = 1e-14
n = 30
np.random.seed(0)
a, b, C = get_instance(n)

# b = 2 * a

f, g = np.zeros(n) - 0.5, 0  # np.zeros(n)

# first = [
#     -0.002866049377580326,
#     -0.06993697273183698,
#     -0.06706712193018063,
#     -0.06102204103265588,
#     -0.052763971897747314,
#     -0.0017383053658718353,
#     -0.06148941907235689,
#     -0.04933962006221239,
#     -0.0894456157412474,
#     -0.004506760212565375,
#     -0.0864545495469519,
#     -0.06352796989308965,
#     -0.04718501857308549,
#     -0.06690312917519385,
#     -0.06641727614366069,
#     -0.0009493390424865448,
#     -0.04883843596003559,
#     -0.07957948238607511,
#     -0.07158567089796233,
#     -0.008383250957203645,
# ]


# pas bon


# g = 2 * np.array(first) - b


tau = 1
Lg = []
Lf = []
marg_b = []


marg_error = []
objs = []
for iteration in range(10000):
    print(iteration)
    # Graph method
    # # with UOT
    # graph = UOT_L2(C - f[:, None] - g[None, :], a, b, n, tau=tau)
    # with SUOT
    graph = SUOT_L2(C, a, b + tau * g, n, tau=tau)

    tp_graph = graph.solve(n_max=10000)
    u = deepcopy(graph.f_array)
    v = deepcopy(graph.g_array)
    first_order = tp_graph.T @ np.ones(n) - (b + tau * g - v)
    print("F0", np.abs(first_order).sum())
    f += u
    g = v
    if iteration == 0:
        print("here", np.min(v))
        print(list(v))
    if iteration % 10 == 0:
        tau = 1
    Lf.append(list(f))
    Lg.append(list(g))
    objs.append((tp_graph * C).sum())

    # marg_b.append(list(np.abs(tp_graph.sum(0) - b)))

    marg_error.append(
        max((tp_graph.sum(0) - b).max(), np.abs(tp_graph.sum(1) - a).max())
    )
    print(np.abs(tp_graph.sum(0) - b).max(), np.abs(tp_graph.sum(1) - a).max())
    # # print(f)
    # print("g", g)
    # print(b + tau * g)
    if marg_error[-1] < EPS:
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
