from copy import deepcopy
from time import time

import matplotlib.pyplot as plt
import numpy as np
import ot
from solvers.adapted_ns.suot_l2 import SUOT_L2
from solvers.adapted_ns.uot_l2 import UOT_L2
from utils.utils import get_instance, primal_cost, primal_cost_full_quad

EPS = 1e-15
n = 30
np.random.seed(0)
a, b, C = get_instance(n)
f, g = np.zeros(n) - 0.5, np.zeros(n) - 10
tau = 0.05
Lg = []
Lf = []
marg_b = []


marg_error = []

for iteration in range(300):
    print(iteration)
    # Graph method
    # # with UOT
    graph = UOT_L2(C - f[:, None] - g[None, :], a, b, n, tau=tau)
    # with SUOT
    # graph = SUOT_L2(C - g[None, :], a, b, n, tau=tau)

    tp_graph = graph.solve(n_max=1000)
    u = deepcopy(graph.f_array)
    v = deepcopy(graph.g_array)
    f += u
    g += v
    # tau /= 1
    Lf.append(list(f))
    Lg.append(list(g))

    # marg_b.append(list(np.abs(tp_graph.sum(0) - b)))

    marg_error.append(
        max((tp_graph.sum(0) - b).max(), np.abs(tp_graph.sum(1) - a).max())
    )
    print(np.abs(tp_graph.sum(0) - b).max(), np.abs(tp_graph.sum(1) - a).max())
    print(f)
    print(g)

tp_ot = ot.emd(a, b, C)

print(
    "Marg POT", np.abs(tp_ot.sum(0) - b).max(), np.abs(tp_ot.sum(1) - a).max()
)
print(np.sum(tp_ot * C), np.sum(tp_graph * C))
# print(np.array(L).shape)
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

plt.figure()
plt.title(r"$\max(\|X\mathbf{1}-a\|_\infty,\|X^\top\mathbf{1}-b\|_\infty)$")
plt.yscale("log")
plt.grid()
plt.plot(marg_error)
plt.show()
