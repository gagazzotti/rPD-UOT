import numpy as np
from solvers.adapted_ns.suot_l2 import SUOT_L2
from utils.utils import get_instance

# X = np.random.rand(n, n)

# print(Hr)
# print(Hc)
# print(X.shape, X.flatten().shape, Hr.shape, Hc.shape)
# print(Hr @ X.flatten() - X.sum(1), X @ np.ones(n) - X.sum(1))
# print(Hc @ X.flatten() - X.sum(0), X.T @ np.ones(n) - X.sum(0))


EPS = 1e-11
n = 10
Hc = np.repeat(np.eye(n), n, axis=1)
Hr = np.tile(np.eye(n), n)
Nmax = 200
np.random.seed(1)
a, b, C = get_instance(n)
# tau = 1
v, u = np.zeros(n), np.zeros(n)


def recover_f(C, g, support):
    # si f g  les var tq f\oplus g = C_ij sur le support, on retrouve f a partir de g
    f = np.sum(support * (C - g[None, :]), axis=1) / np.sum(support, axis=1)
    # do the checks C\geq u\oplus v and (C-u\oplus v)[support] = 0
    return f


def get_Ek(supp, Hr, Hc):
    print("card sup", np.sum(supp), "2n+card", 2 * n + np.sum(supp))
    # supp should be the flattend support
    # Nk = np.linalg.inv((Hc[:, supp].T @ Hc[:, supp])) @ Hc[:, supp].T
    # Ek = -Hr[:, supp] @ Nk
    card_supp = int(np.sum(supp))
    Mk = np.block(
        [
            [Hr[:, supp].T, Hc[:, supp].T, np.zeros((card_supp, card_supp))],
            [np.zeros((n, n)), np.zeros((n, n)), Hc[:, supp]],
            [
                np.eye(n),
                np.zeros((n, n)),
                Hr[:, supp],
            ],
        ]
    )
    print(Mk.shape)

    # return Ek, Nk
    return Mk


graph = SUOT_L2(C, a, b, n)
# 1. get an X_1 associated to u0
X = graph.solve(n_max=int(1e4))
# 2. compute support of X_1
support = X > EPS
# 3. compute u1 = b + u0 - X^T1
u = b + u - X.T @ np.ones(n)
# 3. compute v1 = recover_f(C,u1,support)

Mk = get_Ek(support.flatten(), Hr, Hc)

# IT WORKS
# vec = np.vstack(
#     [C.flatten()[support.flatten()][:, None], a[:, None], b[:, None]]
# )
# sol = np.linalg.inv(Mk) @ vec
# print((np.linalg.inv(Mk) @ vec))

# print(sol[-np.sum(support) :][:, 0])
# print(X.flatten()[support.flatten()])

# print(sol, u)


def check_kkt(x, C, u, v):
    if np.any(x < -EPS):
        print("positivity")
        return False
    if np.any(C - u[:, None] - v[None, :] < -EPS):
        print("-- dual")
        return False
    return True


for iteration in range(Nmax):
    # - I -    If the previous support is still compatible
    # 1. if support compatible with kkt (X\geq 0 and C-u\oplus v \geq 0)
    # X is still unchanged
    # u^{k+1} = b+u^k-X^T1
    vec_kp1 = np.vstack(
        [
            C.flatten()[support.flatten()][:, None],
            a[:, None],
            b[:, None] + u[:, None],
        ]
    )
    sol_kp1 = np.linalg.inv(Mk) @ vec_kp1
    X_kp1 = sol_kp1[-np.sum(support) :][:, 0]
    u_kp1, v_kp1 = sol_kp1[:n][:, 0], sol_kp1[n : 2 * n][:, 0]
    check = check_kkt(X_kp1, C, u, v)
    print(check)
    if check:
        print(np.abs(u_kp1 - u).sum())
        u = u_kp1.copy()
    else:
        graph = SUOT_L2(C, a, b + u, n)
        # 1. get an X_1 associated to u0
        X = graph.solve(n_max=int(1e4))
        support = X > EPS
        print(np.abs(X.T @ np.ones(n) - b).sum())
        u = b + u - X.T @ np.ones(n)
        Mk = get_Ek(support.flatten(), Hr, Hc)
