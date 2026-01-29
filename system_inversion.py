import numpy as np
from utils.utils import get_instance

n = 3
a, b, C = get_instance(n)

Hr = np.repeat(np.eye(n), n, axis=1)
Hc = np.tile(np.eye(n), n)

# X = np.random.rand(n, n)

# print(Hr)
# print(Hc)
# print(X.shape, X.flatten().shape, Hr.shape, Hc.shape)
# print(Hr @ X.flatten() - X.sum(1), X @ np.ones(n) - X.sum(1))
# print(Hc @ X.flatten() - X.sum(0), X.T @ np.ones(n) - X.sum(0))


EPS = 1e-11
n = 10
Nmax = 3000
# np.random.seed(1)
a, b, C = get_instance(n)
# tau = 1
u, v = np.zeros(n), np.ones(n)


def recover_f(C, g, support):
    f = np.sum(support * (C - g[None, :]), axis=1) / np.sum(support, axis=1)
    # do the checks C\geq u\oplus v and (C-u\oplus v)[support] = 0
    return f


# 1. get an X_1 associated to u
# 2. compute support of X_1
# 3. compute u1 = b + u0 - X^T1
# 3. compute v1 = recover_f(C,u1,support)
for iteration in range(Nmax):
    # - I -    If the previous support is still compatible
    # 1. if support compatible with kkt (X\geq 0 and C-u\oplus v \geq 0)
    # X is still unchanged
    # u^{k+1} = b+u^k-X^T1

    ...
