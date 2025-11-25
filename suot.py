import numpy as np
from solvers.cvx import Cvx


def pdhg_transport_one_relaxed(
    C, a, b, alpha, tau=0.1, sigma=0.1, niter=1000, verbose=False
):
    """
    PDHG pour :  min_{P>=0, P1=a} <P,C> + (1/(2α)) ||P^T1 - b||^2
    """
    n, m = C.shape
    P = np.zeros((n, m))
    P_prev = np.zeros_like(P)
    u = np.zeros(n)
    v = np.zeros(m)

    obj_values = []

    for k in range(niter):
        # Extrapolation (optionnelle)
        P_bar = 2 * P - P_prev if k > 0 else P.copy()

        # Dual updates
        row_sum = P_bar @ np.ones(m)
        col_sum = P_bar.T @ np.ones(n)

        u += sigma * (a - row_sum)  # stricte contrainte
        v += sigma * (b - col_sum - alpha * v)  # relaxée

        # Primal update (avec projection sur P >= 0)
        grad = C - u[:, None] - v[None, :]
        P_next = np.maximum(0.0, P - tau * grad)

        # Sauvegarde
        P_prev = P
        P = P_next

        # Évaluation de l'objectif
        col_sum_P = P.T @ np.ones(n)
        obj = np.sum(P * C) + 0.5 / alpha * np.sum((col_sum_P - b) ** 2)
        obj_values.append(obj)

        if verbose and (k % 100 == 0 or k == niter - 1):
            print(f"Iter {k:4d} | Obj = {obj:.2e}")

    return P, u, v, np.array(obj_values)


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    np.random.seed(0)
    n = 30
    x = np.linspace(0, 1, n)
    C = (x[:, None] - x[None, :]) ** 2
    a = np.random.rand(n)
    a /= a.sum()
    b = np.random.rand(n)
    b /= b.sum()
    C = C / C.max()
    solver = Cvx(C, a, b, n, 1)
    tp = solver.solve(reg="quad")
    obj_cvx = np.sum(tp * C) + 0.5 / 1 * np.sum((tp.T @ np.ones(n) - b) ** 2)
    tau = sigma = 2 / np.sqrt(2 * n)
    P, u, v, objs = pdhg_transport_one_relaxed(
        C.T, b, a, alpha=1.0, tau=tau, sigma=sigma, niter=100_000, verbose=True
    )
    P = P.T
    obj_pd = np.sum(P * C) + 0.5 / 1 * np.sum((P.T @ np.ones(n) - b) ** 2)
    print(
        obj_cvx - obj_pd,
        np.abs(P @ np.ones(n) - a).max(),
        np.abs(P.T @ np.ones(n) - a).max(),
    )
