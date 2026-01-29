import numpy as np
from solvers.adapted_ns.suot_l2 import SUOT_L2


# conditions respect kkt
def kkt(P, u, v, C, a, b, alpha, tol=1e-13, verbose=False, proj=False):
    u_exact = (-P @ np.ones(P.shape[1]) + a) / alpha
    v_exact = (-P.T @ np.ones(P.shape[0]) + b) / alpha
    if proj:
        grad = C - v[None, :]
    else:
        grad = C - u[:, None] - v[None, :]
    r_compl = np.linalg.norm(P * grad, ord="fro")
    mask_zero = P <= tol
    r_dual = np.linalg.norm(np.minimum(grad, 0.0) * mask_zero, ord="fro")
    r_primal = np.linalg.norm(np.minimum(P, 0.0), ord="fro")
    r_uv = np.sum(np.abs(v - v_exact)) + np.sum(np.abs(u - u_exact))
    # r_uv = np.sum(np.abs(u - v_exact))

    if verbose:
        print(
            f"Condition detail: {r_compl:.2e}  {r_dual:.2e}   {r_primal:.2e}   {r_uv:.2e}"
        )
    if not proj:
        return r_compl + r_dual + r_primal + r_uv
    else:
        return r_uv


def pdhg_transport_one_relaxed(
    C,
    a,
    b,
    alpha,
    tau=0.1,
    sigma=0.1,
    niter=10000,
    verbose=False,
    P0=None,
    u0=None,
    v0=None,
    tol=1e-14,
):
    """
    PDHG pour :  min_{P>=0, P1=a} <P,C> + (1/(2α)) ||P^T1 - b||^2
    """
    n, m = C.shape
    if P0 is not None:
        P = P0.copy()
    else:
        P = np.zeros((n, m))
    if u0 is not None:
        u = u0.copy()
    else:
        u = np.zeros(n)
    if v0 is not None:
        v = v0.copy()
    else:
        v = np.zeros(n)

    P_prev = np.zeros_like(P)

    obj_values = []
    kkt_conds = []

    for k in range(niter):
        # Extrapolation (optionnelle)
        kkt_cond = kkt(P, u, v, C, a, b, alpha)
        kkt_conds.append(kkt_cond)
        if kkt_cond < tol:
            print(k)
            break
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
        obj = np.sum(P * C) + 0.5 / 1 * np.sum((P.T @ np.ones(n) - b) ** 2)
        obj_values.append(obj)

        if verbose and (k % 1000 == 0 or k == niter - 1):
            print(f"Iter {k:4d} | Obj = {obj:.2e} | KKT = {kkt_cond:.2e}")

    print("k", k, f"KKT = {kkt_cond:.2e}")

    return P, u, v, np.array(obj_values), kkt_conds


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


def projection_simplex_conditional(V, a, z=1):
    """
    Projection sur le simplex de masse z uniquement pour les lignes
    dont la somme est > a_i.

    Parameters
    ----------
    V : ndarray, shape (n_samples, n_features)
    a : ndarray, shape (n_samples,)
        Seuil sur la marginale ligne
    z : float
        Masse du simplex

    Returns
    -------
    V_proj : ndarray
    """
    V_proj = V.copy()

    # marginales lignes
    row_sums = V.sum(axis=1)

    # lignes à projeter
    mask = row_sums > a

    if not np.any(mask):
        return V_proj

    V_sub = V[mask]
    n_features = V_sub.shape[1]

    # projection simplex standard
    U = np.sort(V_sub, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1) - z
    ind = np.arange(1, n_features + 1)

    cond = U - cssv / ind > 0
    rho = np.count_nonzero(cond, axis=1)
    theta = cssv[np.arange(len(V_sub)), rho - 1] / rho

    V_proj[mask] = np.maximum(V_sub - theta[:, None], 0)

    return V_proj


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
        kkt_cond = 0  # kkt(P, None, v, C, a, b, alpha, proj=True)
        kkt_conds.append(kkt_cond)
        if kkt_cond + 3 * tol < tol:
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
            # print("here")
            # print(u, P_bar)
            # u = u + sigma * (a - P_bar @ np.ones(n))
            # grad = C - u[:, None] - v[None, :]
            grad = C - v[None, :]

            mask = (P - tau * grad) @ np.ones(n) > a
            P_proj = projection_simplex((P - tau * grad)[mask], a[mask], axis=1)
            P_next = np.maximum(P - tau * grad, 0)
            P_next[mask] = P_proj
            if mask.sum() > 0:
                print(mask)
                raise KeyError
            # print("res", np.sum(mask))

        # Sauvegarde
        P_prev = P
        P = P_next

        # Évaluation de l'objectif
        obj = np.sum(P * C) + 0.5 / 1 * np.sum((P.T @ np.ones(n) - b) ** 2)
        obj_values.append(obj)

        if verbose and (k % 1000 == 0 or k == niter - 1):
            print(f"Iter {k:4d} | Obj = {obj:.2e} | KKT = {kkt_cond:.2e}")

    print("k", k, f"KKT = {0:.2e}")
    print("FO", np.abs(P.T @ np.ones(n) - (b - v)).max())

    return P, u, v, np.array(obj_values), kkt_conds


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    np.random.seed(0)
    n = 80
    x = np.linspace(0, 1, n)
    C = (x[:, None] - x[None, :]) ** 2
    a = np.random.rand(n)
    a /= a.sum()
    b = np.random.rand(n)
    b /= b.sum()
    C = C / C.max()
    alpha = 0.1
    solver = SUOT_L2(C, a, b, n, alpha)
    tp = solver.solve(n_max=10000)
    obj_cvx = np.sum(tp * C) + 0.5 / 1 * np.sum((tp.T @ np.ones(n) - b) ** 2)
    tau = sigma = 1 / np.sqrt(2 * n)
    P1, u, v, objs1, kkt_conds = pdhg_transport_one_relaxed(
        C, a, b, alpha=alpha, tau=tau, sigma=sigma, niter=30_000, verbose=True
    )
    P1 = P1.T
    obj_pd1 = np.sum(P1 * C) + 0.5 / 1 * np.sum((P1.T @ np.ones(n) - b) ** 2)
    print(
        obj_cvx - obj_pd1,
        np.abs(P1 @ np.ones(n) - a).max(),
        np.abs(P1.T @ np.ones(n) - a).max(),
    )
    P2, u, v, objs2, kkt_conds2 = pdhg_transport_one_relaxed_proj(
        C, a, b, alpha=alpha, tau=tau, sigma=sigma, niter=30_000, verbose=True
    )
    P2 = P2.T
    obj_pd2 = np.sum(P2 * C) + 0.5 / 1 * np.sum((P2.T @ np.ones(n) - b) ** 2)
    print(
        obj_cvx - obj_pd2,
        np.abs(P2 @ np.ones(n) - a).max(),
        np.abs(P2.T @ np.ones(n) - a).max(),
    )

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(np.abs(obj_cvx - objs1), label="obj1")
    plt.plot(np.abs(obj_cvx - objs2), label="obj2")

    plt.plot(kkt_conds, label="kkt1")
    plt.plot(kkt_conds2, label="kkt2")
    plt.legend()

    plt.yscale("log")
    plt.show()
