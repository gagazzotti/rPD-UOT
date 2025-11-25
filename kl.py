import numpy as np
from scipy.special import lambertw


def kl_div(x, a):
    """KL(x|a) component-wise, safe for x==0 (0*log0 := 0)."""
    x = np.asarray(x)
    a = np.asarray(a)
    # use where to avoid 0*log0 NaNs
    term = np.where(x > 0, x * np.log(x / a), 0.0)
    return np.sum(term - x + a)


def prox_sigma_fstar_scalar(p, sigma, a, rho):
    """
    prox_{sigma * f^*}(p) where f(x) = rho * KL(x | a).
    Formula derived in the chat:
      prox = p - rho * W( (sigma * a / rho) * exp(p / rho) )
    Computed elementwise with principal branch W0. Returns real part.
    """
    # argument for W
    arg = (sigma * a / rho) * np.exp(p / rho)
    # lambertw may return complex with tiny imag part; take real part
    w = lambertw(arg, k=0)
    return p - rho * np.real(w)


def pdhg_uot_kl(
    C,
    a,
    b,
    rho_a=1.0,
    rho_b=1.0,
    tau=0.1,
    sigma=0.1,
    niter=500,
    P0=None,
    verbose=False,
    tol=1e-13,
):
    """
    Chambolle-Pock (PDHG) for:
      min_{P>=0} <C,P> + rho_a * KL(P 1 | a) + rho_b * KL(P^T 1 | b)

    Inputs:
      C : (n,m) cost matrix
      a : vector length n (target row marginals)
      b : vector length m (target col marginals)
      rho_a, rho_b : positive weights for the KL penalties
      tau, sigma : step sizes (must satisfy tau * sigma * ||K||^2 < 1 in practice)
      niter, P0, verbose, tol : as before

    Returns:
      P, u, v, obj_values
    """
    n, m = C.shape
    a = np.asarray(a).reshape(n)
    b = np.asarray(b).reshape(m)

    if P0 is None:
        P = np.zeros((n, m))
    else:
        P = P0.copy()
    P_prev = P.copy()

    # dual variables (size n and m)
    u = np.zeros(n)
    v = np.zeros(m)

    obj_values = []

    for k in range(niter):
        # extrapolation
        P_bar = 2 * P - P_prev if k > 0 else P.copy()

        # K(P_bar) = (row_sum, col_sum)
        row_sum_bar = P_bar @ np.ones(m)
        col_sum_bar = P_bar.T @ np.ones(n)

        # Dual update: y^{k+1} = prox_{sigma F^*}( y^k + sigma K(P_bar) )
        # where F^* for scaled KL is: (rho * a) * (exp(y/rho) - 1)
        u_tilde = u + sigma * row_sum_bar
        v_tilde = v + sigma * col_sum_bar

        # apply prox componentwise using the scaled-lambert formula
        u = prox_sigma_fstar_scalar(u_tilde, sigma, a, rho_a)
        v = prox_sigma_fstar_scalar(v_tilde, sigma, b, rho_b)

        # Primal update:
        # x^{k+1} = prox_{tau G}( x^k - tau K^T y^{k+1} )
        # with G(P)=<C,P> + I_{P>=0} => prox is projection of (previous - tau*C)
        # So P_next = max(0, P - tau*(C + u[:,None] + v[None,:]))
        KT_y = u[:, None] + v[None, :]
        P_next = np.maximum(0.0, P - tau * (C + KT_y))

        P_prev = P
        P = P_next

        # objective for monitoring
        row_sum_P = P @ np.ones(m)
        col_sum_P = P.T @ np.ones(n)
        obj = (
            np.sum(P * C)
            + rho_a * kl_div(row_sum_P, a)
            + rho_b * kl_div(col_sum_P, b)
        )
        obj_values.append(obj)

        # optional verbose / stopping on KKT-like residual (we keep a simple primal-dual residual)
        if verbose and (k % 100 == 0 or k == niter - 1):
            # compute a simple primal-dual gap proxy or residual norm
            primal_res = np.linalg.norm(
                np.minimum(P, C + KT_y)
            )  # complementarity proxy
            print(
                f"Iter {k:4d} | Obj = {obj:.6f} | primal_res ~ {primal_res:.2e}"
            )

        # optional tiny tol check: change in P
        if k > 0 and np.linalg.norm(P - P_prev) < tol:
            if verbose:
                print(f"converged (delta P < {tol}) at iter {k}")
            break

    return P, u, v, np.array(obj_values)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from solvers.cvx import Cvx

    # === Import de ta fonction ===
    # (si le code est dans un fichier pdhg_uot_kl.py)
    # from pdhg_uot_kl import pdhg_uot_kl
    # Ici on suppose que pdhg_uot_kl est déjà défini dans ton notebook.

    # === Données jouet ===
    n, m = 50, 50
    np.random.seed(0)

    # Matrice de coût (distance au carré)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, m)
    C = (x[:, None] - y[None, :]) ** 2

    # Distributions de référence (normalisées)
    a = np.random.rand(n)
    a /= a.sum()
    b = np.random.rand(m)
    b /= b.sum()
    alpha = 0.2
    # === Appel de l’algorithme ===
    P, u, v, obj = pdhg_uot_kl(
        C,
        a,
        b,
        rho_a=alpha,  # régularisation KL pour la marge gauche
        rho_b=alpha,  # régularisation KL pour la marge droite
        tau=(2 * n) ** (-0.5),
        sigma=(2 * n) ** (-0.5),
        niter=100000,
        verbose=True,
    )

    solver = Cvx(C, a, b, n, 1 / alpha)
    tp = solver.solve(reg="full_kl")
    obj_cvx = (
        np.sum(tp * C)
        + alpha * kl_div(tp @ np.ones(n), a)
        + alpha * kl_div(P.T @ np.ones(n), b)
    )

    # === Vérifications ===
    print("\nSomme totale du plan :", P.sum())
    print("Marge ligne (P1) :", np.round(P @ np.ones(m), 4))
    print("Marge cible a    :", np.round(a, 4))
    print("Marge colonne (P^T1) :", np.round(P.T @ np.ones(n), 4))
    print("Marge cible b       :", np.round(b, 4))

    # === Affichage de la convergence ===
    plt.figure()
    plt.plot(np.abs(obj - obj_cvx))
    plt.xlabel("Itération")
    plt.ylabel("Objectif")
    plt.yscale("log")
    plt.title("Convergence PDHG (UOT avec deux KL)")
    plt.grid(True)
    plt.show()

    print(np.abs(P - tp).max())
