import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

########################
# Solver avec CVXPY ####
########################


class CvxSolver:
    def __init__(
        self,
        cost_matrix: npt.NDArray[np.float64],
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        tau: float = 1,
        alpha: npt.NDArray[np.float64] = None,
        beta: npt.NDArray[np.float64] = None,
    ):
        self.cost_matrix: npt.NDArray = cost_matrix
        self.a: npt.NDArray = a
        self.b: npt.NDArray = b
        self.tau = tau
        self.n: int = self.get_n()
        self.alpha = alpha
        self.beta = beta

    def get_n(self):
        assert len(self.a) == len(self.b)
        return len(self.a)

    def solve(self, eps: float = 1e-15, max_iter: int = 100_000):
        P = cp.Variable((self.n, self.n))
        u = np.ones((self.n, 1))
        constraints = [P >= 0]
        objective = cp.Minimize(
            cp.sum(cp.multiply(P, self.cost_matrix))
            + 0.5
            * self.tau ** (-1)
            * cp.sum_squares(cp.matmul(P, u) - self.a[:, None])
            + 0.5
            * self.tau ** (-1)
            * cp.sum_squares(cp.matmul(P.T, u) - self.b[:, None])
        )
        prob = cp.Problem(objective, constraints)
        prob.solve(
            solver=cp.CLARABEL,
            max_iter=max_iter,
            tol_gap_abs=eps,
            tol_feas=eps,
            tol_gap_rel=eps,
            tol_infeas_abs=eps,
            tol_infeas_rel=eps,
            tol_ktratio=eps,
        )
        return P.value


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
):
    n, m = C.shape
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

        # acceleration
        if accel:
            u_bar = u + theta * (u - u_prev)
            v_bar = v + theta * (v - v_prev)

            # Primal update
            grad = C - u_bar[:, None] - v_bar[None, :]
            P = np.maximum(0.0, P - tau * grad)

            # Dual updates
            row_sum = P @ np.ones(m)
            col_sum = P.T @ np.ones(n)
            factor = 1 / (1 + sigma * alpha)
            u_next = factor * (u + sigma * (a - row_sum))
            v_next = factor * (v + sigma * (b - col_sum))

            u_prev = u
            u = u_next

            v_prev = v
            v = v_next

            theta = 1.0 / np.sqrt(1 + sigma * (alpha))
            tau = tau / theta
            sigma = sigma * theta

        # version normale
        else:
            P_bar = 2 * P - P_prev if k > 0 else P.copy()

            # dual
            row_sum = P_bar @ np.ones(m)
            col_sum = P_bar.T @ np.ones(n)
            u = (u + sigma * (a - row_sum)) / (1 + sigma * alpha)
            v = (v + sigma * (b - col_sum)) / (1 + sigma * alpha)

            # primal
            grad = C - u[:, None] - v[None, :]
            P_next = np.maximum(0.0, P - tau * grad)

            # sauvegarde
            P_prev = P
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


if __name__ == "__main__":
    """
    On résout le pb:
    min_{P>= 0} P:C + 1/(2*alpha) [|P1-a|^2+|P^T1-b|^2]
    """
    np.random.seed(2)
    # instance du probleme
    n = 40
    x = np.linspace(0, 1, n)
    C = np.abs(x[:, None] - x[None, :]) ** 2
    C /= C.max()
    a = np.random.rand(n)
    a /= a.sum()
    b = np.random.rand(n)
    b /= b.sum()
    alpha = 0.1

    # resolution avec CVXPY
    solver = CvxSolver(C, a, b, alpha)
    tp = solver.solve(eps=1e-18)
    cost_cvx = primal_cost(tp, C, a, b, alpha)

    tau = sigma = 0.8 / (2 * n) ** 0.5
    # version acceleree
    P_accel, u_accel, v_accel, objs_accel, tps_accel, kkts_accel = pdhg_uot(
        C,
        a,
        b,
        alpha=alpha,
        tau=tau,
        sigma=sigma,
        niter=20_000,
        verbose=True,
        P_ref=tp,
        accel=True,
    )

    # version normale
    P, u, v, objs, tps, kkts = pdhg_uot(
        C,
        a,
        b,
        alpha=alpha,
        tau=tau,
        sigma=sigma,
        niter=20_000,
        verbose=True,
        P_ref=tp,
        accel=False,
    )

    #################
    ## Affichage ####
    #################

    plt.figure(figsize=(6, 4))
    plt.plot(
        np.abs(objs_accel - cost_cvx),
        label=r"Obj-Obj$^*$ (accel)",
        color="green",
    )
    plt.plot(tps_accel, label=r"$\|P-P^*\|_\infty$ (accel)", color="blue")
    plt.plot(kkts_accel, label="KKT (accel)", color="red")

    plt.plot(
        np.abs(objs - cost_cvx),
        marker="x",
        label=r"Obj-Obj$^*$",
        color="green",
        ms=4,
    )
    plt.plot(tps, label=r"$\|P-P^*\|_\infty$", color="blue", marker="x", ms=4)
    plt.plot(kkts, label="KKT", color="red", marker="x", ms=4)
    plt.legend()
    plt.xlabel("Iteration")
    plt.yscale("log")
    # plt.title(f"PDHG convergence (n={n})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
