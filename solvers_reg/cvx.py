import cvxpy as cp
import numpy as np
import numpy.typing as npt


class Cvx:
    def __init__(
        self,
        cost_matrix: npt.NDArray[np.float64],
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        n: None,
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

    def solve(
        self,
        eps: float = 1e-15,
        reg="quad",
        p: int = 2,
        max_iter: int = 100_000,
        eps_reg=1.0,
    ):
        if reg == "quad":
            P = cp.Variable((self.n, self.n))
            u = np.ones((self.n, 1))
            # b_sqrt = self.beta[:, None] ** 0.5
            if self.alpha is None:
                self.alpha = np.ones(self.n)
            a_sqrt = self.alpha[:, None] ** 0.5
            constraints = [P >= 0, cp.matmul(P.T, u) == self.b[:, None]]
            objective = cp.Minimize(
                cp.sum(cp.multiply(P, self.cost_matrix))
                + 0.5
                * self.tau ** (-1)
                * cp.sum_squares(
                    cp.multiply(cp.matmul(P, u) - self.a[:, None], a_sqrt)
                )
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
        elif reg == "kl":
            P = cp.Variable((self.n, self.n))
            u = np.ones((self.n, 1))
            constraints = [P >= 0, cp.matmul(P.T, u) == self.b[:, None]]
            objective = cp.Minimize(
                cp.sum(cp.multiply(P, self.cost_matrix))
                + self.tau ** (-1)
                * cp.sum(cp.kl_div(cp.matmul(P, u), self.a[:, None]))
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

        elif reg == "full_quad":
            if self.alpha is None:
                self.alpha = np.ones(self.n)
            if self.beta is None:
                self.beta = np.ones(self.n)

            P = cp.Variable((self.n, self.n))
            u = np.ones((self.n, 1))
            constraints = [P >= 0]
            a_sqrt = self.alpha[:, None] ** 0.5
            b_sqrt = self.beta[:, None] ** 0.5
            objective = cp.Minimize(
                cp.sum(cp.multiply(P, self.cost_matrix))
                + 0.5
                * self.tau ** (-1)
                * cp.sum_squares(
                    cp.multiply(cp.matmul(P, u) - self.a[:, None], a_sqrt)
                )
                + 0.5
                * self.tau ** (-1)
                * cp.sum_squares(
                    cp.multiply(cp.matmul(P.T, u) - self.b[:, None], b_sqrt)
                )
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

        elif reg == "full_quad_reg":
            if self.alpha is None:
                self.alpha = np.ones(self.n)
            if self.beta is None:
                self.beta = np.ones(self.n)

            P = cp.Variable((self.n, self.n))
            u = np.ones((self.n, 1))
            constraints = [P >= 0]
            a_sqrt = self.alpha[:, None] ** 0.5
            b_sqrt = self.beta[:, None] ** 0.5
            print("here")
            objective = cp.Minimize(
                cp.sum(cp.multiply(P, self.cost_matrix))
                + 0.5 * eps_reg * cp.sum_squares(P)
                + 0.5
                * self.tau ** (-1)
                * cp.sum_squares(
                    cp.multiply(cp.matmul(P, u) - self.a[:, None], a_sqrt)
                )
                + 0.5
                * self.tau ** (-1)
                * cp.sum_squares(
                    cp.multiply(cp.matmul(P.T, u) - self.b[:, None], b_sqrt)
                )
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

        elif reg == "full_kl":
            P = cp.Variable((self.n, self.n))
            u = np.ones((self.n, 1))
            constraints = [P >= 0]
            objective = cp.Minimize(
                cp.sum(cp.multiply(P, self.cost_matrix))
                + self.tau ** (-1)
                * cp.sum(cp.kl_div(cp.matmul(P, u), self.a[:, None]))
                + self.tau ** (-1)
                * cp.sum(cp.kl_div(cp.matmul(P.T, u), self.b[:, None]))
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

        elif reg == "full_lp":
            if self.alpha is None:
                self.alpha = np.ones(self.n)
            if self.beta is None:
                self.beta = np.ones(self.n)

            P = cp.Variable((self.n, self.n))
            u = np.ones((self.n, 1))
            constraints = [P >= 0]
            a_sqrt = self.alpha[:, None] ** 0.5
            b_sqrt = self.beta[:, None] ** 0.5
            objective = cp.Minimize(
                cp.sum(cp.multiply(P, self.cost_matrix))
                + (self.tau * p) ** (-1)
                * cp.sum(
                    cp.power(
                        cp.multiply(cp.matmul(P, u) - self.a[:, None], a_sqrt),
                        p,
                    )
                )
                + (p * self.tau) ** (-1)
                * cp.sum(
                    cp.power(
                        cp.multiply(
                            cp.matmul(P.T, u) - self.b[:, None], b_sqrt
                        ),
                        p,
                    )
                )
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
