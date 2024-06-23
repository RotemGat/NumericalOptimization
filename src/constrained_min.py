import math
from typing import Any

from utils import *


class InteriorPointSolver:
    def __init__(self, init_t: int = INIT_T, mu: int = MU,  alpha: float = ALPHA, wolfe_const: float = WOLFE_CONST,
                 backtracking_const: float = BACKTRACK_CONST):
        self.init_t = init_t
        self.mu = mu
        self.alpha = alpha
        self.c = wolfe_const
        self.backtracking_const = backtracking_const
        self.x_history = []
        self.f_history = []

    def minimize(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, outer_loop_max_iter=100, inner_loop_max_iter=10,
                 epsilon=1e-6, obj_tol=1e-8, param_tol=1e-6):
        """ Minimize the function func subject to the list of inequality constraints specified by ineq_constraints and to the affine
        equality constraints specified by eq_constraints_mat and the right hand side vector eq_constraints_rhs """

        x = x0
        t = self.init_t
        success = False
        phi, phi_tag, phi_hessian = self._get_log_barrier(x, ineq_constraints)
        f, g, h = self.barrier_func(x, func, t, phi, phi_tag, phi_hessian)

        outer_i = 0
        while outer_i < outer_loop_max_iter and len(ineq_constraints) / t > epsilon:
            for inner_iteration in range(inner_loop_max_iter):
                p = self._solve_kkt_system(g, h, eq_constraints_mat, eq_constraints_rhs)

                alpha = backtracking(self.alpha, self.c, self.backtracking_const, func, x, f, p, g)
                x_new = x + p * alpha

                phi, phi_tag, phi_hessian = self._get_log_barrier(x_new, ineq_constraints)
                f_new, g_new, h_new = self.barrier_func(x_new, func, t, phi, phi_tag, phi_hessian)
                lambda_squared = 0.5 * ((p.T @ (h_new @ p)) ** 0.5)

                # np.abs(f_new - f) < obj_tol or np.linalg.norm(x - x_new) < param_tol
                if lambda_squared < obj_tol:
                    success = True
                    x = x_new
                    break
                x, f, g, h = x_new, f_new, g_new, h_new

            self._save_history(outer_i, x, f)
            t *= self.mu
            if success:
                break
            outer_i += 1

        return x, func(x)[0]

    def _save_history(self, outer_i: int, x0: np.ndarray, fx: np.ndarray):
        self.x_history.append(x0)
        self.f_history.append(fx)
        print(f'Outer Iteration: i={outer_i} , Current location: x_i={x0}, Objective value: {fx}')

    @staticmethod
    def _solve_kkt_system(g, h, eq_constraints_mat, eq_constraints_rhs):
        if eq_constraints_mat.size > 0:
            # Construct the block matrix for KKT system
            upper_block = np.concatenate([h, eq_constraints_mat.T], axis=1)
            lower_block = np.concatenate([eq_constraints_mat, np.zeros((eq_constraints_mat.shape[0], eq_constraints_mat.shape[0]))], axis=1)
            block_matrix = np.concatenate([upper_block, lower_block], axis=0)
            rhs = np.concatenate([-g, np.zeros(block_matrix.shape[0] - len(g))]).T
        else:
            block_matrix = h
            rhs = -g.T

        # Solve the KKT system
        solution = np.linalg.solve(block_matrix, rhs)
        p = solution if not eq_constraints_mat.size > 0 else solution[:eq_constraints_mat.shape[1]]
        return p

    @staticmethod
    def _get_log_barrier(x: np.ndarray, ineq_constraints):
        phi = 0
        phi_tag = np.zeros_like(x)
        phi_hessian = np.zeros((len(x), len(x)))

        for func in ineq_constraints:
            f, g, h = func(x, True)
            try:
                phi -= math.log(-f)
            except ValueError:
                pass
            phi_tag -= g / f
            phi_hessian += np.outer(g / f, g / f) / (f ** 2) - h / f

        return phi, phi_tag, phi_hessian

    @staticmethod
    def barrier_func(x, func, t, phi, phi_tag, phi_hessian, hessian=True):
        f, g, h = func(x, hessian)
        f_barrier = t * f + phi
        g_barrier = t * g + phi_tag
        h_barrier = t * h + phi_hessian
        return f_barrier, g_barrier, h_barrier
