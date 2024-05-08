import numpy as np

from utils import *


class LineSearchMinimization:
    def __init__(self, min_method: MinimizationMethod, alpha: float = ALPHA, wolfe_const: float = WOLFE_CONST,
                 backtracking_const: float = BACKTRACK_CONST):
        self.min_method = min_method
        self.x_history = []
        self.f_history = []
        self.alpha = alpha
        self.c = wolfe_const
        self.backtracking_const = backtracking_const

    def minimize(self, f, x0, obj_tol: float, param_tol: float, max_iter: int):
        match self.min_method:
            case MinimizationMethod.Newton:
                return self._general_min(func=f, x0=x0, obj_tol=obj_tol, param_tol=param_tol, max_iter=max_iter,
                                         search_direction=self._get_newton)
            case MinimizationMethod.GradientDescent:
                return self._general_min(func=f, x0=x0, obj_tol=obj_tol, param_tol=param_tol, max_iter=max_iter,
                                         search_direction=self._get_steepest_descent)

    def _save_history(self, i: int, x0: np.ndarray, fx: np.ndarray):
        self.x_history.append(x0)
        self.f_history.append(fx)
        print(f'Iteration: i={i}, Current location: x_i={x0}, Objective value: {fx}')

    def _general_min(self, func, x0, obj_tol: float, param_tol: float, max_iter: int, search_direction):
        success = False
        is_hessian = True if self.min_method == MinimizationMethod.Newton else False

        f, g, h = func(x0, is_hessian)
        i, x, x_new = 0, x0, x0
        self._save_history(i, x, f)

        try:
            while not success and i < max_iter:
                p = search_direction(x, func)
                alpha = self._backtracking(func, x, f, p)
                x_new = x + p * alpha
                f_new, g_new, h_new = func(x_new, is_hessian)
                self._save_history(i, x_new, f_new)
                lambda_squared = 0 if not is_hessian else (0.5 * p.T @ (h_new @ p)) ** 0.5
                success = (self.min_method == MinimizationMethod.GradientDescent and np.abs(f - f_new) < obj_tol) or \
                          (self.min_method == MinimizationMethod.Newton and lambda_squared < obj_tol) or \
                          (np.linalg.norm(x - x_new) < param_tol)
                x, f = x_new, f_new
                i += 1
            print(f'Line search result: {success}')
            return x, f, self.x_history, self.f_history, success
        except Exception as e:
            print('An error occurred')
            print(e)
            return x, f, self.x_history, self.f_history, False

    def _backtracking(self, func, x, f, p):
        alpha, c, rho = self.alpha, self.c, self.backtracking_const
        while func(x + alpha * p, False)[0] > f + c * alpha * np.dot(-p, p):
            alpha = rho * alpha
        return alpha

    @staticmethod
    def _get_steepest_descent(x, objective_function):
        return -objective_function(x, False)[1]

    @staticmethod
    def _get_newton(x, objective_function):
        f, g, h = objective_function(x, True)
        try:
            return -np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            raise Warning('Could not solve Newtons method due to np.linalg.LinAlgError')





