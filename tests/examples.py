import numpy as np
import math


def objective_quadratic_function(q: np.ndarray, x: np.ndarray, hessian: bool = False) -> \
        (float, np.ndarray, np.ndarray):
    f = np.dot(x, q.dot(x))
    g = 2 * q.dot(x)
    h = 2 * q if hessian else None
    return f, g, h


def q_circle(x: np.array, hessian: bool = False):
    q = np.array([[1, 0], [0, 1]])
    return objective_quadratic_function(q, x, hessian)


def q_aligned_ellipses(x: np.array, hessian: bool = False):
    q = np.array([[1, 0], [0, 100]])
    return objective_quadratic_function(q, x, hessian)


def q_rotated_ellipses(x: np.array, hessian: bool = False):
    q1 = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    q2 = np.array([[1, 0], [0, 100]])
    q = q1.T @ q2 @ q1
    return objective_quadratic_function(q, x, hessian)


def rosenbrock(x: np.array, hessian: bool = False):
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2
    g = np.array([400 * x[0] ** 3 - 400 * x[0] * x[1] + 2 * x[0] - 2,
                  200 * (x[1] - x[0] ** 2)])
    h = np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
                 [-400 * x[0], 200]]) if hessian else None
    return f, g.T, h


def linear_function(x: np.array, hessian: bool = False):
    a = np.array([1, 2])
    f = a.T @ x
    g = a
    h = np.array([0, 0]) if hessian else None
    return f, g.T, h


def exp_comb_function(x: np.array, hessian: bool = False):
    f = math.e ** (x[0] + 3 * x[1] - 0.1) + math.e ** (x[0] - 3 * x[1] - 0.1) + math.e ** (-x[0] - 0.1)
    g = np.array([math.e ** (x[0] + 3 * x[1] - 0.1) + math.e ** (x[0] - 3 * x[1] - 0.1) - math.e ** (-x[0] - 0.1),
                  3 * math.e ** (x[0] + 3 * x[1] - 0.1) - 3 * math.e ** (x[0] - 3 * x[1] - 0.1)])
    h = np.array([[math.e ** (x[0] + 3 * x[1] - 0.1) + math.e ** (x[0] - 3 * x[1] - 0.1) + math.e ** (-x[0] - 0.1), 0],
                  [0, 9 * math.e ** (x[0] + 3 * x[1] - 0.1) + 9 * math.e ** (x[0] - 3 * x[1] - 0.1)]]) if hessian else None
    return f, g.T, h
