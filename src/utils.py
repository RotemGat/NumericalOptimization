from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np


class MinimizationMethod(Enum):
    GradientDescent = auto()
    Newton = auto()


WOLFE_CONST = 0.01
BACKTRACK_CONST = 0.5
ALPHA = 1.0
OBJ_TOL = 1e-12
STEP_TOL = 1e-8
MAX_ITERS = 100
MAX_ITERS_GD_ROSENBROCK = 10000
INIT_XO = np.array([1, 1])
INIT_X0_ROSENBROCK = np.array([-1, 2], dtype=np.float64)

INIT_T = 1
MU = 10
EPSILON = 1e-6
QP_INIT_XO = np.array([0.1, 0.2, 0.7], dtype=np.float64)
LP_INIT_XO = np.array([0.5, 0.75], dtype=np.float64)


def backtracking(alpha, c, rho, func, x, f, p, g=np.array([])):
    while func(x + alpha * p, False)[0] > f + c * alpha * (np.dot(-p, p) if not g.size > 0 else g.dot(p)):
        alpha = rho * alpha
    return alpha


def plot_contour_lines(objective_func, x_history_list: list[list[float]], labels: list[str], example_name: str,
                       x_lim: tuple[int, int], y_lim: tuple[int, int], close_up_factor: int = 2, save_flag: bool = False):
    fig, axs = plt.subplots(2, figsize=(6, 6))

    for ix, ax in enumerate(axs):
        x, y = np.linspace(x_lim[0], x_lim[1], 100), np.linspace(y_lim[0], y_lim[1], 100)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        for i in range(len(x)):
            for j in range(len(y)):
                z[i, j], a, b = objective_func(np.array([x[i, j], y[i, j]]))
        ax.contour(x, y, z, levels=10)

        colors = ['r', 'b']
        for x_history, label, color in zip(x_history_list, labels, colors):
            x_hist, y_hist = zip(*x_history)
            ax.plot(x_hist, y_hist, label=label, color=color)
        title = f'Contour lines of {example_name}' if ix == 0 else f'Contour lines of {example_name} - close up'
        ax.set_title(label=title)
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if ix == 0:
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)
        else:
            close_up = 1 / close_up_factor
            ax.set_xlim(x_lim[0] * close_up, x_lim[1] * close_up)
            ax.set_ylim(y_lim[0] * close_up, y_lim[1] * close_up)
    plt.tight_layout()
    plt.show()
    if save_flag:
        fig.savefig(f'contour_lines_{example_name}.png')


def plot_function_values(f_values_list: list[list[float]], labels: list[str], example_name: str, save_flag: bool = False):
    fig = plt.figure()
    for f_values, label in zip(f_values_list, labels):
        plt.plot(range(len(f_values)), f_values, label=label)
    plt.title(label=f'Function values of {example_name}')
    plt.legend()
    plt.xlabel('Iteration i')
    plt.ylabel('Function Value')
    if save_flag:
        fig.savefig(f'function_values_{example_name}.png')
    plt.show()


def plot_feasible_set_2d(path, title, save_flag: bool = False):
    fig, ax = plt.subplots(1, 1)
    x_coords, y_coords = zip(*path)
    ax.plot(x_coords, y_coords, c='black', label='path')
    ax.scatter(path[-1][0], path[-1][1], s=60, color='black', marker='o', label='final candidate')

    x = np.linspace(-1, 3, 300)
    y = np.linspace(-2, 2, 300)
    constraints = {'y=0': (x, x * 0),
                   'y=1': (x, x * 0 + 1),
                   'x=2': (y * 0 + 2, y),
                   'y=-x+1': (x, -x + 1)}

    colors = ['pink', 'purple', 'blue', 'skyblue']
    for constraint, point, color in zip(constraints.keys(), constraints.values(), colors[:len(constraints) + 1]):
        ax.plot(point[0], point[1], label=constraint, color=color)

    ax.fill([0, 2, 2, 1], [1, 1, 0, 0], label='feasible region', alpha=0.7, color='lightgray')
    ax.set_title(f'Feasible region and Path 2D of {title}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()
    if save_flag:
        fig.savefig(f'Feasible_region_3D_{title}.png')


def plot_feasible_regions_3d(path, title, save_flag: bool = False):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    path = np.array(path)

    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color='lightgray', alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path')
    ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=50, c='black', marker='o', label='Final candidate')
    ax.set_title(f"Feasible Regions and Path 3D of {title}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    ax.view_init(45, 45)
    plt.show()
    if save_flag:
        fig.savefig(f'Feasible_region_2D_{title}.png')
