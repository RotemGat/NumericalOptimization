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

QP_INIT_XO = np.array([0.1, 0.2, 0.7], dtype=np.float64)
LP_INIT_XO = np.array([0.5, 0.75], dtype=np.float64)


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
