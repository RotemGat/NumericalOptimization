from unittest import TestCase
from examples import *
from unconstrained_min import LineSearchMinimization
from utils import *


class TestUnconstrainedMin(TestCase):
    def setUp(self):
        self.gradient_descent_min = LineSearchMinimization(min_method=MinimizationMethod.GradientDescent)
        self.newton_min = LineSearchMinimization(min_method=MinimizationMethod.Newton)
        self.save_flag: bool = True

    def base_func(self, example_func, example_name: str, x0=INIT_XO, max_iters=MAX_ITERS, x_lim=(-10, 10), y_lim=(-10, 10),
                  is_rosenbrock: bool = False, close_up_factor: int = 2):
        x0 = INIT_X0_ROSENBROCK if is_rosenbrock else x0

        gradient_descent_min = self.gradient_descent_min.minimize(f=example_func, x0=x0, obj_tol=OBJ_TOL,
                                                                  param_tol=STEP_TOL,
                                                                  max_iter=max_iters if not is_rosenbrock else MAX_ITERS_GD_ROSENBROCK)
        newton_min = self.newton_min.minimize(f=example_func, x0=x0, obj_tol=OBJ_TOL,
                                              param_tol=STEP_TOL, max_iter=max_iters)
        labels = ['Gradient Descent', 'Newton']

        plot_contour_lines(objective_func=example_func, x_history_list=[gradient_descent_min[2], newton_min[2]],
                           labels=labels, example_name=example_name, x_lim=x_lim, y_lim=y_lim, close_up_factor=close_up_factor,
                           save_flag=self.save_flag)
        plot_function_values(f_values_list=[gradient_descent_min[3], newton_min[3]], labels=labels,
                             example_name=example_name, save_flag=self.save_flag)

    def test_q_circle(self):
        self.base_func(example_func=q_circle, example_name='Quadratic Circle')

    def test_q_aligned_ellipses(self):
        self.base_func(q_aligned_ellipses, example_name='Quadratic Aligned Ellipses', x_lim=(-100, 100), y_lim=(-100, 100), close_up_factor=50)

    def test_q_rotated_ellipses(self):
        self.base_func(q_rotated_ellipses, example_name='Quadratic Rotated Ellipses', x_lim=(-100, 100), y_lim=(-100, 100), close_up_factor=50)

    def test_rosenbrock(self):
        self.base_func(example_func=rosenbrock, example_name='Rosenbrock', is_rosenbrock=True, x_lim=(-10, 10), y_lim=(-10, 10), close_up_factor=2)

    def test_linear_func(self):
        self.base_func(example_func=linear_function, example_name='Linear Function', x_lim=(-1000, 1000), y_lim=(-1000, 1000), close_up_factor=10)

    def test_exp_comb_func(self):
        self.base_func(example_func=exp_comb_function, example_name='Exponents Combination Function', x_lim=(-700, 700), y_lim=(-700, 700),
                       close_up_factor=200)
