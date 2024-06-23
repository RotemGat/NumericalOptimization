from unittest import TestCase
from constrained_min import InteriorPointSolver
from examples import qp, qp_ineq_constraints, lp, lp_ineq_constraints
from utils import *


class TestConstrainedMin(TestCase):

    def base_func(self, func, ineq_constraints, eq_mat, eq_r, init_x0, name, plot_dim):
        solver = InteriorPointSolver()
        x, f = solver.minimize(func, ineq_constraints, eq_mat, eq_r, init_x0)
        print(f'Final candidate: {x}')
        print(f'Objective function of final candidate: {f}')
        print(f'Inequality constraints of final candidate: {[con(x) for con in ineq_constraints]}')
        final_path = solver.x_history

        plot_function_values(f_values_list=[solver.f_history], labels=[name], example_name=name)
        if plot_dim == '3D':
            plot_feasible_regions_3d(final_path, name)
        else:
            plot_feasible_set_2d(final_path, name)

    def test_qp(self):
        self.base_func(qp, qp_ineq_constraints, np.array([1, 1, 1]).reshape(1, 3), 0, QP_INIT_XO, 'QP', '3D')

    def test_lp(self):
        self.base_func(lp, lp_ineq_constraints, np.array([]), np.array([]), LP_INIT_XO, 'LP', '2D')


