
class InteriorPointSolver:
    def __init__(self, init_t: int = 1, mu: int = 10):
        self.init_t = init_t
        self.mu = mu

    def minimize(self, func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
        """ Minimize the function func subject to the list of inequality constraints specified by ineq_constraints and to the affine
        equality constraints specified by eq_constraints_mat and the right hand side vector eq_constraints_rhs """