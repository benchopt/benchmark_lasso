from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import blitzl1


class Solver(BaseSolver):
    name = 'Blitz'
    stop_strategy = 'tolerance'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/tommoral/blitzl1.git@FIX_setup_deps'
    ]
    references = [
        'T. B. Johnson and C. Guestrin, "Blitz: A Principled Meta-Algorithm '
        'for Scaling Sparse Optimization", ICML, '
        'vol. 37, pp. 1171-1179 (2015)'
    ]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        blitzl1.set_use_intercept(False)
        self.problem = blitzl1.LassoProblem(self.X, self.y)

    def run(self, tolerance):
        blitzl1.set_tolerance(tolerance)
        self.coef_ = self.problem.solve(self.lmbd).x

    def get_result(self):
        return self.coef_.flatten()
