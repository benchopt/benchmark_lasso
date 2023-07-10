from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    import blitzl1


class Solver(BaseSolver):
    """T. B. Johnson and C. Guestrin, "Blitz: A Principled Meta-Algorithm for
    Scaling Sparse Optimization", ICML 2015
    """
    name = 'Blitz'
    sampling_strategy = 'iteration'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/tbjohns/blitzl1.git@master'
    ]

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        blitzl1.set_use_intercept(fit_intercept)
        blitzl1.set_tolerance(0)
        self.problem = blitzl1.LassoProblem(self.X, self.y)

    def get_next(self, previous):
        "Linear growth for n_iter."
        return previous + 1

    def run(self, n_iter):
        sol = self.problem.solve(self.lmbd, max_iter=n_iter)
        coef = sol.x
        if self.fit_intercept:
            coef = np.r_[coef, sol.intercept]
        self.coef = coef

    def get_result(self):
        return self.coef
