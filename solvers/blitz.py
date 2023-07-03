from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from scipy.sparse.linalg import LinearOperator
    import blitzl1


class Solver(BaseSolver):
    name = 'Blitz'
    stopping_strategy = 'iteration'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/tbjohns/blitzl1.git@master'
    ]
    references = [
        'T. B. Johnson and C. Guestrin, "Blitz: A Principled Meta-Algorithm '
        'for Scaling Sparse Optimization", ICML, '
        'vol. 37, pp. 1171-1179 (2015)'
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        if isinstance(X, LinearOperator):
            return True, f"{self.name} does not handle implicit operator"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd

        blitzl1.set_use_intercept(False)
        blitzl1.set_tolerance(0)
        self.problem = blitzl1.LassoProblem(self.X, self.y)

    def get_next(self, previous):
        "Linear growth for n_iter."
        return previous + 1

    def run(self, n_iter):
        self.coef_ = self.problem.solve(self.lmbd, max_iter=n_iter).x

    def get_result(self):
        return self.coef_.flatten()
