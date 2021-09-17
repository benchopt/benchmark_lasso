from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import cvxpy as cp


class Solver(BaseSolver):
    name = 'cvxpy'

    install_cmd = 'conda'
    requirements = ['cvxpy']
    references = [
        'S. Diamond and S. Boyd, "CVXPY: A Python-embedded modeling language '
        'for convex optimization", J. Mach. Learn. Res., vol. 17, no. 83, '
        'pp. 1-5, (2016)'
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"
        if X.shape[1] > 50000:
            return True, "problem too large."

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        n_features = self.X.shape[1]
        self.beta = cp.Variable(n_features)

        loss = 0.5 * cp.norm2(self.y - cp.matmul(self.X, self.beta))**2
        self.problem = cp.Problem(cp.Minimize(
            loss + self.lmbd * cp.norm(self.beta, 1)
        ))

        # Hack cvxpy to be able to retrieve a suboptimal solution when
        # reaching max_iter
        cp.reductions.solvers.conic_solvers.ECOS.STATUS_MAP[-1] = (
            'optimal_inaccurate'
        )

        cp.settings.ERROR = ['solver_error']

        self.problem.solve(max_iters=1, verbose=False)

    def run(self, n_iter):
        self.problem.solve(
            max_iters=n_iter, verbose=True, reltol=1e-15
        )

    def get_result(self):
        return self.beta.value.flatten()
