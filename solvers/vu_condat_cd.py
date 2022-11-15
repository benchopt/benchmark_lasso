from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from cd_solver.sklearn_api import Lasso


class Solver(BaseSolver):
    name = 'vu-condat-cd'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/Badr-MOUFAD/Fercoq-Bianchi-solver.git@sklearn-api'
    ]

    references = [
        "Olivier Fercoq and Pascal Bianchi, "
        "'A coordinate descent primal-dual algorithm with large step size and possibly non separable functions', "
        "SIAM Journal on Optimization, 29(1), 100-134, "
        "https://arxiv.org/pdf/1508.04625.pdf, "
        "code: https://github.com/Badr-MOUFAD/Fercoq-Bianchi-solver",
    ]

    parameters = {
        'smooth_formulation': [True, False]
    }

    def __init__(self, smooth_formulation):
        self.smooth_formulation = smooth_formulation

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y = X, y
        self.lmbd = lmbd

        # when smooth_formulation=True, the estimator considers the datafit term smooth
        # and uses it gradient to find the min. Otherwise, estimator considers
        # the datafit non-smooth and uses it prox operator instead
        self.estimator = Lasso(
            alpha=lmbd,
            smooth_formulation=self.smooth_formulation
        )

    def run(self, n_iter):
        X, y = self.X, self.y

        if n_iter == 0:
            self.w = np.zeros(X.shape[1])
            return

        self.estimator.max_iter = n_iter
        self.estimator.fit(X, y)
        self.w = self.estimator.coef_.flatten()

    def get_result(self):
        return self.w
