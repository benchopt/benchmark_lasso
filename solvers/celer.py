import warnings

from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from celer import Lasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'Celer'
    stop_strategy = 'iteration'

    install_cmd = 'conda'
    requirements = ['pip:celer']
    references = [
        'M. Massias, A. Gramfort and J. Salmon, ICML, '
        '"Celer: a Fast Solver for the Lasso with Dual Extrapolation", '
        'vol. 80, pp. 3321-3330 (2018)'
    ]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        n_samples = self.X.shape[0]
        self.lasso = Lasso(
            alpha=self.lmbd / n_samples, max_iter=1, max_epochs=50_000,
            tol=1e-12, prune=True, fit_intercept=False, normalize=False,
            warm_start=False, positive=False, verbose=False,
        )

    def run(self, n_iter):
        self.lasso.max_iter = n_iter + 1
        self.lasso.fit(self.X, self.y)

    def get_result(self):
        return self.lasso.coef_.flatten()
