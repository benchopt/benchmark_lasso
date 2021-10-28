import warnings
from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from glum import GeneralizedLinearRegressor
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'glum'

    install_cmd = 'conda'
    requirements = ['glum']

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept
        n_samples = self.X.shape[0]
        self.model = GeneralizedLinearRegressor(
            family='gaussian',
            l1_ratio=1.0,
            alpha=self.lmbd / n_samples,
            fit_intercept=fit_intercept,
            gradient_tol=1e-8,
        )
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

    def run(self, n_iter):
        self.model.max_iter = n_iter + 1
        self.model.fit(self.X, self.y)

    def get_result(self):
        beta = self.model.coef_.flatten()
        if self.fit_intercept:
            beta = np.r_[beta, self.model.intercept_]
        return beta
