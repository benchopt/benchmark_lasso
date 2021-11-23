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
    stop_strategy = 'tolerance'

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept
        n_samples = self.X.shape[0]
        # all coefs are 0 when starting, so we can take anything in
        # [-alpha, alpha] as subgradient of penalty. The minimal norm
        # subgradient of the objective is thus ST(grad, alpha), and its
        # L1 norm is:
        self.tol_scale = np.sum(
            np.maximum(np.abs(X.T @ y / len(y)) - lmbd / n_samples, 0))
        self.model = GeneralizedLinearRegressor(
            family='gaussian',
            l1_ratio=1.0,
            alpha=self.lmbd / n_samples,
            fit_intercept=fit_intercept,
            gradient_tol=1,
        )
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

    def run(self, tol):
        # support sparse while avoiding MemoryError if n_features too large,
        # as glum intanciates a Hessian
        if self.X.shape[1] < 20_000:
            self.model.gradient_tol = self.tol_scale * tol
            self.model.fit(self.X, self.y)

    def get_result(self):
        beta = self.model.coef_.flatten()
        if self.fit_intercept:
            beta = np.r_[beta, self.model.intercept_]
        return beta
