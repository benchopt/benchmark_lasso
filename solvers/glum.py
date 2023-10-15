import warnings
from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    from glum import GeneralizedLinearRegressor
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'glum'

    install_cmd = 'conda'
    requirements = ['glum']
    stopping_criterion = SufficientProgressCriterion(
        patience=5, strategy='tolerance')

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept
        n_samples = self.X.shape[0]
        # glum's stopping criterion is on the L1 norm of the minimal norm
        # subgradient of the objective. All coefs are 0 when starting, so
        # anything in [-alpha, alpha] is a subgradient of the penalty.
        # The minimal norm subgradient of the objective is thus
        # soft_thresholding(X.T @ y / len(y), alpha), and its L1 norm is:
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

    def skip(self, X, y, lmbd, fit_intercept):
        # glum instantiates a Hessian of size (n_features, n_features)
        if isinstance(X, LinearOperator):
            return True, f"{self.name} does not handle implicit operator"

        if X.shape[1] > 20_000:
            return True, "glum does not support n_features >= 20000"
        return False, None

    def run(self, tol):
        self.model.gradient_tol = self.tol_scale * tol
        self.model.fit(self.X, self.y)

    def get_result(self):
        beta = self.model.coef_.flatten()
        if self.fit_intercept:
            beta = np.r_[beta, self.model.intercept_]
        return dict(beta=beta)
