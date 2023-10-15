from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import warnings
    import numpy as np
    from scipy.sparse.linalg import LinearOperator
    from skglm import Lasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    """Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel and
    M. Massias, "Beyond L1: Faster and Better Sparse Models with skglm",
    NeurIPS 2022.
    """
    name = "skglm"
    sampling_strategy = "iteration"

    install_cmd = 'conda'
    requirements = [
        'pip:skglm'
    ]

    references = [
        'Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel'
        'and M. Massias'
        '"Beyond L1: Faster and Better Sparse Models with skglm", '
        'https://arxiv.org/abs/2204.07826'
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"
        if isinstance(X, LinearOperator):
            return True, f"{self.name} does not handle implicit operator"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        n_samples = self.X.shape[0]
        self.lasso = Lasso(
            alpha=self.lmbd / n_samples, max_iter=1, max_epochs=50_000,
            tol=1e-12, fit_intercept=fit_intercept, warm_start=False,
        )

        # Cache Numba compilation
        self.run(1)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1] + self.fit_intercept])
        else:
            self.lasso.max_iter = n_iter
            self.lasso.fit(self.X, self.y)

            coef = self.lasso.coef_.flatten()
            if self.fit_intercept:
                coef = np.r_[coef, self.lasso.intercept_]
            self.coef = coef

    def get_next(self, stop_val):
        return stop_val + 1

    def get_result(self):
        return dict(beta=self.coef)
