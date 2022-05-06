import warnings
from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.linear_model import LassoLars
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "lars"

    install_cmd = "conda"
    requirements = ["scikit-learn"]
    references = [
        "B. Efron, T. Hastie, I. Johnstone, R. Tibshirani"
        '"Least Angle Regression", Annals of Statistics, '
        " vol. 32 (2), pp. 407-499 (2004)"
    ]

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        n_samples = self.X.shape[0]

        self.clf = LassoLars(
            alpha=self.lmbd / n_samples, fit_intercept=fit_intercept,
            normalize=False,
        )

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        beta = self.clf.coef_.flatten()
        if self.fit_intercept:
            beta = np.r_[beta, self.clf.intercept_]
        return beta
