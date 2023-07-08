from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from lightning.regression import CDRegressor


class Solver(BaseSolver):
    name = 'Lightning'

    install_cmd = 'conda'
    requirements = [
        'cython',
        'pip:git+https://github.com/scikit-learn-contrib/lightning.git'
    ]
    references = [
        'M. Blondel, K. Seki and K. Uehara, '
        '"Block coordinate descent algorithms for large-scale sparse '
        'multiclass classification" '
        'Mach. Learn., vol. 93, no. 1, pp. 31-52 (2013)'
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept and sparse.issparse(X):
            return (
                True,
                f"{self.name} doesn't handle fit_intercept with sparse data",
            )

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        # lightning has an attribut intercept_ but it is not handled properly
        # (as it is simply set to zero). For this reason, we handle intercept
        # manually: center y and X beforehand (for dense data only)
        if fit_intercept and not sparse.issparse(self.X):
            self.X_offset = np.average(X, axis=0)
            X -= self.X_offset
            self.y_offset = np.average(y, axis=0)
            y -= self.y_offset

        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        self.clf = CDRegressor(
            loss='squared', penalty='l1', C=.5, alpha=self.lmbd,
            tol=1e-15, random_state=0, permute=False, shrinking=False,
        )

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        beta = self.clf.coef_.flatten()

        if self.fit_intercept:
            intercept = self.y_offset - self.X_offset @ beta
            beta = np.r_[beta, intercept]
        return beta
