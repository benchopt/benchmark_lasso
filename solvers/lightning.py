from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from lightning.regression import CDRegressor


# TODO: lightning always fit an intercept
#       it is thus not optimizing the same cost function
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
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
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
            beta = np.r_[beta, self.clf.intercept_]
        return beta
