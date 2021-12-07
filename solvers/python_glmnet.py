from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt import stopping_criterion
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:
    import numpy as np
    import glmnet


class Solver(BaseSolver):
    name = "python_glmnet"

    install_cmd = 'conda'
    requirements = ['glmnet']
    references = [
        'J. Friedman, T. J. Hastie and R. Tibshirani, "Regularization paths '
        'for generalized linear models via coordinate descent", '
        'J. Stat. Softw., vol. 33, no. 1, pp. 1-22, NIH Public Access (2010)'
    ]

    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy='tolerance')
    support_sparse = False

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.lmbd_max = max(abs(X.T @ y))
        self.fit_intercept = fit_intercept
        self.clf = glmnet.ElasticNet(
            alpha=1, lambda_path=[self.lmbd_max / len(y), lmbd / len(y)],
            standardize=False, max_iter=1_000_000, fit_intercept=False)

    def run(self, tol):
        self.clf.tol = tol ** 2.3
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return self.clf.coef_
