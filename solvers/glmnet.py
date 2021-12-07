from benchopt import BaseSolver, safe_import_context
from benchopt.runner import INFINITY
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:
    import numpy as np

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    from benchopt.helpers.r_lang import import_rpackages

    # Setup the system to allow rpy2 running
    numpy2ri.activate()
    import_rpackages('glmnet')


class Solver(BaseSolver):
    name = "glmnet"

    install_cmd = 'conda'
    requirements = ['r-base', 'rpy2', 'r-glmnet']
    references = [
        'J. Friedman, T. J. Hastie and R. Tibshirani, "Regularization paths '
        'for generalized linear models via coordinate descent", '
        'J. Stat. Softw., vol. 33, no. 1, pp. 1-22, NIH Public Access (2010)'
    ]
    support_sparse = False

    stopping_criterion = SufficientProgressCriterion(
        patience=7, eps=1e-38, strategy='tolerance')

    def skip(self, X, y, lmbd, fit_intercept):
        # XXX - glmnet support intercept, adapt the API
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        self.glmnet = robjects.r['glmnet']

    def run(self, tol):
        maxit = 0 if tol == INFINITY else 1_000_000
        fit_dict = {"lambda": self.lmbd / len(self.y)}

        glmnet_fit = self.glmnet(self.X, self.y, intercept=False,
                                 standardize=False, maxit=maxit,
                                 thresh=tol ** 2.3, **fit_dict)
        results = dict(zip(glmnet_fit.names, list(glmnet_fit)))
        as_matrix = robjects.r['as']
        coefs = np.array(as_matrix(results["beta"], "matrix"))
        self.w = coefs.flatten()

    def get_result(self):
        return self.w
