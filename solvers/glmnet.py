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
    # We use the tolerance strategy because if maxit is too low and glmnet
    # convergence check fails, it returns an empty model

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
        # Even if maxit=0, glmnet can return non zero coefficients. To get the
        # initial point on the curve, we set tol=0: this way, glmnet
        # convergence check fails, and rather than returning the current
        # iterate, it returns a 0 model.
        if tol == INFINITY:
            maxit = 0
            thresh = 0
        else:
            maxit = 1_000_000
            # we need thresh to decay fast, otherwise the objective curve can
            # plateau before convergence
            thresh = tol ** 1.8

        # We fit with a single lambda because when computing a path, glmnet
        # early stops the path based on an uncontrollable criterion. Fitting
        # with a single lambda may be suboptimal if it is small, but there is
        # no other way to force glmnet to solve for a prescribed lambda.
        fit_dict = {"lambda": self.lmbd / len(self.y)}

        glmnet_fit = self.glmnet(self.X, self.y, intercept=False,
                                 standardize=False, maxit=maxit,
                                 thresh=thresh, **fit_dict)
        results = dict(zip(glmnet_fit.names, list(glmnet_fit)))
        as_matrix = robjects.r['as']
        coefs = np.array(as_matrix(results["beta"], "matrix"))
        self.w = coefs.flatten()

    def get_result(self):
        return self.w
