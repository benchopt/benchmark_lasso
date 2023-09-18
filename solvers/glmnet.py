from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import INFINITY
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri, packages
    from benchopt.helpers.r_lang import import_rpackages

    # Setup the system to allow rpy2 running
    numpy2ri.activate()
    import_rpackages('glmnet')


class Solver(BaseSolver):
    """J. Friedman, T. J. Hastie and R. Tibshirani, "Regularization paths for
     generalized linear models via coordinate descent", J. Stat. Softw.,
     vol. 33, no. 1, pp. 1-22, 2010"""
    name = "glmnet"

    install_cmd = 'conda'
    requirements = ['r-base', 'rpy2', 'r-glmnet', 'r-matrix']
    references = [

    ]
    support_sparse = True

    # We use the tolerance strategy because if maxit is too low and glmnet
    # convergence check fails, it returns an empty model
    stopping_criterion = SufficientProgressCriterion(
        patience=7, eps=1e-38, strategy='tolerance'
    )

    def set_objective(self, X, y, lmbd, fit_intercept):
        if sparse.issparse(X):
            r_Matrix = packages.importr("Matrix")
            X = X.tocoo()
            self.X = r_Matrix.sparseMatrix(
                i=robjects.IntVector(X.row + 1),
                j=robjects.IntVector(X.col + 1),
                x=robjects.FloatVector(X.data),
                dims=robjects.IntVector(X.shape)
            )
        else:
            self.X = X
        self.y, self.lmbd = y, lmbd
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

        self.glmnet_fit = self.glmnet(
            self.X, self.y, intercept=self.fit_intercept,
            standardize=False, maxit=maxit, thresh=thresh, **fit_dict)

    def get_result(self):
        results = dict(zip(self.glmnet_fit.names, list(self.glmnet_fit)))
        as_matrix = robjects.r['as']
        coefs = np.array(as_matrix(results["beta"], "matrix"))
        beta = coefs.flatten()

        return dict(
            beta=np.r_[beta, results["a0"]] if self.fit_intercept else beta
        )
