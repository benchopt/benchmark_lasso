from benchopt import BaseSolver, safe_import_context


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
    name = "glmnet"

    install_cmd = 'conda'
    requirements = ['r-base', 'rpy2', 'r-glmnet', 'r-matrix']
    references = [
        'J. Friedman, T. J. Hastie and R. Tibshirani, "Regularization paths '
        'for generalized linear models via coordinate descent", '
        'J. Stat. Softw., vol. 33, no. 1, pp. 1-22, NIH Public Access (2010)'
    ]
    stop_strategy = 'iteration'
    support_sparse = True

    def skip(self, X, y, lmbd, fit_intercept):
        # XXX - glmnet support intercept, adapt the API
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        if sparse.issparse(X):
            r_Matrix = packages.importr("Matrix")
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

        self.lmbd_max = np.max(np.abs(X.T @ y))
        self.glmnet = robjects.r['glmnet']

    def run(self, n_iter):
        fit_dict = {"lambda.min.ratio": self.lmbd / self.lmbd_max}
        glmnet_fit = self.glmnet(self.X, self.y, intercept=False,
                                 standardize=False, maxit=n_iter,
                                 thresh=1e-14, **fit_dict)
        results = dict(zip(glmnet_fit.names, list(glmnet_fit)))
        as_matrix = robjects.r['as']
        coefs = np.array(as_matrix(results["beta"], "matrix"))
        self.w = coefs[:, -1]

    def get_result(self):
        return self.w
