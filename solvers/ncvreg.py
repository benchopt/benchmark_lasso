from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.helpers.r_lang import import_rpackages
    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    from scipy import sparse

    # Setup the system to allow rpy2 running
    numpy2ri.activate()
    import_rpackages("ncvreg")


class Solver(BaseSolver):
    name = "ncvreg"

    install_cmd = "conda"
    requirements = ["r-base", "rpy2", "r-ncvreg"]
    references = [
        'P. Breheny and J. Huang, "Coordinate descent algorithms for nonconvex'
        "penalized regression, with applications to biological feature"
        'selection," The Annals of Applied Statistics, vol. 5, no. 1, pp.'
        "232–253, Mar. 2011, doi: 10.1214/10-AOAS388."
    ]

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.lmbd = lmbd
        self.fit_intercept = fit_intercept
        self.n, self.p = X.shape

        penalty_factor = np.ones(self.p)

        # NOTE(jolars): To fit the intercept, we add a column of ones at the
        # end of X and set its penalty factor to zero.
        self.X = X

        if self.fit_intercept:
            X = np.hstack((X, np.ones((self.n, 1))))
            penalty_factor = np.hstack((penalty_factor, np.array([0.0])))

        self.penalty_factor = robjects.vectors.FloatVector(penalty_factor)

        self.Xmod = robjects.r.matrix(X, X.shape[0], X.shape[1])
        self.y = robjects.vectors.FloatVector(y)

        # Standardization of X cannot be turned off in standard interface to
        # ncvreg, so we have to use ncvfit directly instead
        self.ncvfit = robjects.r["ncvfit"]

    def skip(self, X, y, lmbd, fit_intercept):
        if sparse.issparse(X):
            return True, "ncvreg does not support sparse X"

        return False, None

    def run(self, n_iter):
        fit_dict = {
            "max.iter": n_iter,
            "lambda": self.lmbd / self.n,
            "penalty.factor": self.penalty_factor,
        }

        self.fit = self.ncvfit(self.Xmod, self.y, penalty="lasso", eps=1e-15, **fit_dict)

    def get_result(self):
        results = dict(zip(self.fit.names, list(self.fit)))

        return results["beta"]
