from benchopt import BaseSolver, safe_import_context
from benchopt.utils.sys_info import _get_cuda_version


with safe_import_context() as import_ctx:
    from snapml import LinearRegression
    import numpy as np


class Solver(BaseSolver):
    name = "snapml"

    install_cmd = "conda"
    requirements = ["pip:snapml"]

    parameters = {"gpu": [False, True]}

    def skip(self, X, y, lmbd, fit_intercept):
        if self.gpu and _get_cuda_version() is None:
            return True, "snapml[gpu=True] needs a GPU to run"
        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        self.clf = LinearRegression(
            fit_intercept=self.fit_intercept,
            regularizer=self.lmbd,
            penalty="l1",
            tol=1e-12,
            dual=False,
            use_gpu=self.gpu,
        )

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1] + self.fit_intercept])
            return

        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)
        coef = self.clf.coef_.flatten()
        if self.fit_intercept:
            coef = np.r_[coef, self.clf.intercept_]
        self.coef = coef

    def get_result(self):
        return self.coef
