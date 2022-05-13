from benchopt import BaseSolver, safe_import_context
from benchopt.utils.sys_info import _get_cuda_version

cuda_version = _get_cuda_version()

with safe_import_context() as import_ctx:
    if cuda_version is not None:
        cuda_version = cuda_version.split("cuda_", 1)[1][:4]
    else:
        raise ImportError("cuml solver needs a nvidia GPU.")

    import cudf
    import numpy as np
    from cuml.linear_model import Lasso


class Solver(BaseSolver):
    name = "cuml"

    install_cmd = "conda"
    requirements = [
        "rapidsai::rapids",
        f"nvidia::cudatoolkit={cuda_version}",
        "dask-sql",
    ] if cuda_version is not None else []

    parameters = {
        "solver": [
            "qn",
            "cd"
        ],
    }
    parameter_template = "{solver}"
    support_sparse = False

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.X = cudf.DataFrame(self.X)
        self.y = cudf.Series(self.y)
        self.fit_intercept = fit_intercept

        self.lasso = Lasso(
            fit_intercept=self.fit_intercept,
            alpha=self.lmbd / self.X.shape[0],
            tol=1e-15,
            solver=self.solver,
            verbose=0
        )

    def run(self, n_iter):
        self.lasso.solver_model.max_iter = n_iter
        self.result_lasso = self.lasso.fit(self.X, self.y)

    def get_result(self):
        if self.fit_intercept:
            return np.r_[self.result_lasso.coef_, self.result_lasso.intercept_]
        else:
            return self.result_lasso.coef_
