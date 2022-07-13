from benchopt import BaseSolver, safe_import_context
from benchopt.helpers.requires_gpu import requires_gpu
from benchopt.stopping_criterion import SufficientProgressCriterion

cuda_version = None
with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    cuda_version = requires_gpu()

    if cuda_version is not None:

        import cudf
        import cupy as cp
        import cupyx.scipy.sparse as cusparse
        from cuml.linear_model import Lasso


class Solver(BaseSolver):
    name = "cuml"

    install_cmd = "conda"
    requirements = [
        "rapidsai::rapids",
        f"nvidia::cudatoolkit={cuda_version}",
        "dask-sql", "cupy"
    ] if cuda_version is not None else []

    parameters = {
        "solver": [
            "qn",
            "cd"
        ],
    }
    parameter_template = "{solver}"
    references = [
        "S. Raschka, J. Patterson and C. Nolet, "
        '"Machine Learning in Python: Main developments and technology trends '
        'in data science, machine learning, and artificial intelligence", '
        "arXiv preprint arXiv:2002.04803 (2020)"
    ]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-12, patience=5, strategy='iteration'
    )

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        if sparse.issparse(X):
            if sparse.isspmatrix_csc(X):
                self.X = cusparse.csc_matrix(X)
            elif sparse.isspmatrix_csr(X):
                self.X = cusparse.csr_matrix(X)
            else:
                raise ValueError("Non suported sparse format")
        else:
            self.X = cudf.DataFrame(self.X.astype(np.float32))
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
        if isinstance(self.lasso.coef_, cp.ndarray):
            coef = self.lasso.coef_.get().flatten()
            if self.lasso.fit_intercept:
                coef = np.r_[coef, self.lasso.intercept_.get()]
        else:
            coef = self.lasso.coef_.to_numpy().flatten()
            if self.lasso.fit_intercept:
                coef = np.r_[coef, self.lasso.intercept_.to_numpy()]

        return coef
