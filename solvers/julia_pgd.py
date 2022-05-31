from pathlib import Path
from benchopt import safe_import_context

from benchopt.helpers.julia import JuliaSolver
from benchopt.helpers.julia import get_jl_interpreter
from benchopt.helpers.julia import assert_julia_installed

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.sparse import issparse
    from sklearn.linear_model._base import _preprocess_data
    assert_julia_installed()


# File containing the function to be called from julia
JULIA_SOLVER_FILE = str(Path(__file__).with_suffix('.jl'))


class Solver(JuliaSolver):

    # Config of the solver
    name = 'Julia-PGD'
    stopping_strategy = 'iteration'
    requirements = ["scikit-learn"]
    references = [
        'I. Daubechies, M. Defrise and C. De Mol, '
        '"An iterative thresholding algorithm for linear inverse problems '
        'with a sparsity constraint", Comm. Pure Appl. Math., '
        'vol. 57, pp. 1413-1457, no. 11, Wiley Online Library (2004)',
        'A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding '
        'algorithm for linear inverse problems", SIAM J. Imaging Sci., '
        'vol. 2, no. 1, pp. 183-202 (2009)'
    ]
    support_sparse = False

    def skip(self, X, y, lmbd, fit_intercept):
        # XXX - fit intercept is not yet implemented in julia.jl for sparse X
        if fit_intercept and issparse(X):
            return True, \
                f"{self.name} doesn't handle fit_intercept with sparse data",

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        # sklearn way of handling intercept: center y and X for dense data
        if fit_intercept:
            X, y, X_offset, y_offset, _ = _preprocess_data(
                X, y, fit_intercept, return_mean=True, copy=True,
            )
            self.X_offset = X_offset
            self.y_offset = y_offset

        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        jl = get_jl_interpreter()
        self.solve_lasso = jl.include(JULIA_SOLVER_FILE)

    def run(self, n_iter):
        self.beta = self.solve_lasso(self.X, self.y, self.lmbd, n_iter)

        if self.fit_intercept and not issparse(self.X):
            intercept = self.y_offset - self.X_offset @ self.beta
            self.beta = np.r_[self.beta.ravel(), intercept]

    def get_result(self):
        return self.beta.ravel()
