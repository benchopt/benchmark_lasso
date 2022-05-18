from benchopt import safe_import_context
from benchopt.helpers.julia import (
    JuliaSolver,
    assert_julia_installed,
    get_jl_interpreter,
)
from benchopt.runner import INFINITY
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    assert_julia_installed()
    import warnings
    import numpy as np
    from scipy import sparse

    from pathlib import Path
    # File containing the function to be called from julia
    JULIA_SOLVER_FILE = str(Path(__file__).with_suffix(".jl"))


class Solver(JuliaSolver):
    name = "lasso_jl"
    stopping_criterion = SufficientProgressCriterion(
        patience=7, eps=1e-15, strategy="tolerance"
    )
    julia_requirements = [
        "Lasso",
        "PyCall",
        "SparseArrays",
    ]
    references = [
        'J. Friedman, T. J. Hastie and R. Tibshirani, "Regularization paths '
        'for generalized linear models via coordinate descent", '
        "J. Stat. Softw., vol. 33, no. 1, pp. 1-22, NIH Public Access (2010)"
    ]

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.n, self.p = X.shape
        self.X = X
        self.y = y
        self.lmbd = np.array([lmbd])
        self.fit_intercept = fit_intercept

        jl = get_jl_interpreter()
        jl.include(str(JULIA_SOLVER_FILE))
        self.solve_lasso = jl.solve_lasso

        if sparse.issparse(X):
            scipyCSC_to_julia = jl.pyfunctionret(
                jl.scipyCSC_to_julia, jl.Any, jl.PyObject
            )
            self.X = scipyCSC_to_julia(X)

        # Trigger Julia JIT compilation
        w_dim = self.p + 1 if fit_intercept else self.p
        self.prev_solution = np.zeros(w_dim)

        warnings.filterwarnings("ignore", category=FutureWarning)
        self.run(1e-2)

    def run(self, tol):
        # remove possibly spurious warnings from pyjulia
        # TODO: remove filter when https://github.com/JuliaPy/pyjulia/issues/497
        # is resolved or otherwise fix the warning
        warnings.filterwarnings("ignore", category=FutureWarning)

        coefs, converged = self.solve_lasso(
            self.X,
            self.y,
            self.lmbd / len(self.y),
            self.fit_intercept,
            tol**1.8,
            tol == INFINITY,
        )
        if converged:
            self.coefs = coefs
            self.prev_solution = coefs
        else:
            self.coefs = self.prev_solution

    def get_result(self):
        coefs = np.ravel(self.coefs)

        if self.fit_intercept:
            coefs = np.hstack((coefs[1:], coefs[0]))

        return coefs
