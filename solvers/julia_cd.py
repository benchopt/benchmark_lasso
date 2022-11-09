from pathlib import Path
from benchopt import safe_import_context

from benchopt.helpers.julia import JuliaSolver
from benchopt.helpers.julia import get_jl_interpreter
from benchopt.helpers.julia import assert_julia_installed

with safe_import_context() as import_ctx:
    assert_julia_installed()


# File containing the function to be called from julia
JULIA_SOLVER_FILE = str(Path(__file__).with_suffix('.jl'))


class Solver(JuliaSolver):

    # Config of the solver
    name = 'Julia-CD'
    stopping_strategy = 'iteration'
    references = [
        'H.J.M. Shi et al., "A Primer on Coordinate Descent Algorithms", In: '
        'arXiv preprint arXiv:1610.00040 (2016).'
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        # XXX - fit intercept is not yet implemented in julia.jl
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        jl = get_jl_interpreter()
        self.solve_lasso = jl.include(JULIA_SOLVER_FILE)

    def run(self, n_iter):
        L = (self.X ** 2).sum(axis=0)
        self.beta = self.solve_lasso(self.X, self.y, self.lmbd, L, n_iter)

    def get_result(self):
        return self.beta.ravel()