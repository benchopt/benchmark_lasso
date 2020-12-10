import numpy as np
from pathlib import Path

from benchopt.base import BaseSolver
from benchopt.util import safe_import_context

with safe_import_context() as import_ctx:

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    from benchopt.utils.r_helpers import import_func_from_r_file

    # Setup the system to allow rpy2 running
    R_FILE = str(Path(__file__).with_suffix('.R'))
    import_func_from_r_file(R_FILE)
    numpy2ri.activate()


class Solver(BaseSolver):
    name = "R-PGD"

    install_cmd = 'conda'
    requirements = ['r-base', '-c conda-forge r rpy2']
    stop_strategy = 'iteration'
    support_sparse = False

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.lmbd_max = np.max(np.abs(X.T @ y))
        self.r_pgd = robjects.r['proximal_gradient_descent']

    def run(self, n_iter):
        coefs = self.r_pgd(
            self.X, self.y[:, None], self.lmbd,
            n_iter=n_iter)
        as_matrix = robjects.r['as']
        self.w = np.array(as_matrix(coefs, "matrix"))

    def get_result(self):
        return self.w.flatten()
