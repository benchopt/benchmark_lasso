from pathlib import Path

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    from benchopt.helpers.r_lang import import_func_from_r_file

    # Setup the system to allow rpy2 running
    R_FILE = str(Path(__file__).with_suffix('.R'))
    import_func_from_r_file(R_FILE)
    numpy2ri.activate()


class Solver(BaseSolver):
    name = "R-PGD"

    install_cmd = 'conda'
    requirements = ['r-base', 'rpy2']
    stop_strategy = 'iteration'
    support_sparse = False
    references = [
        'I. Daubechies, M. Defrise and C. De Mol, '
        '"An iterative thresholding algorithm for linear inverse problems '
        'with a sparsity constraint", Comm. Pure Appl. Math., '
        'vol. 57, pp. 1413-1457, no. 11, Wiley Online Library (2004)',
        'A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding '
        'algorithm for linear inverse problems", SIAM J. Imaging Sci., '
        'vol. 2, no. 1, pp. 183-202 (2009)'
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept
        self.r_pgd = robjects.r['proximal_gradient_descent']

    def run(self, n_iter):
        coefs = self.r_pgd(
            self.X, self.y[:, None], self.lmbd,
            n_iter=n_iter)
        as_r = robjects.r['as']
        self.w = np.array(as_r(coefs, "vector"))

    def get_result(self):
        return self.w.flatten()
