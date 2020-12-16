import numpy as np

from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from modopt.opt.algorithms import POGM
    from modopt.opt.proximity import SparseThreshold
    from modopt.opt.linear import Identity
    from modopt.opt.gradient import GradBasic

class Solver(BaseSolver):
    name = 'ModOpt-POGM'
    stop_strategy = 'iteration'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/CEA-COSMIC/ModOpt.git',
    ]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        n_features = self.X.shape[1]
        L = np.linalg.norm(self.X, ord=2) ** 2
        beta_param = 1 / L
        sigma_bar = 0.96
        var_init = np.zeros(n_features)
        self.pogm = POGM(
            x=var_init,  # this is the coefficient w
            u=var_init,
            y=var_init,
            z=var_init,
            grad=GradBasic(
                op=lambda w: self.X@w,
                trans_op=lambda res: self.X.T@res,
                data=y,
            ),
            prox=SparseThreshold(Identity(), lmbd),
            beta_param=beta_param,
            metric_call_period=None,
            sigma_bar=sigma_bar,
            auto_iterate=False,
            progress=False,
            cost=None,
        )

    def run(self, n_iter):
        self.pogm.iterate(max_iter=n_iter)

    def get_result(self):
        return self.pogm.x_final
