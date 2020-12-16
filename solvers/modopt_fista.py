import numpy as np

from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from modopt.opt.algorithms import ForwardBackward
    from modopt.opt.proximity import SparseThreshold
    from modopt.opt.linear import Identity
    from modopt.opt.gradient import GradBasic


class Solver(BaseSolver):
    name = 'ModOpt-FISTA'
    stop_strategy = 'iteration'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/CEA-COSMIC/ModOpt.git',
    ]

    parameters = {
        'restart_strategy': ['greedy', 'adaptive-1'],
    }

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        n_features = self.X.shape[1]
        L = np.linalg.norm(self.X, ord=2) ** 2
        if self.restart_strategy == 'greedy':
            beta_param = 1.3 * (1/L)
            min_beta = 1.0
            s_greedy = 1.1
            p_lazy = 1.0
            q_lazy = 1.0
        else:
            beta_param = 1 / L
            min_beta = None
            s_greedy = None
            p_lazy = 1 / 30
            q_lazy = 1 / 10
        self.fb = ForwardBackward(
            x=np.zeros(n_features),  # this is the coefficient w
            grad=GradBasic(
                op=lambda w: self.X@w,
                trans_op=lambda res: self.X.T@res,
                data=y,
            ),
            prox=SparseThreshold(Identity(), lmbd),
            beta_param=beta_param,
            min_beta=min_beta,
            metric_call_period=None,
            restart_strategy=self.restart_strategy,
            xi_restart=0.96,
            s_greedy=s_greedy,
            p_lazy=p_lazy,
            q_lazy=q_lazy,
            auto_iterate=False,
            progress=False,
            cost=None,
        )

    def run(self, n_iter):
        self.fb.iterate(max_iter=n_iter)

    def get_result(self):
        return self.fb.x_final
