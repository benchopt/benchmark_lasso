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
    references = [
        'S. Farrens, A. Grigis, L. El Gueddari, Z. Ramzi, G. R. Chaithya, '
        'S. Starck, B. Sarthou, H. Cherkaoui, P. Ciuciu and J.-L. Starck, '
        '"PySAP: Python Sparse Data Analysis Package for multidisciplinary '
        'image processing", Astronomy and Computing, vol. 32, '
        ' pp. 100402 (2020)'
    ]
    support_sparse = False

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        n_samples, n_features = self.X.shape
        x_shape = n_features
        if fit_intercept:
            x_shape += n_samples
        self.var_init = np.zeros(x_shape)
        if self.restart_strategy == 'greedy':
            min_beta = 1.0
            s_greedy = 1.1
            p_lazy = 1.0
            q_lazy = 1.0
        else:
            min_beta = None
            s_greedy = None
            p_lazy = 1 / 30
            q_lazy = 1 / 10

        if fit_intercept:
            def op(w):
                return self.X @ w[:n_features] + w[n_features:]
        else:
            def op(w):
                return self.X @ w
        # TODO implement correct gradient if fit_intercept

        self.fb = ForwardBackward(
            x=self.var_init,  # this is the coefficient w
            grad=GradBasic(
                input_data=y, op=op,
                trans_op=lambda res: self.X.T@res,
            ),
            prox=SparseThreshold(Identity(), lmbd),
            beta_param=1.0,
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
        L = np.linalg.norm(self.X, ord=2) ** 2
        if self.restart_strategy == 'greedy':
            beta_param = 1.3 / L
        else:
            beta_param = 1 / L
        self.fb.beta_param = beta_param
        self.fb._beta = self.fb.step_size or beta_param

        # reset this internal state otherwise warm start is used:
        self.fb._x_old = self.var_init
        self.fb._z_old = self.var_init
        # no attribute x_final if max_iter=0
        self.fb.iterate(max_iter=n_iter + 1)
        # MM: modopt makes input not writeable, this breaks other solvers
        # so we revert manually
        self.X.flags.writeable = True
        self.y.flags.writeable = True

    def get_result(self):
        return self.fb.x_final
