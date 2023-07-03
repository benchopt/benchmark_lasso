import numpy as np

from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from scipy import sparse
    from scipy.sparse.linalg import LinearOperator
    from scipy.linalg.interpolative import estimate_spectral_norm
    from modopt.opt.algorithms import ForwardBackward
    from modopt.opt.proximity import SparseThreshold
    from modopt.opt.linear import Identity
    from modopt.opt.gradient import GradBasic


class Solver(BaseSolver):
    name = 'ModOpt-FISTA'
    stopping_strategy = 'callback'
    install_cmd = 'conda'
    requirements = [
        'pip:modopt',
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

    def skip(self, X, y, lmbd, fit_intercept):
        if sparse.issparse(X) and fit_intercept:
            return True, "modopt doesn't support sparse X and fit_intercept"
        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept
        n_features = self.X.shape[1]
        if fit_intercept:
            self.var_init = np.zeros(n_features + 1)
        else:
            self.var_init = np.zeros(n_features)

    def run(self, callback):
        if isinstance(self.X, LinearOperator):
            L = estimate_spectral_norm(self.X) ** 2
        elif sparse.issparse(self.X):
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2
        else:
            L = np.linalg.norm(self.X, ord=2) ** 2

        if self.restart_strategy == 'greedy':
            beta_param = 1.3 / L
            min_beta = 1. / L
            s_greedy = 1.1
            p_lazy = 1.
            q_lazy = 1.
        else:
            beta_param = 1. / L
            min_beta = None
            s_greedy = None
            p_lazy = 1 / 30
            q_lazy = 1 / 10

        n_features = self.X.shape[1]

        if self.fit_intercept:
            def op(w):
                return self.X @ w[:n_features] + w[-1]

            def trans_op(res):
                return np.hstack([self.X.T @ res, res.sum()])

            weights = np.full(self.X.shape[1] + 1, self.lmbd)
            weights[-1] = 0
        else:
            def op(w):
                return self.X @ w

            def trans_op(res):
                return self.X.T @ res

            weights = np.full(self.X.shape[1], self.lmbd)

        self.fb = ForwardBackward(
            x=self.var_init,  # this is the coefficient w
            grad=GradBasic(
                input_data=self.y,
                op=op,
                trans_op=trans_op,
                input_data_writeable=True,
            ),
            prox=SparseThreshold(Identity(), weights),
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

        self.fb.iterate(max_iter=1)
        while callback(self.fb.x_final):
            self.fb.iterate(max_iter=10)

    def get_result(self):
        return self.fb.x_final.copy()
