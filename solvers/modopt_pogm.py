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
    stopping_strategy = 'callback'

    install_cmd = 'conda'
    requirements = [
        'pip:modopt',
    ]
    references = [
        'S. Farrens, A. Grigis, L. El Gueddari, Z. Ramzi, G. R. Chaithya, '
        'S. Starck, B. Sarthou, H. Cherkaoui, P. Ciuciu and J.-L. Starck, '
        '"PySAP: Python Sparse Data Analysis Package for multidisciplinary '
        'image processing", Astronomy and Computing, vol. 32, '
        ' pp. 100402 (2020)'
    ]
    support_sparse = False

    def skip(self, X, y, lmbd, fit_intercept):
        # XXX - not implemented but not too complicated here.
        # if fit_intercept:
        #     return True, f"{self.name} does not handle fit_intercept"

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
        if self.fit_intercept:
            def op(w):
                return self.X @ w[:self.X.shape[1]] + w[-1]

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

        L = np.linalg.norm(self.X, ord=2) ** 2  # TODO sparse X?
        self.pogm = POGM(
            x=self.var_init,  # this is the coefficient w
            u=self.var_init,
            y=self.var_init,
            z=self.var_init,
            grad=GradBasic(
                input_data=self.y,
                op=op,
                trans_op=trans_op,
                input_data_writeable=True,
            ),
            prox=SparseThreshold(Identity(), weights),
            beta_param=1. / L,
            metric_call_period=None,
            sigma_bar=0.96,
            auto_iterate=False,
            progress=False,
            cost=None,
        )

        self.pogm.iterate(max_iter=1)
        while callback(self.pogm.x_final):
            self.pogm.iterate(max_iter=10)

    def get_result(self):
        return self.pogm.x_final
