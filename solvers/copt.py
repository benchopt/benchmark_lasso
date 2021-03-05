import warnings
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.utils.extmath import row_norms
    import copt as cp
    import copt.loss
    import copt.penalty


class Solver(BaseSolver):
    name = 'copt'

    install_cmd = 'conda'
    requirements = ['pip:https://github.com/openopt/copt/archive/master.zip']

    parameters = {
        'accelerated': [False, True],
        'solver': ['pgd', 'svrg', 'saga'],
    }

    def skip(self, X, y, lmbd):
        if X.shape[1] > 50000:
            return True, "problem too large."
        if X.shape[1] > X.shape[0] and self.solver in ['svrg', 'saga']:
            return True, "n_features is bigger than n_samples"
        if self.accelerated and self.solver != "pgd":
            return True, f"accelerated is not available for {self.solver}"

        return False, None

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def run(self, n_iter):
        X, y, solver = self.X, self.y, self.solver

        n_features = X.shape[1]
        f = copt.loss.SquareLoss(X, y)
        g = copt.penalty.L1Norm(self.lmbd / len(X))

        warnings.filterwarnings('ignore', category=RuntimeWarning)

        if solver == 'pgd':
            step_size = 1.0 / f.lipschitz
            result = cp.minimize_proximal_gradient(
                f.f_grad,
                np.zeros(n_features),
                g.prox,
                step=lambda x: step_size,
                tol=0,
                max_iter=n_iter,
                jac=True,
                accelerated=self.accelerated,
            )
        elif solver == 'saga':
            max_squared_sum = row_norms(X, squared=True).max()
            step_size = 1.0 / (3 * max_squared_sum)
            result = cp.minimize_saga(
                f.partial_deriv,
                X,
                y,
                np.zeros(n_features),
                prox=g.prox_factory(n_features),
                step_size=step_size,
                tol=0,
                max_iter=100,
            )
        else:
            assert solver == 'svrg'
            max_squared_sum = row_norms(X, squared=True).max()
            step_size = 1.0 / (3 * max_squared_sum)
            result = cp.minimize_svrg(
                f.partial_deriv,
                X,
                y,
                np.zeros(n_features),
                prox=g.prox_factory(n_features),
                step_size=step_size,
                tol=0,
                max_iter=100,
            )

        self.beta = result.x

    def get_result(self):
        return self.beta.flatten()
