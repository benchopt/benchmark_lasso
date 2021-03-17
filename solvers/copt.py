import warnings
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    import copt as cp
    import copt.loss
    import copt.penalty


class Solver(BaseSolver):
    name = 'copt'

    install_cmd = 'conda'
    requirements = ['pip:https://github.com/openopt/copt/archive/master.zip']

    parameters = {
        'accelerated': [False, True],
        'line_search': [False, True],
        'solver': ['pgd', 'svrg', 'saga'],
    }

    def skip(self, X, y, lmbd):
        if (X.shape[1] > 50_000) and self.solver not in ['svrg', 'saga']:
            return True, (
                f"problem too large (n_features={X.shape[1]} > 50000) "
                f"for solver {self.solver}."
            )
        if X.shape[1] > X.shape[0] and self.solver in ['svrg', 'saga']:
            msg = (
                f"n_features ({X.shape[1]}) is bigger than "
                f"n_samples ({X.shape[0]})"
            )
            return True, msg
        if (self.accelerated or self.line_search) and self.solver != "pgd":
            return (
                True,
                f"accelerated or line_search is not available for "
                f"{self.solver}"
            )

        return False, None

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        # Make sure we cache the numba compilation.
        if self.solver in ['svrg', 'saga']:
            self.run(1)

    def run(self, n_iter):
        X, y, solver = self.X, self.y, self.solver

        n_features = X.shape[1]

        x0 = np.zeros(n_features)
        if n_iter == 0:
            self.beta = x0
            return

        f = copt.loss.SquareLoss(X, y)
        g = copt.penalty.L1Norm(self.lmbd / X.shape[0])

        warnings.filterwarnings('ignore', category=RuntimeWarning)

        if solver == 'pgd':
            if self.line_search:
                step = 'backtracking'
            else:
                def step(x):
                    return 1.0 / f.lipschitz
            result = cp.minimize_proximal_gradient(
                f.f_grad,
                x0,
                g.prox,
                step=step,
                tol=0,
                max_iter=n_iter,
                jac=True,
                accelerated=self.accelerated,
            )
        elif solver == 'saga':
            step_size = 1.0 / (3 * f.max_lipschitz)
            result = cp.minimize_saga(
                f.partial_deriv,
                X,
                y,
                x0,
                prox=g.prox_factory(n_features),
                step_size=step_size,
                tol=0,
                max_iter=n_iter,
            )
        else:
            assert solver == 'svrg'
            step_size = 1.0 / (3 * f.max_lipschitz)
            result = cp.minimize_svrg(
                f.partial_deriv,
                X,
                y,
                x0,
                prox=g.prox_factory(n_features),
                step_size=step_size,
                tol=0,
                max_iter=n_iter,
            )

        self.beta = result.x

    def get_result(self):
        return self.beta.flatten()
