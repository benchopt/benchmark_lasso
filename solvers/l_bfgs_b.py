import numpy as np
from numpy.linalg import norm

from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from scipy.optimize import fmin_l_bfgs_b


class Solver(BaseSolver):
    name = "L-BFGS-B"

    install_cmd = 'conda'
    requirements = ['scipy']
    stop_strategy = 'iteration'
    support_sparse = False

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def run(self, n_iter):
        X, y, lmbd = self.X, self.y, self.lmbd
        n_features = X.shape[1]

        def f(x):
            x = x[::2] - x[1::2]
            return 0.5 * norm(X.dot(x) - y) ** 2 + lmbd * np.abs(x).sum()

        def gradf(x):
            x_pos = x[::2]
            x_neg = -x[1::2]
            x = x_pos + x_neg
            grad = np.empty(2 * x.size)
            R = (X.dot(x) - y)
            grad[::2] = X.T.dot(R)
            grad[1::2] = -grad[::2]
            grad += lmbd
            return grad

        x0 = np.zeros(2 * n_features)
        bounds = (2 * n_features) * [(0, None)]  # set positivity bounds
        x_hat, _, _ = fmin_l_bfgs_b(f, x0, gradf, bounds=bounds,
                                    pgtol=0., factr=0., maxiter=n_iter)
        x_hat = x_hat[::2] - x_hat[1::2]

        self.w = x_hat

    def get_result(self):
        return self.w.flatten()
