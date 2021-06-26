from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import scipy.optimize as sciop


class Solver(BaseSolver):
    name = "Noncvx-Pro"

    stop_strategy = 'iteration'
    support_sparse = False

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def run(self, n_iter):
        X, y, lmbd = self.X, self.y, self.lmbd
        n_features = X.shape[1]

        def objfn(u):
            Xu = X * u
            M = lmbd * np.eye(n_features) + np.dot(Xu.T, Xu)

            v1 = np.linalg.solve(M, Xu.T @ y)

            res = X @ (v1 * u) - y
            grad = (X * v1).T @ res + lmbd * u
            f = (
                (res * res).sum() / 2
                + lmbd / 2 * ((u * u).sum() + (v1 * v1).sum())
            )
            return f, grad

        # run lbfgs
        myopts = {'gtol': 1e-8, 'maxiter': n_iter, 'maxcor': 100, 'ftol': 0}
        u0 = np.ones(n_features)

        lbfgs_res = sciop.minimize(
            objfn, u0, method='L-BFGS-B', jac=True, options=myopts
        )
        u1 = lbfgs_res.x
        Xu1 = X * u1
        M = lmbd * np.eye(n_features) + np.dot(Xu1.T, Xu1)

        v1 = np.linalg.solve(M, Xu1.T @ y)

        self.w = u1 * v1

    def get_result(self):
        return self.w.flatten()
