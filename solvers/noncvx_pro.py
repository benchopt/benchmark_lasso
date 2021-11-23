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

    def efficient_solve(self, X, y, lmbd):
        n_samples, n_features = X.shape
        if n_samples >= n_features:
            M = lmbd * np.eye(n_features) + np.dot(X.T, X)
            v1 = np.linalg.solve(M, X.T @ y)
        else:
            M = lmbd * np.eye(n_samples) + np.dot(X, X.T)
            v1 = X.T @ np.linalg.solve(M, y)
        return v1

    def run(self, n_iter):
        X, y, lmbd = self.X, self.y, self.lmbd
        n_samples, n_features = X.shape

        def objfn(u):
            Xu = X * u
            v1 = self.efficient_solve(Xu, y, lmbd)

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
        v1 = self.efficient_solve(Xu1, y, lmbd)

        self.w = u1 * v1

    def get_result(self):
        return self.w.flatten()
