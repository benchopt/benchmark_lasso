from benchopt import BaseSolver
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientDescentCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    import scipy.optimize as sciop
    from scipy.sparse import issparse


class Solver(BaseSolver):
    name = "noncvx-pro"

    stopping_strategy = 'iteration'
    stopping_criterion = SufficientDescentCriterion(eps=1e-10, patience=5)
    references = [
        "Clarice Poon and Gabriel Peyré, "
        "'Smooth Bilevel Programming for Sparse Regularization', "
        "Advances in Neural Information Processing Systems (2021)"
    ]

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        if X.shape[0] >= X.shape[1]:
            self.C = X.T @ X
            if issparse(self.C):
                self.C = self.C.toarray()

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"
        return False, None

    def run(self, n_iter):
        # implementation from: https://github.com/gpeyre/numerical-tours/blob/
        # master/python/optim_7_noncvx_pro.ipynb
        X, y, lmbd = self.X, self.y, self.lmbd
        n_samples, n_features = X.shape

        if n_samples < n_features:
            def u_opt(v):
                if issparse(X):
                    S = X.multiply(v**2) @ X.T + lmbd * np.eye(n_samples)
                else:
                    S = X * v**2 @ X.T + lmbd * np.eye(n_samples)

                return v * (X.T @ np.linalg.solve(S, y))

            def nabla_f(v):
                u = u_opt(v)
                res = X @ (u*v) - y
                f = 1/(2 * lmbd) * norm(res)**2 + \
                    (norm(u)**2 + norm(v)**2) / 2
                g = u * (X.T @ res) / lmbd + v
                return f, g
        else:
            Xty = X.T @ y

            def u_opt(v):
                return np.linalg.solve(
                    self.C * v[:, None] * v[None, :] +
                    lmbd * np.eye(n_features), v * Xty)

            def nabla_f(v):
                u = u_opt(v)
                x = u * v
                Cx = self.C @ x
                E = Cx @ x + np.sum(y ** 2) - 2 * x @ Xty
                f = 1/(2*lmbd) * E + (norm(u)**2 + norm(v)**2)/2
                g = u * (Cx - Xty) / lmbd + v
                return f, g

        opts = {'gtol': 1e-30, 'maxiter': n_iter, 'maxcor': 5, 'ftol': 1e-30}
        u0 = np.ones(n_features)

        lbfgs_res = sciop.minimize(
            nabla_f, u0, method='L-BFGS-B', jac=True, options=opts
        )
        v = lbfgs_res.x
        self.w = v * u_opt(v)

    def get_result(self):
        return self.w.flatten()
