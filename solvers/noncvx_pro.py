from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import scipy.optimize as sciop
    from scipy.sparse import linalg as slinalg


class Solver(BaseSolver):
    name = "noncvx-pro"

    stop_strategy = 'iteration'
    support_sparse = False
    parameters = {'old': [True, False]}

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd

    def efficient_solve(self, A, b, lmbd):
        """Solve (A.T @ A + lmbd Id) x = b without forming the LHS matrix."""
        n_samples, n_features = A.shape
        if self.old:
            if n_samples >= n_features:
                M = lmbd * np.eye(n_features) + np.dot(A.T, A)
                v1 = np.linalg.solve(M, A.T @ b)
            else:
                M = lmbd * np.eye(n_samples) + np.dot(A, A.T)
                v1 = A.T @ np.linalg.solve(M, b)
        else:
            if n_samples >= n_features:
                def mv(x):
                    return lmbd * x + A.T @ (A @ x)
                linop = slinalg.LinearOperator(
                    shape=(n_features, n_features), matvec=mv
                )
                v1 = slinalg.cg(linop, A.T @ b)[0]
            else:
                def mv(z):
                    return lmbd * z + A @ (A.T @ z)
                linop = slinalg.LinearOperator(
                    shape=(n_samples, n_samples), matvec=mv
                )
                v1 = A.T @ slinalg.cg(linop, b)[0]
        return v1

    def v_opt(self, X, y, lmbd, u):
        """Optimal v for a fixed u.
        Minimizes in v:
            .5 * norm(X @ (u * v) - y) ** 2
            + .5 * lmbd * (norm(u) ** 2 + norm(v) ** 2)

        with Xu = X * u, X @ (u * v) = Xu @ v and so the solution v solves:
            (Xu.T @ Xu + lambda eye(n_features)) v = Xu.T @ y

        We use Sherman Morrison trick when n_samples > n_features.
        """
        n_samples, n_features = X.shape
        if n_samples >= n_features:
            def mv(x):
                # Xu.T @ Xu @ x = u * (X.T @ X @ (u * x))
                return lmbd * x + u * (X.T @ (X @ (u * x)))
            linop = slinalg.LinearOperator(
                shape=(n_features, n_features), matvec=mv
            )
            v = slinalg.cg(linop, u * (X.T @ y))[0]
        else:
            def mv(z):
                # Xu @ Xu.T @ z = Xu @ (u * (X.T @ z))
                return lmbd * z + X @ (u * (u * (X.T @ z)))
            linop = slinalg.LinearOperator(
                shape=(n_samples, n_samples), matvec=mv
            )
            v = u * (X.T @ slinalg.cg(linop, y)[0])
        return v

    def run(self, n_iter):
        X, y, lmbd = self.X, self.y, self.lmbd
        n_samples, n_features = X.shape

        def objfn(u):
            if self.old:
                Xu = X * u  # This breaks for sparse matrices
                v1 = self.efficient_solve(Xu, y, lmbd)
            else:
                v1 = self.v_opt(X, y, lmbd, u)

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
