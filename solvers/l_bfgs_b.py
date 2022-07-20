from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from scipy.optimize import fmin_l_bfgs_b
    from scipy.sparse import issparse


class Solver(BaseSolver):
    name = "L-BFGS-B"

    stopping_strategy = 'iteration'
    support_sparse = False
    references = [
        'R. H. Byrd, P. Lu and J. Nocedal, '
        '"A Limited Memory Algorithm for Bound Constrained Optimization", '
        'SIAM J. Sci. Comput., '
        'vol 16, no. 5, pp. 1190-1208. (1995)',
        'C. Zhu, R. H. Byrd and J. Nocedal, '
        '"Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale '
        'bound-constrained optimization", '
        'ACM Trans. Math. Software, vol. 23, no. 4, pp. 550-560 (1997)',
        'J. L. Morales and J. Nocedal, "Remark on "algorithm 778: L-BFGS-B: '
        'Fortran subroutines for large-scale bound constrained optimization"" '
        'optimization", ACM Trans. Math. Software, vol. 38, no 1., pp. 1-4 '
        '(2011)',
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        # XXX - intercept not implemented for sparse X for now
        if fit_intercept and issparse(X):
            return (True,
                f"{self.name} doesn't handle fit_intercept with sparse data")

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        # Handling intercept: center y and X (dense data only)
        if fit_intercept and not issparse(self.X):
            self.X_offset = np.average(X, axis=0)
            X -= self.X_offset
            self.y_offset = np.average(y, axis=0)
            y -= self.y_offset

        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        X, y, lmbd = self.X, self.y, self.lmbd
        n_features = X.shape[1]

        def f(w):
            w = w[::2] - w[1::2]
            return 0.5 * norm(X.dot(w) - y) ** 2 + lmbd * np.abs(w).sum()

        def gradf(w):
            x_pos = w[::2]
            x_neg = -w[1::2]
            w = x_pos + x_neg
            grad = np.empty(2 * w.size)
            R = (X.dot(w) - y)
            grad[::2] = X.T.dot(R)
            grad[1::2] = -grad[::2]
            grad += lmbd
            return grad

        w0 = np.zeros(2 * n_features)
        bounds = (2 * n_features) * [(0, None)]  # set positivity bounds
        w_hat, _, _ = fmin_l_bfgs_b(f, w0, gradf, bounds=bounds,
                                    pgtol=0., factr=0., maxiter=n_iter)
        w_hat = w_hat[::2] - w_hat[1::2]

        self.w = w_hat

        if self.fit_intercept and not issparse(self.X):
            intercept = self.y_offset - self.X_offset @ self.w
            self.w = np.r_[self.w, intercept]

    def get_result(self):
        return self.w.flatten()
