from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


class Solver(BaseSolver):
    name = 'Python-PGD'  # proximal gradient, optionally accelerated
    stopping_strategy = "callback"

    # any parameter defined here is accessible as a class attribute
    parameters = {'use_acceleration': [False, True]}
    references = [
        'I. Daubechies, M. Defrise and C. De Mol, '
        '"An iterative thresholding algorithm for linear inverse problems '
        'with a sparsity constraint", Comm. Pure Appl. Math., '
        'vol. 57, pp. 1413-1457, no. 11, Wiley Online Library (2004)',
        'A. Beck and M. Teboulle, "A fast iterative shrinkage-thresholding '
        'algorithm for linear inverse problems", SIAM J. Imaging Sci., '
        'vol. 2, no. 1, pp. 183-202 (2009)'
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        # XXX - intercept not implemented for sparse X but it shouldn't be hard
        if fit_intercept and sparse.issparse(X):
            return (
                True,
                f"{self.name} doesn't handle fit_intercept with sparse data",
            )

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        # Handling intercept: center y and X (dense data only)
        if fit_intercept and not sparse.issparse(self.X):
            self.X_offset = np.average(X, axis=0)
            X -= self.X_offset
            self.y_offset = np.average(y, axis=0)
            y -= self.y_offset

        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

    def run(self, callback):
        L = self.compute_lipschitz_constant()

        n_features = self.X.shape[1]
        w = np.zeros(n_features)
        if self.use_acceleration:
            z = np.zeros(n_features)

        intercept = self.y_offset if self.fit_intercept else []

        t_new = 1
        while callback(np.r_[w, intercept]):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
                w_old = w.copy()
                z -= self.X.T @ (self.X @ z - self.y) / L
                w = self.st(z, self.lmbd / L)
                z = w + (t_old - 1.) / t_new * (w - w_old)
            else:
                w -= self.X.T @ (self.X @ w - self.y) / L
                w = self.st(w, self.lmbd / L)

            if self.fit_intercept:
                intercept = self.y_offset - self.X_offset @ w

        self.w = np.r_[w, intercept]

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w

    def get_result(self):
        return self.w

    def compute_lipschitz_constant(self):
        if not sparse.issparse(self.X):
            L = np.linalg.norm(self.X, ord=2) ** 2
        else:
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2
        return L
