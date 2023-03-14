from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import cupy as cp
    import numpy as np
    from scipy import sparse
    from math import sqrt


class Solver(BaseSolver):

    name = 'Python-PGD'  # proximal gradient, optionally accelerated
    stopping_strategy = "callback"

    requirements = [
        'conda-forge:cupy',
        'conda-forge:cudatoolkit=11.5'
    ]

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'use_acceleration': [False, True],
        'use_gpu': [False, True]
    }

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
        if sparse.issparse(X) and self.use_gpu:
            return True, "sparse is not supported with GPU"

        # XXX - not implemented but not too complicated to implement
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        if self.use_gpu:
            # transfert X, y to GPU
            X, y = cp.array(X), cp.array(y)

        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

    def run(self, callback):
        L = self.compute_lipschitz_constant()

        xp = cp if self.use_gpu else np

        n_features = self.X.shape[1]
        w = xp.zeros(n_features)
        if self.use_acceleration:
            z = xp.zeros(n_features)

        t_new = 1
        while callback(w):
            if self.use_acceleration:
                t_old = t_new
                t_new = (1 + sqrt(1 + 4 * t_old ** 2)) / 2
                w_old = w.copy()
                z -= self.X.T @ (self.X @ z - self.y) / L
                w = self.st(z, self.lmbd / L)
                z = w + (t_old - 1.) / t_new * (w - w_old)
            else:
                w -= self.X.T @ (self.X @ w - self.y) / L
                w = self.st(w, self.lmbd / L)

        if self.use_gpu:
            w = cp.asnumpy(w)

        self.w = w

    def st(self, w, mu):
        xp = cp if self.use_gpu else np
        w -= xp.clip(w, -mu, mu)
        return w

    def get_result(self):
        return self.w

    def compute_lipschitz_constant(self):
        if not sparse.issparse(self.X):
            xp = cp if self.use_gpu else np
            return xp.linalg.norm(self.X, ord=2) ** 2

        return sparse.linalg.svds(self.X, k=1)[1][0] ** 2
