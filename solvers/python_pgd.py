from math import sqrt
import numpy as np
from scipy import sparse

from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import cupy as cp


class Solver(BaseSolver):
    name = 'Python-PGD'  # proximal gradient, optionally accelerated

    requirements = [
        'conda-forge:cupy',
        'conda-forge:cudatoolkit=11.5'
    ]

    # any parameter defined here is accessible as a class attribute
    parameters = {
        'use_acceleration': [False, True],
        'use_gpu': [False, True]
    }

    def skip(self, X, y, lmbd):
        if sparse.issparse(X) and self.use_gpu:
            return True, "sparse is not supported with GPU"
        return False, None

    def set_objective(self, X, y, lmbd):
        if self.use_gpu:
            X, y = cp.array(X), cp.array(y)
        self.X, self.y, self.lmbd = X, y, lmbd

    def run(self, n_iter):
        L = self.compute_lipschitz_cste()

        xp = cp if self.use_gpu else np

        n_features = self.X.shape[1]
        w = xp.zeros(n_features)
        if self.use_acceleration:
            z = xp.zeros(n_features)

        t_new = 1
        for _ in range(n_iter):
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

    def compute_lipschitz_cste(self, max_iter=100):
        if not sparse.issparse(self.X):
            xp = cp if self.use_gpu else np
            return xp.linalg.norm(self.X, ord=2) ** 2

        n, m = self.X.shape
        if n < m:
            A = self.X.T
        else:
            A = self.X

        b_k = np.random.rand(A.shape[1])
        b_k /= np.linalg.norm(b_k)
        rk = np.inf

        for _ in range(max_iter):
            # calculate the matrix-by-vector product Ab
            b_k1 = A.T @ (A @ b_k)

            # compute the eigenvalue and stop if it does not move anymore
            rk1 = rk
            rk = b_k1 @ b_k
            if abs(rk - rk1) < 1e-10:
                break

            # re normalize the vector
            b_k = b_k1 / np.linalg.norm(b_k1)

        return rk
