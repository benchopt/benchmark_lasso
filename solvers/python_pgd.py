import numpy as np
from scipy import sparse


from benchopt import BaseSolver


class Solver(BaseSolver):
    name = 'Python-PGD'  # proximal gradient, optionally accelerated

    # any parameter defined here is accessible as a class attribute
    parameters = {'use_acceleration': [False, True]}

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def run(self, n_iter):
        L = self.compute_lipschitz_cste()

        n_features = self.X.shape[1]
        w = np.zeros(n_features)
        if self.use_acceleration:
            z = np.zeros(n_features)

        t_new = 1
        for _ in range(n_iter):
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

        self.w = w

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w

    def get_result(self):
        return self.w

    def compute_lipschitz_cste(self):
        if not sparse.issparse(self.X):
            return np.linalg.norm(self.X, ord=2) ** 2

        n, m = self.X.shape
        if n < m:
            A = self.X.T
        else:
            A = self.X

        b_k = np.random.rand(A.shape[1])
        diff = np.inf
        rk = np.inf

        while diff > 1e-15:
            # calculate the matrix-by-vector product Ab
            b_k1 = A.T @ (A @ b_k)

            # compute the eigenvalue and the difference of eigenvalue
            rk1 = rk
            rk = b_k1 @ b_k
            diff = abs(rk - rk1)

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return rk
