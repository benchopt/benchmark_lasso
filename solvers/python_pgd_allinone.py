from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


class Solver(BaseSolver):
    name = 'Python-PGD-all-in-one'  # proximal gradient, optionally accelerated
    # uses callbacks at each iterations instead of rerunning multiple times

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

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w

    def compute_lipschitz_cste(self, max_iter=100):
        if not sparse.issparse(self.X):
            return np.linalg.norm(self.X, ord=2) ** 2

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

    def run_with_cb(self, callback):
        L = self.compute_lipschitz_cste()
        n_features = self.X.shape[1]
        w = np.zeros(n_features)
        if self.use_acceleration:
            z = np.zeros(n_features)

        it = 0
        t_new = 1
        while callback(it, w):
            it += 1
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

    def get_result(self):
        return self.w
