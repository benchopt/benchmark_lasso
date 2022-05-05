from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


class Solver(BaseSolver):
    name = "IRLS"  # Iterative reweighted Least Squares
    stopping_strategy = "iteration"

    references = ["???"]
    # see for instance: https://homepages.laas.fr/vmagron/masmode/Gabriel.pdf

    def skip(self, X, y, lmbd, fit_intercept):
        # XXX - not implemented but not too complicated to implement
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        epsilon = 1e-10

        def w_opt(eta):
            T = self.X.T @ self.X + self.lmbd * np.diag(1 / eta)
            return np.linalg.solve(T, self.X.T @ self.y)

        def eta_opt(w):
            return (w ** 2 + epsilon) ** 0.5

        n_features = self.X.shape[1]
        eta = np.full(n_features, epsilon)  # init
        w = np.zeros(n_features)

        for i in range(n_iter):
            # w_old = w.copy()
            w = w_opt(eta)
            eta = eta_opt(w)
            # z = w + (t_old - 1.0) / t_new * (w - w_old)
            # w -= self.X.T @ (self.X @ w - self.y) / L
            # w = self.st(w, self.lmbd / L)

        self.w = w

    def get_result(self):
        return self.w
