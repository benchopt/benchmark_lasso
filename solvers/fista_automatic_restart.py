from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


class Solver(BaseSolver):
    name = 'FISTA_automatic_restart'
    stopping_strategy = "callback"

    references = [
        'J.-F. Aujol, Ch. Dossal, H. Labarrière, A. Rondepierre, '
        '"FISTA restart using an automatic estimation of the '
        'growth parameter", HAL preprint : hal-03153525v4'
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"
        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

    def run(self, callback):
        L = self.compute_lipschitz_constant()
        n_features = self.X.shape[1]
        w = np.zeros(n_features)
        z = np.zeros(n_features)
        C = 6.38  # Parameter controlling the restart frequency, 6.38 is the
        # optimal theoretical value (we refer the user to the preprint for
        # more details).
        ite_per_restart = np.array([int(2*C)])
        objectives = np.array([self.cost_function(w)])
        should_run = callback(w)
        # First call of FISTA
        for i in range(ite_per_restart[-1]):
            w_old = w.copy()
            z -= self.X.T @ (self.X @ z - self.y) / L
            w = self.st(z, self.lmbd / L)
            z = w + i/(i+3) * (w - w_old)
            should_run = callback(w)
            if not should_run:
                break
        ite_per_restart = np.r_[ite_per_restart, ite_per_restart[-1]]
        objectives = np.r_[objectives, self.cost_function(w)]
        while should_run:
            # Restart of FISTA after k_tab[counter] iterations
            for i in range(ite_per_restart[-1]):
                w_old = w.copy()
                z -= self.X.T @ (self.X @ z - self.y) / L
                w = self.st(z, self.lmbd / L)
                z = w + i/(i+3) * (w - w_old)
                should_run = callback(w)
                if not should_run:
                    break
            objectives = np.r_[objectives, self.cost_function(w)]
            # Estimation of the growth parameter
            mu = np.min(4 * L / (ite_per_restart[:-1]+1) ** 2
                        * (objectives[:-2] - objectives[-1]) /
                        (objectives[1:-1] - objectives[-1]))
            # Update of the number of iterations before next restart
            if ite_per_restart[-1] <= C*np.sqrt(L/mu):
                ite_per_restart = np.r_[ite_per_restart,
                                        2 * ite_per_restart[-1]]
            else:
                ite_per_restart = np.r_[ite_per_restart,
                                        ite_per_restart[-1]]
        self.w = w

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w

    def cost_function(self, w):
        F = (.5*np.linalg.norm(self.X @ w - self.y, ord=2)**2
             + self.lmbd*abs(w).sum())
        return F

    def get_result(self):
        return self.w

    def compute_lipschitz_constant(self):
        if not sparse.issparse(self.X):
            L = np.linalg.norm(self.X, ord=2) ** 2
        else:
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2
        return L
