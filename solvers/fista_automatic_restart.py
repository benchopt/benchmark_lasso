from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


class Solver(BaseSolver):
    name = 'FISTA_automatic_restart'
    stopping_strategy = "iteration"

    references = [
        'J.-F. Aujol, Ch. Dossal, H. Labarrière, A. Rondepierre, '
        '"FISTA restart using an automatic estimation of the '
        'growth parameter", HAL preprint : hal-03153525v4, '
        'A. Beck and M. Teboulle,'
        '"A fast iterative shrinkage-thresholding algorithm for'
        ' linear inverse problems", SIAM J. Imaging Sci., '
        'vol. 2, no. 1, pp. 183-202 (2009)'
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"
        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        L = self.compute_lipschitz_constant()
        n_features = self.X.shape[1]
        w = np.zeros(n_features)
        z = np.zeros(n_features)
        C = 6.38
        ite_per_restart = np.zeros(int(n_iter / (2 * C)) + 2)
        F_tab = np.zeros(int(n_iter / (2 * C)) + 2)
        i_glob = 0
        i_int = 0
        counter = 0  # Restart counter
        LastF = 0
        ite_per_restart[counter] = int(2*C)
        F_tab[counter] = (.5*np.linalg.norm(self.X @ w - self.y, ord=2)**2
                          + self.lmbd*abs(w).sum())
        # First call of FISTA
        while i_glob < n_iter and i_int < ite_per_restart[counter]:
            w_old = w.copy()
            z -= self.X.T @ (self.X @ z - self.y) / L
            w = self.st(z, self.lmbd / L)
            z = w + i_int/(i_int+3) * (w - w_old)
            i_int += 1
            i_glob += 1
        counter += 1
        ite_per_restart[counter] = int(2*C)
        F_tab[counter] = (.5*np.linalg.norm(self.X @ w - self.y, ord=2)**2
                          + self.lmbd*abs(w).sum())
        while i_glob < n_iter:
            i_int = 0
            # Restart of FISTA after k_tab[counter] iterations
            while i_glob < n_iter and i_int < ite_per_restart[counter]:
                w_old = w.copy()
                z -= self.X.T @ (self.X @ z - self.y) / L
                w = self.st(z, self.lmbd / L)
                z = w + i_int/(i_int+3) * (w - w_old)
                i_int += 1
                i_glob += 1
            counter += 1
            if i_glob < n_iter:
                LastF = (.5 * np.linalg.norm(self.X @ w - self.y, ord=2) ** 2
                         + self.lmbd * abs(w).sum())
                F_tab[counter] = LastF
                # Estimation of the growth parameter
                mu = (np.min(4 * L / (ite_per_restart[:counter-1]+1) ** 2
                             * (F_tab[:counter-1] - LastF) /
                             (F_tab[1:counter] - LastF)))
                # Update of the number of iterations before next restart
                if ite_per_restart[counter-1] <= C*np.sqrt(L/mu):
                    ite_per_restart[counter] = 2 * ite_per_restart[counter-1]
                else:
                    ite_per_restart[counter] = ite_per_restart[counter-1]
        self.w = w

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
