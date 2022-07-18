from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


class Solver(BaseSolver):
    name = 'Python-FISTA_automatic_restart'  # proximal gradient, optionally accelerated
    stopping_strategy = "iteration"

    # any parameter defined here is accessible as a class attribute
    references = [
        'J.-F. Aujol, Ch. Dossal, H. Labarrière, A. Rondepierre, '
        '"FISTA restart using an automatic estimation of the'
        'growth parameter", A. Beck and M. Teboulle,'
        '"A fast iterative shrinkage-thresholding algorithm for'
        ' linear inverse problems", SIAM J. Imaging Sci., '
        'vol. 2, no. 1, pp. 183-202 (2009)'
    ]

    def skip(self, X, y, lmbd, fit_intercept):
        # XXX - not implemented but not too complicated to implement
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
        C=6.38
        k_tab=np.zeros(int(n_iter/(2*C))+2)
        F_tab=np.zeros(int(n_iter/(2*C))+2)
        i_glob=0
        i_int=0
        counter=0
        k_tab[counter]=int(2*C)
        F_tab[counter]=.5*np.linalg.norm(self.X @ w - self.y,ord=2)**2+self.lmbd*abs(w).sum()
        #First call of FISTA
        while i_glob<n_iter and i_int<k_tab[counter]:
            w_old = w.copy()
            z -= self.X.T @ (self.X @ z - self.y) / L
            w = self.st(z, self.lmbd / L)
            z = w + i_int/(i_int+3) * (w - w_old)
            i_int+=1
            i_glob+=1
        counter+=1
        k_tab[counter]=int(2*C)
        F_tab[counter]=.5*np.linalg.norm(self.X @ w - self.y,ord=2)**2+self.lmbd*abs(w).sum()
        while i_glob<n_iter:
            i_int=0
            #Restart of FISTA after k_tab[counter] iterations
            while i_glob<n_iter and i_int<k_tab[counter]:
                w_old = w.copy()
                z -= self.X.T @ (self.X @ z - self.y) / L
                w = self.st(z, self.lmbd / L)
                z = w + i_int/(i_int+3) * (w - w_old)
                i_int+=1
                i_glob+=1
            counter+=1
            if i_glob<n_iter:
                F_tab[counter]=.5*np.linalg.norm(self.X @ w - self.y,ord=2)**2+self.lmbd*abs(w).sum()
                #Estimation of the growth parameter
                mu=np.min(4*L/(k_tab[:counter-1]+1)**2*(F_tab[:counter-1]-F_tab[counter])/(F_tab[1:counter]-F_tab[counter]))
                #Update of the number of iterations before next restart
                if (k_tab[counter-1]<=C*np.sqrt(L/mu)):
                    k_tab[counter]=2*k_tab[counter-1]
                else:
                    k_tab[counter]=k_tab[counter-1]
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
