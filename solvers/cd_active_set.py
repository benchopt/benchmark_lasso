from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm
    from scipy import sparse
    from numba import njit


if import_ctx.failed_import:

    def njit(f):  # noqa: F811
        return f


@njit
def st(x, mu):
    if x > mu:
        return x - mu, 1
    if x < -mu:
        return x + mu, 1
    return 0.0, 0


@njit
def get_lasso_dgap(X, y, w, active_set, lmbd):
    y_hat = np.dot(X[:, active_set], w)
    R = y - y_hat
    penalty = norm(w, ord=1)
    nR2 = np.dot(R, R)
    p_obj = 0.5 * nR2 + lmbd * penalty

    dual_norm = np.max(np.dot(X.T, R))
    scaling = lmbd / dual_norm
    scaling = min(scaling, 1.0)
    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(
        R * y_hat
    )
    gap = p_obj - d_obj
    return gap, p_obj, d_obj


class Solver(BaseSolver):
    name = "cd_active_set"
    stop_strategy = "callback"

    install_cmd = "conda"
    requirements = ["numba"]
    references = [
        'W. J. Fu, "Penalized Regressions: the Bridge versus the Lasso", '
        "J. Comput. Graph. Statist., vol.7, no. 3, pp. 397-416, "
        "Taylor & Francis (1998)",
        "J. Friedman, T. J. Hastie, H. Höfling and R. Tibshirani, "
        '"Pathwise coordinate optimization", Ann. Appl. Stat., vol 1, no. 2, '
        "pp. 302-332 (2007)",
    ]

    def set_objective(self, X, y, lmbd):
        self.y, self.lmbd = y, lmbd
        self.active_set_size = 10
        self.max_iter = 2000
        self.tol = 1e-8

        if sparse.issparse(X):
            self.X = X
        else:
            # use Fortran order to compute gradient on contiguous columns
            self.X = np.asfortranarray(X)

        # Make sure we cache the numba compilation
        n_features = X.shape[1]
        self.cd(self.X, self.y, self.lmbd, np.ones(n_features), 10, None, 1e-5)

    def run(self, callback):
        n_features = self.X.shape[1]

        if sparse.issparse(self.X):
            L = np.array((self.X.multiply(self.X)).sum(axis=0)).squeeze()
        else:
            L = (self.X ** 2).sum(axis=0)

        # Initializing active set
        active_set = np.zeros(n_features, dtype=bool)
        idx_large_corr = np.argsort(np.dot(self.X.T, self.y))
        new_active_idx = idx_large_corr[-self.active_set_size :]
        active_set[new_active_idx] = True
        as_size = np.sum(active_set)

        coef_init = None
        self.w = np.zeros(n_features)

        while callback(self.w):
            L_as = L[active_set]

            coef, as_ = self.cd(
                self.X[:, active_set],
                self.y,
                self.lmbd,
                L_as,
                self.max_iter,
                coef_init,
                self.tol,
            )

            active_set[active_set] = as_.copy()
            idx_old_active_set = np.where(active_set)[0]

            self.build_full_coefficient_matrix(active_set, coef)

            R = self.y - np.dot(self.X[:, active_set], coef)
            idx_large_corr = np.argsort(np.dot(self.X.T, R))
            new_active_idx = idx_large_corr[-self.active_set_size :]
            active_set[new_active_idx] = True
            idx_active_set = np.where(active_set)[0]
            as_size = np.sum(active_set)
            coef_init = np.zeros((as_size,), dtype=coef.dtype)
            idx = np.searchsorted(idx_active_set, idx_old_active_set)
            coef_init[idx] = coef

    def build_full_coefficient_matrix(self, active_set, coef):
        final_coef_ = np.zeros(len(active_set))
        if coef is not None:
            final_coef_[active_set] = coef
        self.w = final_coef_

    @staticmethod
    @njit
    def cd(X, y, lmbd, L, n_iter, init, tol, dgap_freq=10):
        n_features = X.shape[1]

        if init is None:
            w = np.zeros(n_features)
            R = np.copy(y)
        else:
            w = init
            R = y - np.dot(X, w)

        highest_d_obj = -np.inf
        # np.bool_ is the only boolean type supported by Numba
        # see: https://github.com/numba/numba/issues/1311
        active_set = np.zeros(n_features, dtype=np.bool_)

        for i in range(n_iter):
            for j in range(n_features):
                if L[j] == 0.0:
                    continue
                old = w[j]
                w[j], active_set[j] = st(
                    w[j] + X[:, j] @ R / L[j], lmbd / L[j]
                )
                diff = old - w[j]
                if diff != 0:
                    R += diff * X[:, j]

            if i % dgap_freq == 0:
                _, p_obj, d_obj = get_lasso_dgap(
                    X, y, w[active_set], active_set, lmbd
                )
                highest_d_obj = max(d_obj, highest_d_obj)
                gap = p_obj - highest_d_obj

                if gap < tol:
                    break

        w = w[active_set]
        return w, active_set

    def get_result(self):
        return self.w
