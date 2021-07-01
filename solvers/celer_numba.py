from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from numpy.linalg import norm
    from numba import njit


@njit
def dual_lasso(alpha, norm_y2, theta, y):
    d_obj = 0
    n_samples = theta.shape[0]
    for i in range(n_samples):
        d_obj -= (y[i] / (alpha * n_samples) - theta[i]) ** 2
    d_obj *= 0.5 * alpha ** 2 * n_samples
    d_obj += norm_y2 / (2. * n_samples)
    return d_obj


@njit
def primal_lasso(alpha, R, w):
    return (R @ R) / (2 * len(R)) + alpha * np.sum(np.abs(w))


@njit
def ST(x, tau):
    if x > tau:
        return x - tau
    elif x < -tau:
        return x + tau
    return 0


@njit
def create_dual_pt(alpha, out, R):
    """Copied from cython code hence the unpythonic way to do it."""
    n_samples = R.shape[0]
    scal = 1 / (alpha * n_samples)
    out[:] = R
    out *= scal


@njit
def dnorm_l1(theta, X, skip):
    dnorm = 0
    for j in range(X.shape[1]):
        if not skip[j]:
            dnorm = max(dnorm, np.abs(X[:, j] @ theta))
    return dnorm


@njit
def set_prios(theta, X, norms_X_col, prios, screened, radius, n_screened):
    n_features = X.shape[1]
    for j in range(n_features):
        if screened[j] or norms_X_col[j] == 0:
            prios[j] = np.inf
            continue
        Xj_theta = X[:, j] @ theta
        prios[j] = (1. - np.abs(Xj_theta)) / norms_X_col[j]
        if prios[j] > radius:
            screened[j] = True
            n_screened += 1
    return n_screened


@njit
def numba_celer(X, y, alpha, n_iter, p0=10, tol=1e-12, prune=True, gap_freq=10,
                max_epochs=10_000, verbose=0):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    R = y.copy()
    verbose_in = max(verbose-1, 0)

    # tol *= norm(y) ** 2 / n_samples
    if p0 > n_features:
        p0 = n_features

    prios = np.empty(n_features, dtype=X.dtype)
    screened = np.zeros(n_features, dtype=np.int32)
    notin_WS = np.zeros(n_features, dtype=np.int32)

    # acceleration variables:
    K = 6
    last_K_Xw = np.empty([K, n_samples], dtype=X.dtype)
    U = np.empty([K - 1, n_samples], dtype=X.dtype)
    UtU = np.empty([K - 1, K - 1], dtype=X.dtype)
    onesK = np.ones(K - 1, dtype=X.dtype)

    inv_lc = np.zeros(n_features)
    norms_X_col = np.zeros(n_features)
    for j in range(n_features):
        norms_X_col[j] = norm(X[:, j])
        inv_lc[j] = 1 / norms_X_col[j] ** 2

    norm_y2 = norm(y) ** 2

    gaps = np.zeros(n_iter, dtype=X.dtype)

    theta = np.zeros(n_samples, dtype=X.dtype)
    theta_in = np.zeros(n_samples, dtype=X.dtype)
    thetacc = np.zeros(n_samples, dtype=X.dtype)
    d_obj_from_inner = 0.

    all_features = np.arange(n_features, dtype=np.int32)

    for t in range(n_iter):
        if t != 0:
            create_dual_pt(alpha, theta, R)

            scal = dnorm_l1(theta, X, screened)

            if scal > 1.:
                theta /= scal

            d_obj = dual_lasso(alpha, norm_y2, theta, y)

            # also test dual point returned by inner solver after 1st iter:
            scal = dnorm_l1(theta_in, X, screened)
            if scal > 1.:
                theta_in /= scal

            d_obj_from_inner = dual_lasso(alpha, norm_y2, theta_in, y)
        else:
            d_obj = dual_lasso(alpha, norm_y2, theta, y)

        if d_obj_from_inner > d_obj:
            d_obj = d_obj_from_inner
            theta[:] = theta_in
            # fcopy( & n_samples, & theta_in[0], & inc, & theta[0], & inc)

        highest_d_obj = d_obj

        p_obj = primal_lasso(alpha, R, y, w)
        gap = p_obj - highest_d_obj
        gaps[t] = gap
        if verbose:
            print("Iter %d: primal %.10f, gap %.2e" % (t, p_obj, gap), end="")

        if gap <= tol:
            if verbose:
                print("\nEarly exit, gap: %.2e < %.2e" % (gap, tol))
            break

        radius = np.sqrt(2 * gap / n_samples) / alpha

        n_screened = set_prios(
            theta, X, norms_X_col, prios, screened, radius, n_screened)

        if prune:
            nnz = 0
            for j in range(n_features):
                if w[j] != 0:
                    prios[j] = -1.
                    nnz += 1

            if t == 0:
                ws_size = p0 if nnz == 0 else nnz
            else:
                ws_size = 2 * nnz

        else:
            for j in range(n_features):
                if w[j] != 0:
                    prios[j] = - 1  # include active features
            if t == 0:
                ws_size = p0
            else:
                for j in range(ws_size):
                    if not screened[C[j]]:
                        # include previous features, if not screened
                        prios[C[j]] = -1
                ws_size = 2 * ws_size
        if ws_size > n_features - n_screened:
            ws_size = n_features - n_screened

        # if ws_size === n_features then argpartition will break:
        if ws_size == n_features:
            C = all_features
        else:
            C = np.argpartition(np.asarray(prios), ws_size)[
                :ws_size].astype(np.int32)

        for j in range(n_features):
            notin_WS[j] = 1
        for idx in range(ws_size):
            notin_WS[C[idx]] = 0

        if prune:
            tol_in = 0.3 * gap
        else:
            tol_in = tol

        if verbose:
            print(", %d feats in subpb (%d left)" %
                  (len(C), n_features - n_screened))

        # calling inner solver which will modify w and R inplace
        highest_d_obj_in = 0
        for epoch in range(max_epochs):
            if epoch != 0 and epoch % gap_freq == 0:
                create_dual_pt(alpha, theta_in, R)

                scal = dnorm_l1(theta_in, X, notin_WS)

                if scal > 1.:
                    theta_in /= scal

                d_obj_in = dual_lasso(alpha, norm_y2, theta_in, y)

                if True:  # also compute accelerated dual_point
                    info_dposv = create_accel_pt(
                        pb, n_samples, epoch, gap_freq, alpha, & Xw[0],
                        & thetacc[0], & last_K_Xw[0, 0], U, UtU, onesK, y)

                    if epoch // gap_freq >= K:
                        scal = dnorm_l1(thetacc, X, notin_WS)

                        if scal > 1.:
                            thetacc /= scal

                        d_obj_accel = dual_lasso(
                            alpha, norm_y2, thetacc, y)
                        if d_obj_accel > d_obj_in:
                            d_obj_in = d_obj_accel
                            theta_in[:] = theta_acc
                            # fcopy(& n_samples, & thetacc[0], & inc,
                            #    & theta_in[0], & inc)

                if d_obj_in > highest_d_obj_in:
                    highest_d_obj_in = d_obj_in

                # CAUTION: code does not yet  include a best_theta.
                # Can be an issue in screening: dgap and theta might disagree.

                p_obj_in = primal_lasso(alpha, R, y, w)  # TODO maybe small
                # improvement here
                gap_in = p_obj_in - highest_d_obj_in

                if verbose_in:
                    print("Epoch %d, primal %.10f, gap: %.2e" %
                          (epoch, p_obj_in, gap_in))
                if gap_in < tol_in:
                    if verbose_in:
                        print("Exit epoch %d, gap: %.2e < %.2e" %
                              (epoch, gap_in, tol_in))
                    break

            for k in range(ws_size):
                j = C[k]
                if norms_X_col[j] == 0.:
                    continue
                old_w_j = w[j]

                w[j] += X[0, j] @ R[0] * inv_lc[j]

                w[j] = ST(w[j], alpha * inv_lc[j] * n_samples)

                # R -= (w_j - old_w_j) (X[:, j]
                tmp = old_w_j - w[j]
                if tmp != 0.:
                    R += tmp * X[:, j]
        else:
            print("!!! Inner solver did not converge at epoch "
                  "%d, gap: %.2e > %.2e" % (epoch, gap_in, tol_in))


class Solver(BaseSolver):
    name = 'numba_mathurin'

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

    def run(self, n_iter):
        w = numba_celer(self.X, self.y, self.alpha, n_iter)
        self.w = w

    def st(self, w, mu):
        w -= np.clip(w, -mu, mu)
        return w

    def get_result(self):
        return self.w

    def compute_lipschitz_cste(self, n_iter=100):
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

        for _ in range(n_iter):
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
