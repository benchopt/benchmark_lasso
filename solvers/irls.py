from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.linear_model import Ridge


class Solver(BaseSolver):
    name = "IRLS"  # Iterative Reweighted Least Squares
    stopping_criterion = SufficientProgressCriterion(
        patience=6, strategy="iteration"
    )
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
        epsilon = 1e-25

        def w_opt(eta):
            ridge = Ridge(alpha=self.lmbd, fit_intercept=False).fit(
                self.X @ np.diag(eta ** 0.5), self.y
            )
            return ridge.coef_ * eta ** 0.5

        # # # Equivalent to
        # def w_opt(eta):
        #     T = self.X.T @ self.X + self.lmbd * np.diag(1.0 / eta)
        #     return np.linalg.solve(T, self.X.T @ self.y)

        def eta_opt(w):
            return (w ** 2 + epsilon) ** 0.5

        n_features = self.X.shape[1]
        eta = np.abs(self.X.T @ self.y)  # init needs to be > 0
        w = np.zeros(n_features)

        for i in range(n_iter):
            w = w_opt(eta)
            eta = eta_opt(w)

        self.w = w

    def get_result(self):
        return self.w
