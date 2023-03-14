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

    install_cmd = "conda"
    requirements = ["scikit-learn"]

    references = [
        "Y. Grandvalet, "
        '"Least absolute shrinkage is equivalent to quadratic penalization" '
        "International Conference on Artificial Neural Networks, "
        "pp. 201-206, (1998)",
    ]
    # see also: https://homepages.laas.fr/vmagron/masmode/Gabriel.pdf

    def skip(self, X, y, lmbd, fit_intercept):
        # XXX - not implemented but not too complicated to implement
        if fit_intercept:
            return True, f"{self.name} does not handle fit_intercept"

        return False, None

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        def w_opt(eta):
            sqrt_eta = np.sqrt(eta)
            ridge = Ridge(alpha=self.lmbd, fit_intercept=False).fit(
                self.X * sqrt_eta, self.y
            )
            return ridge.coef_ * sqrt_eta

        def eta_opt(w):
            return np.abs(w)

        n_features = self.X.shape[1]
        eta = np.ones(n_features)  # first iteration: non weighted Ridge
        w = np.zeros(n_features)

        for i in range(n_iter):
            w = w_opt(eta)
            eta = eta_opt(w)

        self.w = w

    def get_result(self):
        return self.w
