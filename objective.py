import numpy as np
from numpy.linalg import norm

from benchopt import BaseObjective


class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "Lasso Regression"

    parameters = {
        'fit_intercept': [True, False],
        'reg': [.5, .1, .05],
    }

    def __init__(self, reg=.1, fit_intercept=False):
        self.reg = reg
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.lmbd = self.reg * self._get_lambda_max()
        self.n_features = self.X.shape[1]

    def get_one_solution(self):
        n_features = self.n_features
        if self.fit_intercept:
            n_features += 1
        return np.zeros(n_features)

    def compute(self, beta):
        beta = beta.astype(np.float64)  # avoid float32 numerical errors
        # compute residuals
        if self.fit_intercept:
            beta, intercept = beta[:self.n_features], beta[self.n_features:]
        diff = self.y - self.X @ beta
        if self.fit_intercept:
            diff -= intercept
        # compute primal objective and duality gap
        p_obj = .5 * diff.dot(diff) + self.lmbd * abs(beta).sum()
        scaling = max(1, norm(self.X.T @ diff, ord=np.inf) / self.lmbd)
        d_obj = (norm(self.y) ** 2 / 2.
                 - norm(self.y - diff / scaling) ** 2 / 2)
        return dict(value=p_obj,
                    support_size=(beta != 0).sum(),
                    duality_gap=p_obj - d_obj,)

    def _get_lambda_max(self):
        if self.fit_intercept:
            return abs(self.X.T @ (self.y - self.y.mean())).max()
        else:
            return abs(self.X.T.dot(self.y)).max()

    def get_objective(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd,
                    fit_intercept=self.fit_intercept)
