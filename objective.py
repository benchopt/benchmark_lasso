from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Lasso Regression"

    parameters = {
        'reg': [0.05, .1, .5],
        'fit_intercept': [False]
    }

    def __init__(self, reg=.1, fit_intercept=False):
        self.reg = reg
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.lmbd = self.reg * self._get_lambda_max()
        self.n_features = self.X.shape[1]

    def compute(self, beta):
        if self.fit_intercept:
            beta, intercept = beta[:self.n_features], beta[self.n_features:]
        diff = self.y - self.X.dot(beta)
        if self.fit_intercept:
            diff -= intercept
        return .5 * diff.dot(diff) + self.lmbd * abs(beta).sum()

    def _get_lambda_max(self):
        return abs(self.X.T.dot(self.y)).max()

    def to_dict(self):
        return dict(X=self.X, y=self.y, lmbd=self.lmbd,
                    fit_intercept=self.fit_intercept)
