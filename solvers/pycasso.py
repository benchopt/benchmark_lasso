from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import pycasso
    import numpy as np
    from numpy.linalg import norm


class Solver(BaseSolver):
    name = "pycasso"
    stopping_strategy = 'iteration'

    requirements = ['pip:pycasso']

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y = np.asfortranarray(X), y
        self.lmbd = lmbd
        lmbd_max = norm(X.T @ y, ord=np.inf)
        lambdas = (2, lmbd / lmbd_max)  # (n_lambdas, lambda_min_ratio)
        self.clf = pycasso.Solver(
            X, y, lambdas=lambdas, penalty="l1",
            useintercept=fit_intercept, family="gaussian",
            verbose=False)
        self.clf.prec = 1e-20

    def run(self, n_iter):
        self.clf.max_ite = n_iter
        self.clf.train()

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def get_result(self):
        return self.clf.coef()['beta'][-1, :].flatten()
