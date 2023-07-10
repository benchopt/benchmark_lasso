import warnings
from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.linear_model import Lasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'sklearn'

    tags = ["first-order", "Coordinate-Descent"]

    install_cmd = 'conda'
    requirements = ['scikit-learn']
    references = [
        'F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, '
        'O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, '
        'J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot'
        ' and E. Duchesnay'
        '"Scikit-learn: Machine Learning in Python", J. Mach. Learn. Res., '
        'vol. 12, pp. 2825-283 (2011)'
    ]

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        n_samples = self.X.shape[0]
        self.lasso = Lasso(alpha=self.lmbd/n_samples,
                           fit_intercept=fit_intercept, tol=0)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1] + self.fit_intercept])
        else:
            self.lasso.max_iter = n_iter
            self.lasso.fit(self.X, self.y)

            coef = self.lasso.coef_.flatten()
            if self.fit_intercept:
                coef = np.r_[coef, self.lasso.intercept_]
            self.coef = coef

    def get_result(self):
        return self.coef
