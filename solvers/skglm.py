import warnings

from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from skglm import Lasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = "skglm"
    stopping_strategy = "iteration"

    # TODO: uncomment when available on pip
    # install_cmd = 'conda'
    # requirements = ['pip:skglm']
    references = [
        'Q. Bertrand and Q. Klopfenstein and P.-A. Bannier and G. Gidel and M. Massias'
        '"Beyond L1: Faster and Better Sparse Models with skglm", '
        'https://arxiv.org/abs/2204.07826'
    ]

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        n_samples = self.X.shape[0]
        self.lasso = Lasso(
            alpha=self.lmbd / n_samples, max_iter=1, max_epochs=50_000, tol=1e-12,
            fit_intercept=False, warm_start=False, verbose=False)
        
        # Cache Numba compilation
        self.run(1)
    
    def run(self, n_iter):
        self.lasso.max_iter = n_iter
        self.lasso.fit(self.X, self.y)
    
    @staticmethod
    def get_next(stop_val):
        return stop_val + 1
    
    def get_result(self):
        beta = self.lasso.coef_.flatten()
        if self.fit_intercept:
            beta = np.r_[beta, self.lasso.intercept_]
        return beta
