from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import scipy
    from cyanure import Regression


class Solver(BaseSolver):
    name = 'Cyanure'

    install_cmd = 'conda'
    requirements = ['pip:cyanure-mkl']
    references = [
        'J. Mairal, "Cyanure: An Open-Source Toolbox for Empirical Risk'
        ' Minimization for Python, C++, and soon more," '
        'Arxiv eprint 1912.08165 (2019)'
    ]

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        if (scipy.sparse.issparse(self.X) and
                scipy.sparse.isspmatrix_csc(self.X)):
            self.X = scipy.sparse.csr_matrix(self.X)

        n_samples = self.X.shape[0]

        self.solver = Regression(loss='square', penalty='l1',
                                 fit_intercept=False)
        self.solver_parameter = dict(
            lambd=self.lmbd / n_samples, solver='auto', it0=1000000,
            tol=1e-12, verbose=False
        )

    def run(self, n_iter):
        self.solver.fit(self.X, self.y, max_epochs=n_iter,
                        **self.solver_parameter)

    def get_result(self):
        return self.solver.get_weights().flatten()
