from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from spams import lasso, fistaFlat
    import numpy as np
    import scipy

class Solver(BaseSolver):
    name = 'spams'

    install_cmd = 'conda'
    requirements = ['numpy','mkl','pip:spams']

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        if scipy.sparse.issparse(self.X):
            if scipy.sparse.isspmatrix_csr(self.X):
                self.X = scipy.spares.csc_matrix(self.X)
        else:
            self.X = np.asfortranarray(self.X)

        self.solver_parameter = dict(
            lambda1 = self.lmbd, verbose = True
        )

    def run(self, n_iter):
        y=np.expand_dims(np.asfortranarray(self.y), axis = 1)
        if (scipy.sparse.issparse(self.X)):
            W0 = np.zeros((self.X.shape[1],1), dtype = y.dtype, order = "F")
            self.w = fistaFlat(y,self.X,W0, **self.solver_parameter, regul = 'l1',
                    it0 = 10000, loss = 'square', tol = 1e-12, max_it = n_iter).flatten()
        else:
            self.w = lasso(y,  D = self.X, L = n_iter,
                        **self.solver_parameter).toarray().flatten()

    def get_result(self):
        return self.w
