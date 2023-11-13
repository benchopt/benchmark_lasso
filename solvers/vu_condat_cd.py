from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from cd_solver import Problem, coordinate_descent


class Solver(BaseSolver):
    name = 'vu-condat-cd'

    install_cmd = 'conda'
    requirements = [
        'pip:'
        'git+https://github.com/Badr-MOUFAD/Fercoq-Bianchi-solver.git@main'
    ]

    references = [
        "Olivier Fercoq and Pascal Bianchi, "
        "'A coordinate descent primal-dual algorithm with large step size "
        "and possibly non separable functions', "
        "SIAM Journal on Optimization, 29(1), 100-134, "
        "https://arxiv.org/pdf/1508.04625.pdf, "
        "code: https://github.com/Badr-MOUFAD/Fercoq-Bianchi-solver",
        "Olivier Fercoq, "
        "'A generic coordinate descent solver for nonsmooth convex optimization', "
        "Optimization Methods and Software, 2019, pp.1-21, "
        "https://hal.archives-ouvertes.fr/hal-01941152v2/document"
    ]

    parameters = {
        'smooth_formulation': [True, False]
    }

    def __init__(self, smooth_formulation):
        self.smooth_formulation = smooth_formulation

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y = X, y
        self.lmbd = lmbd

        # when smooth_formulation=True, the estimator considers
        # the datafit term smooth and uses its gradient to find the min.
        # Otherwise, estimator considers the datafit non-smooth
        # and uses it prox operator instead
        n_samples = X.shape[0]
        if self.smooth_formulation:
            self.datafit_params = {
                'f': ["square"], 'Af': X, 'bf': y,
                'blocks_f': [0, n_samples],
                'cf': [0.5],
            }
        else:
            self.datafit_params = {
                'h': ["square"], 'Ah': X, 'bh': y,
                'blocks_h': [0, n_samples],
                'ch': [0.5],
            }

    def run(self, n_iter):
        n_features = self.X.shape[1]

        if n_iter == 0:
            self.w = np.zeros(n_features)
            return

        pb = Problem(
            N=n_features,
            # datafit
            **self.datafit_params,
            # penalty
            g=["abs"] * n_features, cg=[self.lmbd] * n_features
        )
        coordinate_descent(pb, max_iter=n_iter, per_pass=1)

        self.w = pb.sol.flatten()

    def get_result(self):
        return self.w
