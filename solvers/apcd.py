from benchopt import BaseSolver, safe_import_context

with safe_import_context() as ctx:
    import cd_solver


class Solver(BaseSolver):
    """Accelerated proximal coordinate descent by Fercoq and Bianchi."""

    name = 'apcd'
    stop_strategy = 'iteration'

    install_cmd = "conda"
    requirements = [
        "pip:git+https://github.com/Badr-MOUFAD/fercoq-approx@main"
    ]

    references = [
        'Fercoq, Olivier and Richt{\'a}rik, Peter, '
        'SIAM Journal on Optimization, '
        '"Accelerated, parallel, and proximal coordinate descent", '
        'vol. 25, pp. 1997--2023 (2015)'
    ]

    def set_objective(self, X, y, lmbd, fit_intercept):
        self.X, self.y, self.lmbd = X, y, lmbd
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        n_samples, n_features = self.X.shape
        f = ["square"] * n_samples
        N = n_features
        Af = self.X
        bf = self.y
        cf = [0.5] * n_samples
        g = ["abs"] * n_features
        cg = [self.lmbd] * n_features

        pb = cd_solver.Problem(N=N, f=f, Af=Af, bf=bf, cf=cf, g=g, cg=cg)
        cd_solver.cd_solver_.coordinate_descent(pb, max_iter=n_iter)
        self.w = pb.sol

    def get_result(self):
        return self.w
