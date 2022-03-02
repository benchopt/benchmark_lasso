from celer import Lasso
from cd_solver import Problem
import cd_solver
from numpy.linalg import norm
import numpy as np
from benchopt.datasets import make_correlated_data


X, y, _ = make_correlated_data(random_state=1)

n_samples, n_features = X.shape
alpha = norm(X.T @ y, ord=np.inf) / 10 / n_samples


f = ["square"] * n_samples
N = n_features
Af = X
bf = y
cf = [0.5 / n_samples] * n_samples
g = ["abs"] * n_features
cg = [alpha] * n_features
pb = Problem(N=N, f=f, Af=Af, bf=bf, cf=cf, g=g, cg=cg)


cd_solver.cd_solver_.coordinate_descent(
    pb, max_iter=1000)

print(pb.sol)


clf = Lasso(fit_intercept=False, alpha=alpha).fit(X, y)
print(clf.coef_)
