from benchopt.datasets.simulated import make_correlated_data
from sklearn.linear_model import ElasticNet
import glmnet
from numpy.linalg import norm
import numpy as np

X, y, _ = make_correlated_data(100, 5000, rho=0.6, random_state=27)
alpha_max = norm(X.T @ y, ord=np.inf) / len(y)


clf = ElasticNet(alpha=alpha_max / 2, l1_ratio=1,
                 fit_intercept=False, tol=1e-10).fit(X, y)


clf2 = glmnet.ElasticNet(alpha=1, lambda_path=[
                         alpha_max, alpha_max/2], standardize=False, fit_intercept=False, tol=1e-10, max_iter=10000).fit(X, y)


print(norm(clf.coef_ - clf2.coef_) / norm(clf.coef_))
