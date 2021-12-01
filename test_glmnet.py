import numpy as np
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import celer
from numpy.linalg import norm

numpy2ri.activate()
glmnet = importr('glmnet')
as_matrix = robjects.r['as']

n = 10
p = 30

np.random.seed(1532)

y = np.random.randn(n)
X = np.random.randn(n, p)

lmbd_max = np.max(np.abs(X.T @ y)) / n

for ratio in [0.1, 1e-3]:
    lmbd = ratio * lmbd_max

    fit_dict2 = {"lambda": lmbd}
    fit2 = glmnet.glmnet(X, y, intercept=False,
                         standardize=False, **fit_dict2)
    results2 = dict(zip(fit2.names, list(fit2)))

    beta = np.squeeze(np.array(as_matrix(results2["beta"], "matrix")))

    clf = celer.Lasso(fit_intercept=False, alpha=lmbd, tol=1e-6).fit(X, y)
    print(f"lambda = {ratio}*lambda_max")
    print(beta)
    print(clf.coef_)
    print(f"norm of difference: {norm(beta - clf.coef_)}")
