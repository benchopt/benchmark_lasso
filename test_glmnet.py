import sys
from timeit import default_timer as timer

import benchopt
from benchopt.datasets import make_correlated_data
import numpy as np
from numpy.linalg import norm
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
from celer import Lasso
import time

# Setup the system to allow rpy2 running
numpy2ri.activate()
glmnet = importr('glmnet')

as_matrix = robjects.r['as']

n = 100
p = 500

np.random.seed(1532)

y = np.random.randn(n)
x, y, _ = make_correlated_data(n, p, random_state=0)

lmbd_max = np.max(np.abs(x.T @ y))

lmbd = lmbd_max * 0.01

fit_dict = {"lambda": lmbd / n}

tols = np.geomspace(1e-1, 1e-16, 11)


def compute_gap(x, y, beta, lmbd):
    # compute residuals
    diff = y - x.dot(beta)

    # compute primal objective and duality gap
    p_obj = .5 * diff.dot(diff) + lmbd * np.abs(beta).sum()
    theta = diff / lmbd
    theta /= norm(x.T @ theta, ord=np.inf)
    d_obj = (norm(y)**2 / 2. - lmbd**2 * norm(y / lmbd - theta)**2 / 2)

    return p_obj - d_obj


n_tols = len(tols)

times = np.empty(n_tols)
gaps = np.empty(n_tols)

for i, tol in enumerate(tols):
    print(i, tol)
    t0 = timer()

    fit = glmnet.glmnet(x,
                        y,
                        intercept=False,
                        standardize=False,
                        **fit_dict,
                        thresh=tol,
                        maxit=10_000_000)

    times[i] = timer() - t0

    results = dict(zip(fit.names, list(fit)))
    coefs = np.array(as_matrix(results["beta"], "matrix"))
    beta = coefs.flatten()

    gaps[i] = compute_gap(x, y, beta, lmbd)

plt.close('all')
plt.semilogy(times, gaps)
plt.xlabel("time (s)")
plt.ylabel("duality gap")
plt.show(block=False)


t0 = time.time()
clf = Lasso(alpha=lmbd/n, fit_intercept=False, tol=1e-14, verbose=1).fit(x, y)
t1 = time.time()
print(t1 - t0)
