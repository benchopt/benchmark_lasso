import time
import numpy as np
from numpy.linalg import norm
from celer import Lasso
from libsvmdata import fetch_libsvm
import matplotlib.pyplot as plt

X, y = fetch_libsvm("rcv1.binary")
alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / len(y)

alpha = alpha_max/1000
clf = Lasso(alpha=alpha, warm_start=False, fit_intercept=False,
            max_iter=50, tol=1e-12, verbose=1)

times = []
E = []

for max_iter in range(0, 30, 2):
    clf.max_iter = max_iter
    t0 = time.time()
    clf.fit(X, y)
    times.append(time.time() - t0)
    E.append(np.mean((y - X @ clf.coef_)**2) /
             2. + alpha * norm(clf.coef_, ord=1))

assert clf.dual_gap_ < 2e-12
E = np.array(E)
plt.semilogy(times, E - E[-1])
plt.xlabel("Time (s)")
plt.show(block=False)
