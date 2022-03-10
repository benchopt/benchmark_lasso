import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import norm
from benchopt.datasets import make_correlated_data

from scipy import optimize as sciop

X, y, _ = make_correlated_data(200, 500, rho=0.5, random_state=27)

lmbd = 0.5 * norm(X.T @ y, ord=np.inf)
n_samples, n_features = X.shape

t0 = time.time()


def u_opt(v):
    S = X @ np.diag(v**2) @ X.T + lmbd * np.eye(n_samples)
    return v * (X.T @ np.linalg.solve(S, y))


def nabla_f(v):
    u = u_opt(v)
    f = 1/(2 * lmbd) * norm(X @ (u*v) - y)**2 + \
        (norm(u)**2 + norm(v)**2) / 2
    g = u * (X.T @ (X @ (u*v) - y)) / lmbd + v
    return f, g


n_iter = 100
# run lbfgs
opts = {'gtol': 1e-8, 'maxiter': n_iter, 'maxcor': 100, 'ftol': 0}
u0 = np.ones(n_features)

objs = []
times = []


def callback(v):
    f, g = nabla_f(v)
    objs.append(f)
    times.append(time.time() - t0)
    return f, g


lbfgs_res = sciop.minimize(
    callback, u0, method='L-BFGS-B', jac=True, options=opts
)
v = lbfgs_res.x

# v2: ala benchopt
n_iters = np.geomspace(1, 300, 20).astype(int)
n_reps = 10


objs2 = np.zeros([n_reps, len(n_iters)])
times2 = objs2.copy()

for rep in range(n_reps):
    print(f"rep {rep}")
    for ix, n_iter in enumerate(n_iters):
        t0 = time.time()
        opts = {'gtol': 1e-8, 'maxiter': n_iter, 'maxcor': 100, 'ftol': 0}

        def u_opt(v):
            S = X @ np.diag(v**2) @ X.T + lmbd * np.eye(n_samples)
            return v * (X.T @ np.linalg.solve(S, y))

        def nabla_f(v):
            u = u_opt(v)
            f = 1/(2 * lmbd) * norm(X @ (u*v) - y)**2 + \
                (norm(u)**2 + norm(v)**2) / 2
            g = u * (X.T @ (X @ (u*v) - y)) / lmbd + v
            return f, g

        lbfgs_res = sciop.minimize(
            nabla_f, u0, method='L-BFGS-B', jac=True, options=opts
        )
        objs2[rep, ix] = lbfgs_res.fun
        times2[rep, ix] = time.time() - t0

# objs = np.array(objs)
# objs2 = np.array(objs2)
plt.close('all')
# plt.semilogy(times, objs - np.min(objs), label="callback")
for rep in range(n_reps):
    plt.semilogy(times2[rep], objs2[rep] - np.min(objs2), label=f'{rep}')
plt.legend()
plt.show(block=False)
