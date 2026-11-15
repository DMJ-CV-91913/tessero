"""NumPy/SciPy side of bench/ml_bench.php: the same calls on the same inputs.

    OPENBLAS_NUM_THREADS=1 python3 bench/ml_bench.py calls > ml_calls_numpy.json   (us per call)
    OPENBLAS_NUM_THREADS=1 python3 bench/ml_bench.py heavy > ml_heavy_numpy.json   (ms per case)
"""
import json
import sys
import time

import numpy as np
from scipy import linalg, optimize, sparse, spatial, special, stats
from scipy.sparse import linalg as splinalg
from scipy.spatial import distance


def best(f, inner, rounds):
    """best of `rounds` runs of `inner` calls, in microseconds per call"""
    b = float('inf')
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(inner):
            f()
        b = min(b, (time.perf_counter() - t) / inner * 1e6)
    return b


part = sys.argv[1] if len(sys.argv) > 1 else 'calls'

if part == 'calls':
    x = np.array([0.5, -1.25, 2.0, 3.75, -0.5, 1.5, 4.25, 0.125])
    m = np.array([[2.0, 0.5, 0.25], [0.5, 3.0, 0.125], [0.25, 0.125, 4.0]])
    b = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]])
    bs = sparse.csr_matrix(b)
    pts = np.array([[0.1, 0.2], [0.4, 0.9], [0.8, 0.3], [0.5, 0.5], [0.9, 0.9], [0.2, 0.7], [0.6, 0.1], [0.3, 0.4]])
    norm = stats.norm()
    g = np.random.default_rng(12345)
    cases = {
        'numpy.matmul': lambda: np.matmul(m, m),
        'numpy.std': lambda: np.std(x),
        'numpy.var': lambda: np.var(x),
        'scipy.special.expit': lambda: special.expit(x),
        'scipy.sparse.hstack': lambda: sparse.hstack([bs, bs], format='csr'),
        'scipy.spatial.cKDTree': lambda: spatial.cKDTree(pts),
        'scipy.optimize.minimize': lambda: optimize.minimize(lambda v: (v[0] - 1.0) ** 2 + 10.0 * (v[1] - 2.0) ** 2, [0.0, 0.0], method='L-BFGS-B'),
        'scipy.optimize.least_squares': lambda: optimize.least_squares(lambda v: [v[0] - 1.0, 10.0 * (v[1] - 2.0), v[0] * v[1] - 2.0], [0.0, 0.0], method='lm'),
        'scipy.stats.norm.pdf': lambda: norm.pdf(x),
        'scipy.stats.norm.logpdf': lambda: norm.logpdf(x),
        'scipy.stats.norm.rvs': lambda: norm.rvs(8, random_state=g),
        'Generator.normal': lambda: g.normal(0.0, 1.0, 8),
        'Generator.uniform': lambda: g.uniform(0.0, 1.0, 8),
        'Generator.choice': lambda: g.choice(8, 4),
        'Generator.permutation': lambda: g.permutation(8),
    }
    out = {}
    for name, f in cases.items():
        f()
        out[name] = round(best(f, 50 if name.startswith('scipy.optimize') else 2000, 7), 4)
    print(json.dumps({'backend': 'numpy/scipy', 'unit': 'us', 'calls': out}, indent=1))
    sys.exit()

rng = np.random.default_rng(2026)
A2k = rng.random((2000, 2000))
B2k = rng.random((2000, 2000))
A1k = rng.random((1000, 1000))
S1k = A1k + A1k.T
P1k = A1k @ A1k.T + np.eye(1000) * 1000.0
b1k = rng.random(1000)
Xl = rng.random((10000, 50))
yl = rng.random(10000)
Xp = rng.random((2000, 50))
Xa = rng.random((1000, 50))
Xb = rng.random((1000, 50))
Xs = rng.random((2000, 500))
pts = rng.random((100000, 3))
qpts = rng.random((1000, 3))
v1m = rng.standard_normal(1000000)
Xlr = rng.standard_normal((10000, 50))
wtrue = rng.standard_normal(50)
ylr = (Xlr @ wtrue > 0.0).astype(np.float64) * 2.0 - 1.0


def lr_fun(w):
    nyz = -((Xlr @ w) * ylr)
    return float(np.mean(np.log1p(np.exp(nyz))))


def lr_jac(w):
    nyz = -((Xlr @ w) * ylr)
    return Xlr.T @ -(special.expit(nyz) * ylr) / 10000.0


w0 = np.zeros(50)
tree = spatial.cKDTree(pts)
cases = {
    'numpy.matmul 2000x2000': lambda: A2k @ B2k,
    'scipy.linalg.svd 1000x1000 (full)': lambda: linalg.svd(A1k),
    'scipy.linalg.eigh 1000x1000': lambda: linalg.eigh(S1k),
    'scipy.linalg.solve 1000x1000': lambda: linalg.solve(A1k, b1k),
    'scipy.linalg.cholesky 1000x1000': lambda: linalg.cholesky(P1k),
    'scipy.linalg.lstsq 10,000 x 50': lambda: linalg.lstsq(Xl, yl),
    'scipy.sparse.linalg.svds k=10, 2,000 x 500 (dense input)': lambda: splinalg.svds(Xs, 10, return_singular_vectors=False),
    'scipy.spatial.distance.pdist 2,000 x 50 (euclidean)': lambda: distance.pdist(Xp),
    'scipy.spatial.distance.cdist 1,000 x 1,000 x 50 (euclidean)': lambda: distance.cdist(Xa, Xb),
    'scipy.spatial.cKDTree build 100,000 x 3': lambda: spatial.cKDTree(pts),
    'scipy.spatial.cKDTree query 1,000 points, k=5': lambda: tree.query(qpts, 5),
    'logistic regression, L-BFGS-B, 10,000 x 50': lambda: optimize.minimize(lr_fun, w0, jac=lr_jac, method='L-BFGS-B'),
    'logistic regression, Newton-CG, 10,000 x 50': lambda: optimize.minimize(lr_fun, w0, jac=lr_jac, method='Newton-CG'),
    'logistic regression, TNC, 10,000 x 50': lambda: optimize.minimize(lr_fun, w0, jac=lr_jac, method='TNC'),
    'scipy.special.expit 1M': lambda: special.expit(v1m),
    'scipy.special.logsumexp 1M': lambda: special.logsumexp(v1m),
    'scipy.stats.norm.logpdf 1M': lambda: stats.norm.logpdf(v1m),
}
out = {}
for name, f in cases.items():
    t0 = time.perf_counter()
    r = f()
    first = (time.perf_counter() - t0) * 1e3
    if name.startswith('logistic'):
        print(f"{name}: nit {getattr(r, 'nit', '?')} nfev {r.nfev} fun {r.fun:.12g}", file=sys.stderr)
    rounds = 2 if first > 2000 else (3 if first > 200 else 5)
    inner = 20 if first < 2 else 1
    out[name] = round(best(f, inner, rounds) / 1e3, 4)
print(json.dumps({'backend': 'numpy/scipy', 'unit': 'ms', 'cases': out}, indent=1))
