"""SciPy timings for the cases of bench/scipy_bench.php (median of N runs, ms; same inputs, same seeds).

    OPENBLAS_NUM_THREADS=1 python3 bench/scipy_bench.py > scipy.json
"""
import json
import time

import numpy as np
import scipy.linalg as sl
import scipy.ndimage as nd
import scipy.optimize as so
import scipy.signal as sg
import scipy.special as sp
import scipy.stats as st
from scipy.spatial import distance as D


def med(f, n=11, inner=1):
    f()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        for _ in range(inner):
            f()
        t.append((time.perf_counter() - s) * 1e3 / inner)
    t.sort()
    return t[n // 2]


def r(seed, size):
    return np.random.default_rng(seed).random(size)


A = r(1, (600, 600)) + 600 * np.eye(600)
b = r(2, 600)
M3 = r(3, (300, 300))
S3 = M3 @ M3.T + 300 * np.eye(300)
M4 = r(4, (300, 300))
E = r(5, (200, 200)) * 0.01
x = r(10, 1000000) * 10.0
y = r(11, 1000000)
ta = r(12, 100000)
tb = r(13, 100000) + 0.01
sig = r(14, 100000)
ker = r(15, 1001)
s10 = r(14, 10000)
k101 = r(15, 101)
XA = r(16, (1000, 3))
XB = r(17, (1000, 3))
X2 = r(18, (2000, 3))
img = r(19, (1000, 1000))
g = np.random.default_rng(20)
lpA = g.uniform(0.1, 1.0, (200, 200))
lpB = g.uniform(50, 100, 200)
lpC = g.uniform(-1, 0, 200)
xm5 = x - 5.0
xp = x + 0.1

cases = {
    'scipy.linalg.solve 600x600': lambda: sl.solve(A, b),
    'scipy.linalg.lu_factor 600x600': lambda: sl.lu_factor(A),
    'scipy.linalg.inv 600x600': lambda: sl.inv(A),
    'scipy.linalg.det 600x600': lambda: sl.det(A),
    'scipy.linalg.eigh 300x300': lambda: sl.eigh(S3),
    'scipy.linalg.svd 300x300': lambda: sl.svd(M4),
    'scipy.linalg.expm 200x200': lambda: sl.expm(E),
    'scipy.special.erf 1M': lambda: sp.erf(x),
    'scipy.special.gamma 1M': lambda: sp.gamma(xp),
    'scipy.special.jv(1.5, x) 1M': lambda: sp.jv(1.5, x),
    'scipy.special.expit 1M': lambda: sp.expit(xm5),
    'scipy.special.logsumexp 1M': lambda: sp.logsumexp(x),
    'scipy.stats.norm.cdf 1M': lambda: st.norm.cdf(xm5),
    'scipy.stats.describe 1M': lambda: st.describe(x),
    'scipy.stats.zscore 1M': lambda: st.zscore(x),
    'scipy.stats.pearsonr 1M': lambda: st.pearsonr(x, y),
    'scipy.stats.ttest_ind 100k vs 100k': lambda: st.ttest_ind(ta, tb),
    'scipy.signal.fftconvolve 100k * 1001': lambda: sg.fftconvolve(sig, ker),
    'scipy.signal.convolve 10k * 101': lambda: sg.convolve(s10, k101),
    'scipy.signal.lfilter order 2, 1M': lambda: sg.lfilter([0.0675, 0.1349, 0.0675], [1.0, -1.143, 0.4128], x),
    'scipy.spatial.distance.cdist 1000x1000x3': lambda: D.cdist(XA, XB, 'euclidean'),
    'scipy.spatial.distance.pdist 2000x3': lambda: D.pdist(X2, 'euclidean'),
    'scipy.ndimage.gaussian_filter 1000x1000': lambda: nd.gaussian_filter(img, 2.0),
    'scipy.ndimage.uniform_filter 1000x1000': lambda: nd.uniform_filter(img, 5),
    'scipy.optimize.linprog 200x200 dense': lambda: so.linprog(lpC, A_ub=lpA, b_ub=lpB),
    'scipy.optimize.minimize rosen BFGS n=5': lambda: so.minimize(so.rosen, [1.3, 0.7, 0.8, 1.9, 1.2], method='BFGS'),
    'scipy.optimize.minimize_scalar brent': lambda: so.minimize_scalar(lambda t: (t - 2.0) ** 2),
    'scipy.optimize.brentq': lambda: so.brentq(lambda t: t ** 3 - 2 * t - 5, 2.0, 3.0),
}

out = {}
for k, f in cases.items():
    small = 'brent' in k or 'minimize' in k
    out[k] = med(f, 11, 200 if small else 1)
print(json.dumps(out, indent=1))
