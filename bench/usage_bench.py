"""NumPy/SciPy side of bench/usage_bench.php: the same calls on the same inputs, best-of per-call times in us.

    OPENBLAS_NUM_THREADS=1 python3 bench/usage_bench.py > usage_numpy.json
"""
import json
import time

import numpy as np
from scipy import stats

lst = [0.5, -1.25, 2.0, 3.75, -0.5, 1.5, 4.25, 0.125]
x = np.array(lst)
pos = np.array([0.5, 1.25, 2.0, 3.75, 0.25, 1.5, 4.25, 0.125])
y = np.array([1.0, 0.5, -2.0, 0.25, 3.0, -1.5, 0.75, 2.5])
m = np.array([[2.0, 0.5, 0.25], [0.5, 3.0, 0.125], [0.25, 0.125, 4.0]])
mask = x > 0.0
norm = stats.norm()
chi2 = stats.chi2(3.0)

cases = {
    'np.array': lambda: np.array(lst),
    'np.asarray': lambda: np.asarray(lst),
    'np.zeros': lambda: np.zeros(8),
    'np.dot': lambda: np.dot(m, m),
    'np.arange': lambda: np.arange(8),
    'np.sqrt': lambda: np.sqrt(pos),
    'np.sum': lambda: np.sum(x),
    'np.absolute': lambda: np.absolute(x),
    'np.log': lambda: np.log(pos),
    'np.empty': lambda: np.empty(8),
    'np.where': lambda: np.where(mask, x, y),
    'np.ones': lambda: np.ones(8),
    'np.exp': lambda: np.exp(x),
    'np.any': lambda: np.any(mask),
    'np.isnan': lambda: np.isnan(x),
    'np.mean': lambda: np.mean(x),
    'np.max': lambda: np.max(x),
    'np.all': lambda: np.all(mask),
    'np.eye': lambda: np.eye(3),
    'np.linspace': lambda: np.linspace(0.0, 1.0, 8),
    'np.isfinite': lambda: np.isfinite(x),
    'np.clip': lambda: np.clip(x, -1.0, 1.0),
    'np.cos': lambda: np.cos(x),
    'np.maximum': lambda: np.maximum(x, y),
    'np.sin': lambda: np.sin(x),
    'np.argsort': lambda: np.argsort(x),
    'np.min': lambda: np.min(x),
    'np.full': lambda: np.full(8, 2.5),
    'np.cumsum': lambda: np.cumsum(x),
    'np.ceil': lambda: np.ceil(x),
    'np.sort': lambda: np.sort(x),
    'np.reshape': lambda: np.reshape(x, (2, 4)),
    'np.ravel': lambda: np.ravel(m),
    'np.argmax': lambda: np.argmax(x),
    'np.prod': lambda: np.prod(x),
    'np.log10': lambda: np.log10(pos),
    'np.logical_and': lambda: np.logical_and(mask, mask),
    'np.sign': lambda: np.sign(x),
    'np.floor': lambda: np.floor(x),
    'np.minimum': lambda: np.minimum(x, y),
    'np.argmin': lambda: np.argmin(x),
    'np.power': lambda: np.power(pos, 1.5),
    'stats.norm': lambda: norm.cdf(x),
    'stats.chi2': lambda: chi2.sf(pos),
}

out = {}
for name, f in cases.items():
    f()
    best = float('inf')
    for _ in range(7):
        t = time.perf_counter()
        for _ in range(2000):
            f()
        best = min(best, (time.perf_counter() - t) / 2000 * 1e6)
    out[name] = round(best, 4)
print(json.dumps({'backend': 'numpy/scipy', 'unit': 'us', 'calls': out}, indent=1))
