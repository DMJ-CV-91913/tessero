#!/usr/bin/env python3
"""scipy.interpolate.LSQUnivariateSpline fixtures, verified against scipy. A least-squares B-spline
fit with a user-supplied set of INTERIOR knots t and degree k, evaluated at query points. Under the
hood Tessero delegates to interpolate.make_lsq_spline after augmenting the interior knots to the full
clamped knot vector [x0]*(k+1)+interior+[xN]*(k+1) -- which is exactly what SciPy's LSQUnivariateSpline
does internally (verified: identical to make_lsq_spline on the augmented vector, ~1e-15). A couple of k
values with explicit interior knots at data quantiles, evaluated on a grid of query points.

Pinned env (AVX2 reference host): NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1."""
import json
import os

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

OUT = os.environ.get('LSQ_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_lsq.json')

rng = np.random.default_rng(20261012)
cases = []


def add(x, y, interior, k):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    interior = np.asarray(interior, float)
    sp = LSQUnivariateSpline(x, y, interior, k=k)
    xq = np.linspace(x[0], x[-1], 17)
    cases.append({'fn': 'LSQUnivariateSpline',
                  'x': x.tolist(), 'y': y.tolist(),
                  't': interior.tolist(), 'k': int(k),
                  'query': xq.tolist(),
                  'expect': np.asarray(sp(xq), float).tolist()})


# cubic fit, 3 interior knots at data quantiles (the classic well-conditioned case)
x = np.sort(np.unique(rng.uniform(0.0, 10.0, 32)))
y = np.sin(0.8 * x) + 0.1 * rng.normal(size=len(x))
interior = np.quantile(x, np.linspace(0, 1, 5)[1:-1])
add(x, y, interior, 3)

# linear fit (k=1), same data, 4 interior knots
x = np.sort(np.unique(rng.uniform(0.0, 10.0, 36)))
y = np.cos(0.5 * x) + 0.05 * rng.normal(size=len(x))
interior = np.quantile(x, np.linspace(0, 1, 6)[1:-1])
add(x, y, interior, 1)

# quadratic fit (k=2)
x = np.sort(np.unique(rng.uniform(-2.0, 2.0, 40)))
y = x ** 2 + 0.05 * rng.normal(size=len(x))
interior = np.quantile(x, np.linspace(0, 1, 5)[1:-1])
add(x, y, interior, 2)

# quintic fit (k=5), smooth target, plenty of data per coefficient
x = np.sort(np.unique(rng.uniform(0.0, 6.0, 48)))
y = np.exp(-0.2 * x) * np.sin(x)
interior = np.quantile(x, np.linspace(0, 1, 4)[1:-1])
add(x, y, interior, 5)

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'lsq_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_lsq: {len(cases)} cases')
