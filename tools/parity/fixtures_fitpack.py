#!/usr/bin/env python3
"""scipy.interpolate.UnivariateSpline (FITPACK curfit smoothing spline) fixtures,
verified against scipy 1.17.1. Tessero fits via the f2c-vendored Dierckx curfit
kernel (interpolate.splrep) and evaluates via FITPACK splev -- bit-exact to scipy.

Pinned env (AVX2 reference host): NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1."""
import json
import os

import numpy as np
from scipy.interpolate import UnivariateSpline

OUT = os.environ.get('FITPACK_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_fitpack.json')

rng = np.random.default_rng(20261014)
cases = []


def add(x, y, w, k, s):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    sp = UnivariateSpline(x, y, w=w, k=k, s=float(s))
    xq = np.linspace(float(x[0]), float(x[-1]), 23)
    cases.append({'fn': 'UnivariateSpline',
                  'x': x.tolist(), 'y': y.tolist(), 'w': w.tolist(),
                  'k': int(k), 's': float(s),
                  'query': xq.tolist(),
                  'expect': np.asarray(sp(xq), float).tolist()})


# cubic smoothing, light noise, moderate s
x = np.sort(np.unique(rng.uniform(0.0, 12.0, 40)))
y = np.sin(0.7 * x) + 0.05 * rng.normal(size=len(x))
add(x, y, np.ones(len(x)), 3, 0.5)

# cubic, more noise, larger s
x = np.sort(np.unique(rng.uniform(0.0, 10.0, 60)))
y = np.cos(0.5 * x) + 0.15 * rng.normal(size=len(x))
add(x, y, np.ones(len(x)), 3, 2.0)

# default-ish s = m - sqrt(2m) on 50 pts
x = np.sort(np.unique(rng.uniform(0.0, 8.0, 50)))
y = np.exp(-0.2 * x) * np.sin(x) + 0.02 * rng.normal(size=len(x))
m = len(x)
add(x, y, np.ones(m), 3, m - np.sqrt(2.0 * m))

# linear k=1
x = np.sort(np.unique(rng.uniform(0.0, 10.0, 35)))
y = 0.5 * x + 0.1 * rng.normal(size=len(x))
add(x, y, np.ones(len(x)), 1, 0.3)

# quartic k=4
x = np.sort(np.unique(rng.uniform(0.0, 6.0, 45)))
y = np.sin(x) ** 2 + 0.03 * rng.normal(size=len(x))
add(x, y, np.ones(len(x)), 4, 0.4)

# quintic k=5
x = np.sort(np.unique(rng.uniform(0.0, 9.0, 55)))
y = np.tanh(x - 4.0) + 0.04 * rng.normal(size=len(x))
add(x, y, np.ones(len(x)), 5, 0.6)

# weighted fit
x = np.sort(np.unique(rng.uniform(0.0, 10.0, 48)))
y = np.sin(0.9 * x) + 0.08 * rng.normal(size=len(x))
w = np.abs(rng.normal(size=len(x))) + 0.5
add(x, y, w, 3, 1.0)

payload = {'univariatespline_cases': cases}
with open(OUT, 'w') as f:
    json.dump(payload, f, allow_nan=False)
print(f'wrote {len(cases)} UnivariateSpline cases -> {OUT}')
