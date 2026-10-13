#!/usr/bin/env python3
"""scipy.interpolate.Rbf fixtures, verified against scipy. Rbf is a dense RBF-system solve (node weights from
(A - smooth*I) nodes = d, A_ij=phi(||p_i-p_j||)); Tessero\Interpolate\Rbf is a pure-PHP facade over
Tessero\Linalg\Linalg::solve. Covers all 7 built-in kernels, 1-D and 2-D scattered sites, smooth=0 and >0.
Coordinates are stored dim-major (coords[k] = k-th coordinate array). Host-independent (rtol tolerance)."""
import json
import os

import numpy as np
from scipy.interpolate import Rbf

OUT = os.environ.get('RBF_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_rbf.json')

rng = np.random.default_rng(20261013)
cases = []
KERNELS = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']


def add(coords, d, qcoords, function, smooth=0.0):
    coords = [np.asarray(c, float) for c in coords]
    d = np.asarray(d, float)
    qcoords = [np.asarray(c, float) for c in qcoords]
    rbf = Rbf(*coords, d, function=function, smooth=smooth)
    expect = rbf(*qcoords)
    cases.append({'fn': 'Rbf', 'function': function, 'smooth': float(smooth),
                  'coords': [c.tolist() for c in coords], 'd': d.tolist(),
                  'qcoords': [c.tolist() for c in qcoords],
                  'expect': np.asarray(expect, float).tolist()})


# 1-D
x = np.sort(rng.uniform(0.0, 5.0, 9))
d1 = np.sin(x)
xq = np.linspace(0.3, 4.7, 11)
for fn in KERNELS:
    add([x], d1, [xq], fn)
add([x], d1, [xq], 'multiquadric', smooth=0.1)
add([x], d1, [xq], 'gaussian', smooth=0.05)

# 2-D scattered
N = 12
px = rng.uniform(0.0, 3.0, N)
py = rng.uniform(0.0, 3.0, N)
d2 = np.cos(px) * np.sin(py)
qx = rng.uniform(0.3, 2.7, 7)
qy = rng.uniform(0.3, 2.7, 7)
for fn in KERNELS:
    add([px, py], d2, [qx, qy], fn)
add([px, py], d2, [qx, qy], 'linear', smooth=0.2)

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'rbf_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_rbf: {len(cases)} cases')
