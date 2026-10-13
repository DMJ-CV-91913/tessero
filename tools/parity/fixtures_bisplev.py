#!/usr/bin/env python3
r"""scipy.interpolate.bisplev fixtures, verified against scipy. A bivariate B-spline given by the FITPACK
tck = (tx, ty, c, kx, ky) is evaluated on the outer-product grid x[i] x y[j]; bisplev(x, y, tck) with
dx = dy = 0 is bit-identical to the tensor de Boor kernel Tessero\Interpolate::NdBSpline with c reshaped
to (ncx, ncy) (ncx = len(tx)-kx-1, ncy = len(ty)-ky-1). Deterministic tck inputs are built with bisplrep
(s=0) from a smooth grid; x and y are ascending (as bisplev requires). dx/dy derivatives are out of scope
(the NdBSpline kernel has no derivative path). Host-dependent spline fitting -> run under the pinned env
(NPY_DISABLE_CPU_FEATURES / OPENBLAS_CORETYPE=Haswell / single-threaded)."""
import json
import os

import numpy as np
from scipy.interpolate import bisplev, bisplrep

OUT = os.environ.get('BISPLEV_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_bisplev.json')

cases = []


def _grid_tck(n, kx, ky, f):
    g = np.linspace(0.0, 3.0, n)
    X, Y = np.meshgrid(g, g, indexing='ij')
    Z = f(X, Y)
    return bisplrep(X.ravel(), Y.ravel(), Z.ravel(), kx=kx, ky=ky, s=0.0)


def add(label, n, kx, ky, f, x, y):
    tx, ty, c, kkx, kky = _grid_tck(n, kx, ky, f)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.atleast_2d(np.asarray(bisplev(x, y, (tx, ty, c, kkx, kky)), float))
    # bisplev squeezes size-1 axes; our facade returns the full grid -> reshape expect to (len(x), len(y)).
    z = z.reshape(len(x), len(y))
    cases.append({
        'fn': 'bisplev',
        'label': label,
        'tx': np.asarray(tx, float).tolist(),
        'ty': np.asarray(ty, float).tolist(),
        'c': np.asarray(c, float).tolist(),
        'kx': int(kkx),
        'ky': int(kky),
        'x': x.tolist(),
        'y': y.tolist(),
        'expect': z.tolist(),
    })


# cubic x cubic, sin*cos surface
add('cubic_sincos', 9, 3, 3, lambda X, Y: np.sin(X) * np.cos(Y),
    [0.3, 1.1, 2.2, 2.9], [0.5, 1.7, 2.4])
# cubic x cubic, polynomial surface, different query grid
add('cubic_poly', 8, 3, 3, lambda X, Y: 1.0 + 0.5 * X - 0.3 * Y + 0.2 * X * Y + 0.1 * X * X,
    [0.2, 0.8, 1.5, 2.1, 2.7], [0.4, 1.0, 1.9, 2.8])
# linear x linear
add('linear', 7, 1, 1, lambda X, Y: 2.0 * X - Y + 3.0,
    [0.25, 1.25, 2.25], [0.75, 1.75, 2.75])
# quadratic x quadratic
add('quad', 8, 2, 2, lambda X, Y: np.exp(-0.2 * (X - 1.5) ** 2) * (1.0 + 0.3 * Y),
    [0.4, 1.2, 2.0, 2.6], [0.6, 1.4, 2.2])
# mixed degrees kx=3, ky=2
add('mixed_k3k2', 9, 3, 2, lambda X, Y: np.cos(0.7 * X) * np.sin(0.5 * Y) + 0.1 * X,
    [0.5, 1.3, 2.1, 2.8], [0.5, 1.5, 2.5])
# single query column (len(y)=1) -> still a full grid in the facade
add('col', 8, 3, 3, lambda X, Y: np.sin(X + Y),
    [0.3, 1.0, 1.8, 2.5], [1.25])

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'bisplev_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_bisplev: {len(cases)} cases')
