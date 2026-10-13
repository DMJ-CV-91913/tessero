#!/usr/bin/env python3
"""scipy.interpolate.RectBivariateSpline(x, y, z, kx, ky, s=0) fixtures, verified against scipy. The s=0
(exactly interpolating) tensor-product spline on a regular grid == Tessero\\Interpolate\\RectBivariateSpline,
a pure-PHP facade: two sequential 1-D interpolating-spline solves (make_interp_spline IUS/not-a-knot knot
pattern, built in PHP) followed by the existing tensor de Boor kernel routine interpolate.NdBSpline. Scope is
s=0 with kx, ky in {1, 3} (linear or not-a-knot cubic), matching Tessero's 1-D interpolating spline. The
evaluation grid is the tensor product xq x yq, as RectBivariateSpline(x, y, z)(xq, yq). Host-independent."""
import json
import os

import numpy as np
from scipy.interpolate import RectBivariateSpline

OUT = os.environ.get('RBS_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_rbs.json')

rng = np.random.default_rng(20261012)
cases = []


def add(x, y, z, xq, yq, kx=3, ky=3):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    xq = np.asarray(xq, float)
    yq = np.asarray(yq, float)
    expect = RectBivariateSpline(x, y, z, kx=kx, ky=ky, s=0)(xq, yq)
    cases.append({'fn': 'RectBivariateSpline', 'x': x.tolist(), 'y': y.tolist(), 'z': z.tolist(),
                  'kx': int(kx), 'ky': int(ky), 'xq': xq.tolist(), 'yq': yq.tolist(),
                  'expect': np.asarray(expect, float).tolist()})


# cubic x cubic, smooth analytic surface on a uniform grid
x = np.linspace(0.0, 6.0, 7)
y = np.linspace(0.0, 5.0, 6)
X, Y = np.meshgrid(x, y, indexing='ij')
add(x, y, np.sin(X) * np.cos(Y), np.linspace(0.5, 5.5, 5), np.linspace(0.3, 4.7, 4), kx=3, ky=3)

# cubic x cubic on a non-uniform grid with random values
x = np.sort(rng.uniform(0.0, 10.0, 8))
y = np.sort(rng.uniform(-3.0, 3.0, 7))
add(x, y, rng.normal(size=(8, 7)), np.linspace(x[0] + 0.1, x[-1] - 0.1, 6),
    np.linspace(y[0] + 0.1, y[-1] - 0.1, 5), kx=3, ky=3)

# linear x linear (kx=ky=1): piecewise-bilinear interpolation
x = np.linspace(0.0, 4.0, 5)
y = np.linspace(0.0, 3.0, 4)
X, Y = np.meshgrid(x, y, indexing='ij')
add(x, y, (X**2 - 2.0 * Y), np.linspace(0.2, 3.8, 5), np.linspace(0.1, 2.9, 4), kx=1, ky=1)

# mixed degrees: cubic in x, linear in y
x = np.linspace(0.0, 2.0 * np.pi, 9)
y = np.linspace(0.0, 1.0, 5)
X, Y = np.meshgrid(x, y, indexing='ij')
add(x, y, np.cos(X) * (1.0 + Y), np.linspace(0.3, 6.0, 6), np.linspace(0.05, 0.95, 4), kx=3, ky=1)

# mixed degrees: linear in x, cubic in y, random grid values
x = np.sort(rng.uniform(0.0, 5.0, 6))
y = np.sort(rng.uniform(0.0, 8.0, 7))
add(x, y, rng.normal(size=(6, 7)), np.linspace(x[0] + 0.1, x[-1] - 0.1, 5),
    np.linspace(y[0] + 0.1, y[-1] - 0.1, 6), kx=1, ky=3)

# single query point (scalar-ish grid 1x1)
x = np.linspace(0.0, 10.0, 11)
y = np.linspace(0.0, 10.0, 11)
X, Y = np.meshgrid(x, y, indexing='ij')
add(x, y, np.exp(-0.1 * (X + Y)), [4.3], [7.1], kx=3, ky=3)

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'rbs_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_rbs: {len(cases)} cases')
