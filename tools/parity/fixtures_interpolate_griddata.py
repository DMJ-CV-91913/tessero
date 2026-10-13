#!/usr/bin/env python3
"""scipy.interpolate.griddata(method='nearest') fixtures (ndim>=2), verified against scipy. Nearest-neighbour
scattered interpolation == Tessero\\Interpolate\\NearestNDInterpolator under the hood (a gather, exact). Only
method='nearest' is covered here (linear/cubic need a Delaunay triangulation that Tessero does not have);
1-D nearest routes through interp1d in scipy and is intentionally excluded. Host-independent, no pinned env."""
import json
import os

import numpy as np
from scipy.interpolate import griddata

OUT = os.environ.get('GRIDDATA_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_griddata.json')

rng = np.random.default_rng(20261012)
cases = []


def add(points, values, xi, rescale=False):
    expect = griddata(np.asarray(points), np.asarray(values), np.asarray(xi), method='nearest', rescale=rescale)
    cases.append({'fn': 'griddata', 'points': np.asarray(points, float).tolist(),
                  'values': np.asarray(values, float).tolist(), 'query': np.asarray(xi, float).tolist(),
                  'rescale': bool(rescale), 'expect': np.asarray(expect, float).tolist()})


# 2-D scalar
add(rng.random((40, 2)), rng.random(40), rng.random((15, 2)))
# 2-D scalar, rescale=True
add(rng.random((30, 2)), rng.random(30), rng.random((10, 2)), rescale=True)
# 3-D scalar
add(rng.random((35, 3)), rng.random(35), rng.random((8, 3)))
# 2-D vector-valued values (n, k)
add(rng.random((40, 2)), rng.random((40, 3)), rng.random((12, 2)))
# a small deterministic grid with an unambiguous nearest (avoid exact ties)
add([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [10.0, 20.0, 30.0, 40.0], [[0.1, 0.1], [0.9, 0.2], [0.2, 0.8]])

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'griddata_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_griddata: {len(cases)} cases')
