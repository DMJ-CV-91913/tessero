#!/usr/bin/env python3
"""scipy.spatial.distance.directed_hausdorff fixtures: the pure-PHP Tessero\\Spatial\\Hausdorff::directed
verified against scipy. Only the DISTANCE is compared (the returned index pair depends on scipy's internal
RNG shuffle and tie-break, so it is not asserted). Deterministic embedded point sets, host-independent --
no pinned environment required."""
import json
import os

import numpy as np
from scipy.spatial.distance import directed_hausdorff

OUT = os.environ.get('HAUS_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'spatial_hausdorff.json')

rng = np.random.default_rng(20261012)
cases = []
for _ in range(8):
    dim = int(rng.integers(2, 4))
    nu = int(rng.integers(2, 14))
    nv = int(rng.integers(2, 14))
    u = np.round(rng.standard_normal((nu, dim)), 10)
    v = np.round(rng.standard_normal((nv, dim)), 10)
    d = float(directed_hausdorff(u, v, seed=0)[0])
    cases.append({'fn': 'directed', 'u': u.tolist(), 'v': v.tolist(), 'expect': d})

# a few explicit deterministic cases (incl. an asymmetric pair and a single-point set)
explicit = [
    ([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], [[2., 0.], [0., 2.], [-2., 0.], [0., -4.]]),
    ([[0., 0., 0.], [1., 1., 1.]], [[0., 0., 0.], [2., 2., 2.], [3., 0., 0.]]),
    ([[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]),
]
for u, v in explicit:
    d = float(directed_hausdorff(np.asarray(u), np.asarray(v), seed=0)[0])
    cases.append({'fn': 'directed', 'u': u, 'v': v, 'expect': d})

out = {'module': 'spatial', 'scipy': __import__('scipy').__version__, 'hausdorff_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial_hausdorff: {len(cases)} cases')
