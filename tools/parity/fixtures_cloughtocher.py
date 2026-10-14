#!/usr/bin/env python3
"""scipy.interpolate.CloughTocher2DInterpolator fixtures (pure-PHP Tessero\\Interpolate\\CloughTocher2DInterpolator
over an inlined Bowyer-Watson 2-D Delaunay + SciPy's global minimum-curvature gradient estimate + the C1 cubic
Bezier per triangle): construct + query. 2-D, scalar, GENERAL POSITION points only (continuous random scatter --
no grids, no cocircular/collinear configurations) so the Delaunay triangulation is unique and scipy (Qhull) and the
inlined Bowyer-Watson pick the same simplex SET; the Clough-Tocher interpolant is invariant to the per-triangle
vertex relabelling and simplex/neighbour ordering, so inside-hull values match to ~1e-12 regardless of Qhull's
internal order. Each case mixes query points inside the hull with points outside it (-> NaN). Host-independent
(value-only parity; no triangulation-index parity)."""
import json
import os

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'interpolate_cloughtocher.json')
rng = np.random.default_rng(20261012)
cases = []


def encode(x):
    """NaN -> the string "nan" (bare NaN is invalid JSON), else a plain float."""
    a = np.asarray(x, float)
    if a.ndim == 0:
        return 'nan' if np.isnan(a) else float(a)
    return [encode(e) for e in a]


def case(points, values, query):
    P = np.asarray(points, float)
    V = np.asarray(values, float)
    Q = np.asarray(query, float)
    res = np.asarray(CloughTocher2DInterpolator(P, V)(Q))
    cases.append({'points': P.tolist(), 'values': V.tolist(), 'query': Q.tolist(),
                  'expect': encode(res)})


def mix_queries(P, n_inside, outside):
    lo = P.min(0)
    hi = P.max(0)
    pad = 0.15 * (hi - lo)
    inside = rng.uniform(lo + pad, hi - pad, (n_inside, 2))
    return np.vstack([inside, np.asarray(outside, float)])


# 1) generic random scatter; queries inside + clearly outside the hull
P1 = rng.uniform(0.0, 10.0, (40, 2))
V1 = rng.uniform(-5.0, 5.0, 40)
Q1 = mix_queries(P1, 60, [[-5.0, -5.0], [15.0, 15.0], [-3.0, 5.0], [5.0, 20.0]])
case(P1, V1, Q1)

# 2) different seed/scale; interior + outside
P2 = rng.uniform(-3.0, 7.0, (55, 2))
V2 = rng.uniform(-10.0, 10.0, 55)
Q2 = mix_queries(P2, 80, [[-10.0, 0.0], [12.0, 3.0], [0.0, -10.0], [2.0, 12.0]])
case(P2, V2, Q2)

# 3) smooth function sampled at scatter (tests gradient accuracy across the patch)
P3 = rng.uniform(0.0, 2.0 * np.pi, (60, 2))
V3 = np.sin(P3[:, 0]) * np.cos(P3[:, 1])
Q3 = mix_queries(P3, 90, [[-1.0, 1.0], [7.0, 7.0]])
case(P3, V3, Q3)

# 4) larger scatter, quadratic-ish surface
P4 = rng.uniform(-5.0, 5.0, (80, 2))
V4 = P4[:, 0] ** 2 - P4[:, 1] ** 2 + 0.5 * P4[:, 0] * P4[:, 1]
Q4 = mix_queries(P4, 100, [[-8.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
case(P4, V4, Q4)

# 5) wide value range
P5 = rng.uniform(0.0, 100.0, (50, 2))
V5 = rng.uniform(0.0, 1000.0, 50)
Q5 = mix_queries(P5, 70, [[-10.0, -10.0], [200.0, 200.0]])
case(P5, V5, Q5)

# 6) modest point count
P6 = rng.uniform(2.0, 8.0, (30, 2))
V6 = rng.uniform(-1.0, 1.0, 30)
Q6 = mix_queries(P6, 50, [[0.0, 0.0], [10.0, 10.0], [5.0, 0.0]])
case(P6, V6, Q6)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as fh:
    json.dump({'cloughtocher_cases': cases}, fh)
print(f'wrote {len(cases)} CloughTocher2DInterpolator cases -> {OUT}')
