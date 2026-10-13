#!/usr/bin/env python3
"""scipy.interpolate.LinearNDInterpolator fixtures (pure-PHP Tessero\\Interpolate\\LinearNDInterpolator over an
inlined Bowyer-Watson 2-D Delaunay + barycentric-linear interpolation): construct + query. 2-D, value-only,
GENERAL POSITION points only (continuous random scatter -- no grids, no cocircular/collinear configurations) so
the Delaunay triangulation is unique and scipy (Qhull) and the inlined Bowyer-Watson pick the same simplices;
inside-hull values then match to ~1e-12. Each case mixes query points inside the hull with points outside it to
lock the NaN-outside-hull behaviour exactly. Host-independent (value-only parity; no triangulation-index parity)."""
import json
import os

import numpy as np
from scipy.interpolate import LinearNDInterpolator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'interpolate_linearnd.json')
rng = np.random.default_rng(20261012)
cases = []


def encode(x):
    """Encode a (possibly nested) array of floats to JSON-safe values: NaN -> the string "nan"
    (bare NaN is invalid JSON and PHP's json_decode rejects it), everything else a plain float."""
    a = np.asarray(x, float)
    if a.ndim == 0:
        return 'nan' if np.isnan(a) else float(a)
    return [encode(e) for e in a]


def case(points, values, query, rescale=False):
    P = np.asarray(points, float)
    V = np.asarray(values, float)
    Q = np.asarray(query, float)
    res = np.asarray(LinearNDInterpolator(P, V, rescale=rescale)(Q))
    cases.append({'points': P.tolist(), 'values': V.tolist(), 'query': Q.tolist(),
                  'rescale': bool(rescale), 'expect': encode(res)})


def mix_queries(P, n_inside, outside):
    """Query points: n_inside strictly inside the hull's interior box + explicit outside-hull points (-> NaN)."""
    lo = P.min(0)
    hi = P.max(0)
    pad = 0.15 * (hi - lo)
    inside = rng.uniform(lo + pad, hi - pad, (n_inside, 2))
    return np.vstack([inside, np.asarray(outside, float)])


# 1) 2-D scalar, generic random scatter; queries inside + clearly outside the hull
P1 = rng.uniform(0.0, 10.0, (40, 2))
V1 = rng.uniform(-5.0, 5.0, 40)
Q1 = mix_queries(P1, 60, [[-5.0, -5.0], [15.0, 15.0], [-3.0, 5.0], [5.0, 20.0]])
case(P1, V1, Q1)

# 2) 2-D scalar, different seed/scale; interior + outside
P2 = rng.uniform(-3.0, 7.0, (55, 2))
V2 = rng.uniform(-10.0, 10.0, 55)
Q2 = mix_queries(P2, 80, [[-10.0, 0.0], [12.0, 3.0], [0.0, -10.0], [2.0, 12.0], [100.0, 100.0]])
case(P2, V2, Q2)

# 3) 2-D vector-valued values (npts, 3); interior + outside (NaN must appear on every component)
P3 = rng.uniform(0.0, 1.0, (45, 2))
V3 = rng.uniform(-2.0, 2.0, (45, 3))
Q3 = mix_queries(P3, 50, [[-1.0, -1.0], [2.0, 2.0], [-0.5, 0.5]])
case(P3, V3, Q3)

# 4) rescale=True (anisotropic axes, continuous data); interior + outside
P4 = np.column_stack([rng.uniform(0.0, 1000.0, 50), rng.uniform(0.0, 0.01, 50)])
V4 = rng.uniform(-5.0, 5.0, 50)
Q4 = mix_queries(P4, 60, [[-100.0, 0.005], [1200.0, 0.005], [500.0, -0.01], [500.0, 0.05]])
case(P4, V4, Q4, rescale=True)

# 5) a second generic scalar scatter, larger, for extra coverage
P5 = rng.uniform(-5.0, 5.0, (70, 2))
V5 = rng.uniform(-100.0, 100.0, 70)
Q5 = mix_queries(P5, 100, [[-9.0, 0.0], [9.0, 0.0], [0.0, -9.0], [0.0, 9.0]])
case(P5, V5, Q5)

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'linearnd_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'), allow_nan=False)   # NaN encoded as "nan"; bare NaN -> error
print(f'interpolate_linearnd: {len(cases)} LinearNDInterpolator cases')
