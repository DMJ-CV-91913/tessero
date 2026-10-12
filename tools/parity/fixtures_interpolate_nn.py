#!/usr/bin/env python3
"""scipy.interpolate.NearestNDInterpolator fixtures (pure-PHP Tessero\\Interpolate\\NearestNDInterpolator over
Tessero\\Spatial\\KDTree): construct + query, an exact gather of the nearest data point's value. Continuous
random data (exact ties are measure-zero) so the lowest-index tie rule never diverges from scipy; one explicit
non-rescaled tie case locks the tie behaviour. rescale uses continuous data only (exact-tie-heavy integer grids
under rescale can break ties differently in scipy's cKDTree -- out of scope, documented)."""
import json, os
import numpy as np
from scipy.interpolate import NearestNDInterpolator

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'interpolate_nn.json')
rng = np.random.default_rng(20261012)
cases = []


def case(points, values, query, rescale=False):
    P = np.asarray(points, float); V = np.asarray(values); Q = np.asarray(query, float)
    res = np.asarray(NearestNDInterpolator(P, V, rescale=rescale)(Q))
    cases.append({'points': P.tolist(), 'values': V.tolist(), 'query': Q.tolist(),
                  'rescale': bool(rescale), 'expect': res.tolist()})


# 2-D scalar, queried on a grid
P2 = rng.uniform(-5, 5, (40, 2)); V2 = rng.uniform(-10, 10, 40)
gx, gy = np.meshgrid(np.linspace(-5, 5, 11), np.linspace(-5, 5, 11))
Q2 = np.column_stack([gx.ravel(), gy.ravel()])
case(P2, V2, Q2)
# 3-D scalar
P3 = rng.uniform(-3, 3, (60, 3)); V3 = rng.uniform(-10, 10, 60); Q3 = rng.uniform(-3, 3, (50, 3))
case(P3, V3, Q3)
# 2-D vector-valued y (npts, 3)
Pm = rng.uniform(-5, 5, (30, 2)); Vm = rng.uniform(-10, 10, (30, 3))
case(Pm, Vm, Q2)
# rescale=True (continuous data)
case(P2, V2, Q2, rescale=True)
case(P3, V3, Q3, rescale=True)
# explicit tie at the origin (indices 2,5,6 equidistant; lowest index 2 must win)
Pt = [[10., 10.], [9., 9.], [1., 0.], [8., 8.], [7., 7.], [-1., 0.], [0., 1.]]
Vt = [i * 100.0 for i in range(7)]
case(Pt, Vt, [[0., 0.]])

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'nearestnd_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_nn: {len(cases)} NearestNDInterpolator cases')
