#!/usr/bin/env python3
"""scipy.spatial.Delaunay fixtures for the pure-PHP Tessero\\Spatial\\Delaunay. The Delaunay
triangulation is unique for points in general position (no three collinear, no four cocircular in
2-D; no d+1 coplanar, no d+2 cospherical in N-D), so the set of simplices -- each a sorted tuple of
input-point indices -- is a stable invariant that Tessero reproduces exactly (2-D via Bowyer-Watson,
N-D via the lifting-map lower convex hull). The run-tests branch compares simplices as a SET of
sorted-index tuples and compares find_simplex per query by the CONTAINING simplex (sorted indices),
since per-impl simplex ordering differs. transform is exercised inside find_simplex (barycentric
sign scan).

Deterministic general-position inputs ONLY: continuous Gaussian/uniform points rounded to 10 d.p.
(no grids, no squares, no lattices -- those are cocircular/collinear-heavy and ambiguous for Qhull's
symbolic perturbation). Host-independent -- no pinned environment required."""
import json
import os

import numpy as np
from scipy.spatial import Delaunay

OUT = os.environ.get('DELAUNAY_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'spatial_delaunay.json')


def gen_points(rng, n):
    """Draw n general-position 2-D points; reject if any triangle is near-degenerate (collinear)
    or any pair near-coincident, so the triangulation is unambiguous."""
    while True:
        pts = np.round(rng.standard_normal((n, 2)), 10)
        ok = True
        # no near-coincident points
        for i in range(n):
            for j in range(i + 1, n):
                if np.hypot(*(pts[i] - pts[j])) < 1e-3:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        # no near-collinear triple (tiny signed area)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = pts[i], pts[j], pts[k]
                    area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                    if abs(area2) < 1e-6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return pts


def make_case(pts, query):
    pts = np.asarray(pts, float)
    d = Delaunay(pts)
    simplices = [list(map(int, t)) for t in d.simplices]
    q = np.asarray(query, float)
    fs = [int(x) for x in d.find_simplex(q)]
    return {
        'points': pts.tolist(),
        'simplices': simplices,
        'query': q.tolist(),
        'find_simplex': fs,
    }


rng = np.random.default_rng(20261012)
cases = []

# randomized general-position cases of varying size
for n in (5, 7, 9, 12, 16, 20, 25):
    pts = gen_points(rng, int(n))
    # queries: a spread of interior + exterior points (exterior -> -1)
    qs = []
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    for _ in range(12):
        qs.append(np.round(lo + rng.random(2) * (hi - lo), 10))
    # a few deliberate exterior points
    qs += [[hi[0] + 5.0, hi[1] + 5.0], [lo[0] - 5.0, lo[1]], [0.0, lo[1] - 5.0]]
    cases.append(make_case(pts, qs))

# an explicit small hand-checked case (general position)
explicit = np.array([[0.1, 0.2], [0.9, 0.15], [0.5, 0.85], [0.45, 0.4], [0.7, 0.6]])
cases.append(make_case(explicit, [[0.5, 0.4], [0.5, 0.45], [2.0, 2.0], [0.3, 0.3]]))

# --- N-D (d >= 3) general-position cases: the Delaunay triangulation is unique for points in
# general position, so the SIMPLEX SET -- each a sorted (d+1)-tuple of input indices -- is a stable
# invariant that Tessero reproduces exactly via the lifting-map lower convex hull. find_simplex is
# compared by the CONTAINING simplex (sorted tuple). Deterministic Gaussian clouds in 3-D/4-D,
# rejected unless scipy reports no coplanar points. The run-tests delaunay_cases branch is
# dimension-generic (sorts tuples, indexes simplices) and needs NO change. Host-independent. ---
rng_nd = np.random.default_rng(20261014)


def gen_points_nd(rng_, n, d):
    while True:
        pts = np.round(rng_.standard_normal((n, d)), 10)
        try:
            dt = Delaunay(pts)
        except Exception:
            continue
        if dt.coplanar.size == 0:
            ok = True
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(pts[i] - pts[j]) < 1e-2:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return pts


for _d in (3, 4):
    for _n in (_d + 2, _d + 4, _d + 7, _d + 11):
        _pts = gen_points_nd(rng_nd, int(_n), int(_d))
        _lo = _pts.min(axis=0)
        _hi = _pts.max(axis=0)
        _qs = []
        for _ in range(12):
            _qs.append(np.round(_lo + rng_nd.random(_d) * (_hi - _lo), 10).tolist())
        _qs.append((_hi + 9.0).tolist())   # exterior -> -1
        _qs.append((_lo - 9.0).tolist())   # exterior -> -1
        cases.append(make_case(_pts, _qs))

out = {'module': 'spatial', 'scipy': __import__('scipy').__version__, 'delaunay_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial_delaunay: {len(cases)} cases')
