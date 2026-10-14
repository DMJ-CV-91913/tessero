#!/usr/bin/env python3
"""scipy.spatial.Voronoi fixtures for the pure-PHP Tessero\\Spatial\\Voronoi (dual of the Delaunay
triangulation). For points in general position the diagram is unique, so the SET of finite Voronoi
vertices (= Delaunay circumcenters, rounded) is a stable invariant Tessero reproduces exactly in
2-D AND N-D. In 2-D the SET of ridge-point pairs (= Delaunay edges, as sorted index pairs) is also a
stable invariant; in d >= 3 the ridge_points set depends on Qhull's full unbounded-Voronoi
enumeration (upper-hull / point-at-infinity) and is NOT reproducible from the lower-hull Delaunay,
so N-D cases OMIT the 'ridge_points' key and the harness skips that check. The run-tests branch
compares vertices as an ORDER-FREE set (dimension-generic) and ridge_points as a set WHEN PRESENT.
Qhull's per-vertex/-ridge ordering and infinite-ridge directions are internal and NOT asserted.
Host-independent (no pinned env)."""
import json
import os

import numpy as np
from scipy.spatial import Delaunay, Voronoi

OUT = os.environ.get('VORONOI_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'spatial_voronoi.json')


def gen_points(rng, n):
    """General-position 2-D points: reject near-coincident pairs and near-collinear triples so the
    Delaunay dual (hence the Voronoi diagram) is unambiguous."""
    while True:
        pts = np.round(rng.standard_normal((n, 2)), 10)
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                if np.hypot(*(pts[i] - pts[j])) < 1e-3:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        a, b, c = pts[i], pts[j], pts[k]
                        if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
        if ok:
            return pts


def make_case(pts):
    vor = Voronoi(np.asarray(pts, float))
    rp = np.sort(vor.ridge_points, axis=1)
    return {
        'points': np.asarray(pts, float).tolist(),
        'vertices': np.round(vor.vertices, 6).tolist(),
        'ridge_points': rp.tolist(),
        'npoints': int(len(pts)),
    }


rng = np.random.default_rng(20261013)
cases = []
for n in (4, 6, 8, 10, 12, 16, 20, 25):
    cases.append(make_case(gen_points(rng, int(n))))

# a hand-checked general-position case
explicit = [[0.1, 0.2], [0.9, 0.15], [0.5, 0.85], [0.45, 0.4], [0.7, 0.6], [0.2, 0.7]]
cases.append(make_case(explicit))

# --- N-D (d >= 3) vertex-only cases: the finite Voronoi vertices are the circumcenters of the
# Delaunay simplices (lifting-map lower hull), a stable SET invariant Tessero matches exactly.
# ridge_points is OMITTED (see module docstring) so the harness skips it. Deterministic 3-D/4-D
# Gaussian clouds rejected unless scipy reports no coplanar Delaunay points. ---
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


def make_case_nd(pts):
    vor = Voronoi(np.asarray(pts, float))
    return {
        'points': np.asarray(pts, float).tolist(),
        'vertices': np.round(vor.vertices, 6).tolist(),
        'npoints': int(len(pts)),
        # NOTE: no 'ridge_points' key -> harness skips the ridge check for N-D.
    }


for _d in (3, 4):
    for _n in (_d + 2, _d + 4, _d + 7, _d + 11):
        cases.append(make_case_nd(gen_points_nd(rng_nd, int(_n), int(_d))))

out = {'module': 'spatial', 'scipy': __import__('scipy').__version__, 'voronoi_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial_voronoi: {len(cases)} cases')
