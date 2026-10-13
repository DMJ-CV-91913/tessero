#!/usr/bin/env python3
"""scipy.spatial.Voronoi (2-D) fixtures for the pure-PHP Tessero\\Spatial\\Voronoi (dual of the
Delaunay triangulation). For points in general position the diagram is unique, so the SET of finite
Voronoi vertices (= Delaunay circumcenters, rounded) and the SET of ridge-point pairs (= Delaunay
edges, as sorted index pairs) are stable invariants Tessero reproduces exactly. The run-tests branch
compares both as ORDER-FREE sets; Qhull's per-vertex/-ridge ordering and infinite-ridge directions
are internal and NOT asserted. Host-independent (no pinned env)."""
import json
import os

import numpy as np
from scipy.spatial import Voronoi

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

out = {'module': 'spatial', 'scipy': __import__('scipy').__version__, 'voronoi_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial_voronoi: {len(cases)} cases')
