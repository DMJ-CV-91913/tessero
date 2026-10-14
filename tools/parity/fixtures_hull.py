#!/usr/bin/env python3
"""scipy.spatial.ConvexHull (2-D) fixtures: the pure-PHP Tessero\\Spatial\\ConvexHull verified against
scipy/Qhull. Compared per case: the hull vertex SET (scipy's start index is Qhull-arbitrary, so we store
the SORTED vertex list), the enclosed polygon area (.volume), and the perimeter (.area). Deterministic
GENERAL-POSITION point sets only -- no collinear points and no axis grids, so Qhull never raises and the
hull is unambiguous. Host-independent (no pinned environment required)."""
import json
import os

import numpy as np
from scipy.spatial import ConvexHull

OUT = os.environ.get('HULL_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'spatial_hull.json')


def general_position(pts):
    """True if no three of the points are collinear (reject degenerate sets up front)."""
    n = len(pts)
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                area2 = ((pts[b][0] - pts[a][0]) * (pts[c][1] - pts[a][1])
                         - (pts[b][1] - pts[a][1]) * (pts[c][0] - pts[a][0]))
                if abs(area2) < 1e-9:
                    return False
    return True


rng = np.random.default_rng(20261012)
cases = []
# randomized general-position clouds (reject any that happen to produce a collinear triple)
while len(cases) < 8:
    npts = int(rng.integers(6, 24))
    pts = np.round(rng.uniform(-5.0, 5.0, (npts, 2)), 10)
    ptl = [tuple(p) for p in pts.tolist()]
    if len(set(ptl)) != npts or not general_position(ptl):
        continue
    h = ConvexHull(pts)
    cases.append({
        'points': pts.tolist(),
        'expect': sorted(int(v) for v in h.vertices.tolist()),
        'area': float(h.area),      # perimeter in 2-D
        'volume': float(h.volume),  # polygon area in 2-D
    })

# a few explicit, hand-chosen general-position sets (triangle, convex quad, square + interior point,
# a pentagon-ish cloud) -- all with a mix of interior and boundary points, none collinear.
explicit = [
    [[0.0, 0.0], [4.0, 1.0], [1.0, 3.0]],
    [[0.0, 0.0], [5.0, 1.0], [4.0, 4.0], [1.0, 5.0]],
    [[-2.0, -2.0], [3.0, -1.0], [2.0, 3.0], [-3.0, 2.0], [0.2, 0.1]],
    [[0.0, 0.0], [6.0, 0.5], [5.0, 5.0], [0.5, 6.0], [-4.0, 2.0], [1.5, 1.2], [2.3, 3.1]],
]
for pts in explicit:
    arr = np.asarray(pts, dtype=float)
    assert general_position([tuple(p) for p in pts]), 'explicit set must be general position'
    h = ConvexHull(arr)
    cases.append({
        'points': pts,
        'expect': sorted(int(v) for v in h.vertices.tolist()),
        'area': float(h.area),
        'volume': float(h.volume),
    })

# --- N-D (d>=3) general-position clouds: the N-D quickhull facet triangulation is non-unique, so
# we compare ONLY the well-defined invariants -- the vertex SET (sorted), the solid measure (.volume)
# and the boundary measure (.area). Deterministic random clouds in 3-D/4-D/5-D plus explicit shapes. ---
rng_nd = np.random.default_rng(20261013)


def _add_nd(pts):
    arr = np.asarray(pts, dtype=float)
    h = ConvexHull(arr)
    cases.append({
        'points': arr.tolist(),
        'expect': sorted(int(v) for v in h.vertices.tolist()),
        'area': float(h.area),      # (d-1)-measure: surface area in 3-D, ...
        'volume': float(h.volume),  # d-measure: volume in 3-D, ...
    })


for _d in (3, 4, 5):
    _made = 0
    while _made < (4 if _d == 3 else 3):
        _np = int(rng_nd.integers(_d + 3, 24))
        _pts = np.round(rng_nd.uniform(-5.0, 5.0, (_np, _d)), 10)
        if len({tuple(p) for p in _pts.tolist()}) != _np:
            continue
        try:
            _add_nd(_pts)
        except Exception:
            continue
        _made += 1

# explicit N-D shapes with known invariants
_add_nd([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])  # unit cube
_add_nd([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, 0.25]])                           # tetra + interior
_add_nd(np.vstack([np.eye(4), np.zeros(4), np.full(4, 0.1)]).tolist())                              # 4-simplex + interior
# points on a sphere/hypersphere -> every point is a hull vertex
_sv = rng_nd.standard_normal((28, 3)); _sv /= np.linalg.norm(_sv, axis=1, keepdims=True)
_add_nd(np.round(_sv, 10).tolist())
_sv4 = rng_nd.standard_normal((34, 4)); _sv4 /= np.linalg.norm(_sv4, axis=1, keepdims=True)
_add_nd(np.round(_sv4, 10).tolist())

out = {'module': 'spatial', 'scipy': __import__('scipy').__version__, 'hull_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial_hull: {len(cases)} cases')
