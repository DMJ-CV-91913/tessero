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

out = {'module': 'spatial', 'scipy': __import__('scipy').__version__, 'hull_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial_hull: {len(cases)} cases')
