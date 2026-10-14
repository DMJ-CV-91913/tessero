#!/usr/bin/env python3
r"""scipy.spatial.SphericalVoronoi + scipy.spatial.HalfspaceIntersection fixtures for the pure-PHP
Tessero\Spatial\SphericalVoronoi and Tessero\Spatial\HalfspaceIntersection, both computed via the
convex-hull DUAL over Tessero\Spatial\ConvexHull (no Qhull FFI).

Compared per case (the geometric invariants -- Qhull's per-vertex ORDER is internal, so vertex/point
collections are stored and compared as order-free SETS):
  SphericalVoronoi     ->vertices  = hull-facet circumcenters projected to the sphere (SET, ~1e-9).
  HalfspaceIntersection ->intersections = polytope vertices = duals of the dual-hull facets (SET),
                        ->dual_vertices  = halfspace indices that are facets of the polytope (SET).

Deterministic GENERAL-POSITION inputs only (no co-spherical facets / no co-planar dual facets), so
Qhull never raises and the hull dual is unambiguous. Host-independent (no pinned environment required,
but produced under the pinned parity env like every other fixture)."""
import json
import os

import numpy as np
from scipy.spatial import SphericalVoronoi, HalfspaceIntersection

OUT = os.environ.get('SPHVOR_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'spatial_sphvor.json')

# ---------- SphericalVoronoi: generators on S^2 (and a few with non-unit radius/center) ----------
sv_cases = []
for n in (6, 8, 10, 14, 20):
    rng = np.random.default_rng(90000 + n)
    p = rng.standard_normal((n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    p = np.round(p, 12)
    # renormalize after rounding so the "on the sphere" check passes at 1e-6
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    sv = SphericalVoronoi(p, radius=1.0, center=np.zeros(3))
    sv_cases.append({
        'points': p.tolist(),
        'radius': 1.0,
        'center': [0.0, 0.0, 0.0],
        'vertices': sv.vertices.tolist(),
    })

# explicit octahedron, unit sphere
oct_pts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1.0]])
sv = SphericalVoronoi(oct_pts, radius=1.0, center=np.zeros(3))
sv_cases.append({'points': oct_pts.tolist(), 'radius': 1.0, 'center': [0.0, 0.0, 0.0],
                 'vertices': sv.vertices.tolist()})

# non-unit radius and off-origin center
rng = np.random.default_rng(91234)
base = rng.standard_normal((12, 3))
base /= np.linalg.norm(base, axis=1, keepdims=True)
rad = 2.5
cen = np.array([1.0, -2.0, 0.5])
pp = np.round(cen + rad * base, 12)
# re-project to the exact sphere after rounding
d = pp - cen
pp = cen + rad * d / np.linalg.norm(d, axis=1, keepdims=True)
sv = SphericalVoronoi(pp, radius=rad, center=cen)
sv_cases.append({'points': pp.tolist(), 'radius': rad, 'center': cen.tolist(),
                 'vertices': sv.vertices.tolist()})

# a 2-D circle case (1-sphere): generators on the unit circle, vertices are the midpoint directions
ang = np.array([0.3, 1.1, 2.0, 2.9, 3.7, 4.6, 5.4])
circ = np.round(np.stack([np.cos(ang), np.sin(ang)], axis=1), 12)
circ /= np.linalg.norm(circ, axis=1, keepdims=True)
sv = SphericalVoronoi(circ, radius=1.0, center=np.zeros(2))
sv_cases.append({'points': circ.tolist(), 'radius': 1.0, 'center': [0.0, 0.0],
                 'vertices': sv.vertices.tolist()})

# a 4-D hypersphere case (general d supported by the hull dual)
rng = np.random.default_rng(94444)
p4 = rng.standard_normal((12, 4))
p4 /= np.linalg.norm(p4, axis=1, keepdims=True)
p4 = np.round(p4, 12)
p4 /= np.linalg.norm(p4, axis=1, keepdims=True)
sv = SphericalVoronoi(p4, radius=1.0, center=np.zeros(4))
sv_cases.append({'points': p4.tolist(), 'radius': 1.0, 'center': [0.0, 0.0, 0.0, 0.0],
                 'vertices': sv.vertices.tolist()})


# ---------- HalfspaceIntersection ----------
def box(dims, lo, hi):
    hs = []
    for k in range(dims):
        r = [0.0] * dims; r[k] = -1.0; r.append(lo); hs.append(r)
        r = [0.0] * dims; r[k] = 1.0; r.append(-hi); hs.append(r)
    return hs


hs_specs = [
    (box(3, 0.0, 1.0), [0.5, 0.5, 0.5]),
    (box(2, -1.0, 1.0), [0.0, 0.0]),
    (box(3, -2.0, 3.0), [0.0, 0.0, 0.0]),
    ([[-1.0, 0, 0, 0.0], [0, -1.0, 0, 0.0], [0, 0, -1.0, 0.0], [1.0, 1, 1, -1.0]], [0.2, 0.2, 0.2]),
    ([[-1.0, 0, 0.0], [0, -1.0, 0.0], [1.0, 1, -2.0]], [0.3, 0.3]),
    (box(3, 0.0, 2.0) + [[1.0, 1, 1, -5.0]], [0.5, 0.5, 0.5]),
    (box(4, 0.0, 1.0), [0.5, 0.5, 0.5, 0.5]),
]
hs_cases = []
for hs, x0 in hs_specs:
    hs = np.asarray(hs, float)
    x0 = np.asarray(x0, float)
    hi = HalfspaceIntersection(hs, x0)
    hs_cases.append({
        'halfspaces': hs.tolist(),
        'interior_point': x0.tolist(),
        'intersections': hi.intersections.tolist(),
        'dual_vertices': sorted(int(v) for v in hi.dual_vertices),
    })

out = {'module': 'spatial', 'scipy': __import__('scipy').__version__,
       'sphvor_cases': sv_cases, 'halfspace_cases': hs_cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print('spatial_sphvor: %d SphericalVoronoi + %d HalfspaceIntersection cases' % (len(sv_cases), len(hs_cases)))
