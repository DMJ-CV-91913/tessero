#!/usr/bin/env python3
"""scipy.spatial.transform.RotationSpline fixtures for pure-PHP
Tessero\\Spatial\\Transform\\RotationSpline.

C2 rotation interpolation: per-segment cubic rotation vectors with continuous angular rate and
acceleration. scipy solves a block-tridiagonal nonlinear system (Newton, MAX_ITER=10, TOL=1e-9)
for the interior angular rates; the pure-PHP port reproduces that solve and the resulting PPoly to
~1e-14. Each case records the key times + key quaternions, query times, the expected interpolated
rotation matrices (order 0), and the expected angular rate (order 1) and acceleration (order 2).

Host-independent: deterministic solve, no RNG in the algorithm; no pinned environment / BLAS."""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R, RotationSpline as RS

OUT = os.environ.get('ROTSPLINE_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'spatial_rotspline.json')

rng = np.random.default_rng(20261013)
cases = []


def randrot(n):
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return R.from_quat(q)


specs = []
# 2-key (special-cased: rotvec_dot = angular_rate, no solve)
specs.append((np.array([0.0, 3.0]), randrot(2)))
# small N triggering the banded solve
for n in (3, 4, 5, 6, 8):
    times = np.sort(rng.uniform(0.0, 20.0, n))
    times = np.unique(times)
    specs.append((times, randrot(len(times))))

# the docstring example (Euler-defined, uneven spacing)
times = np.array([0.0, 10.0, 20.0, 40.0], dtype=float)
ang = [[-10, 20, 30], [0, 15, 40], [-30, 45, 30], [20, 45, 90]]
specs.append((times, R.from_euler('XYZ', ang, degrees=True)))

for (times, key) in specs:
    sp = RS(times, key)
    tq = np.unique(np.concatenate([np.linspace(times[0], times[-1], 13), times]))
    mat = sp(tq).as_matrix()
    rate = np.atleast_2d(sp(tq, 1))
    accel = np.atleast_2d(sp(tq, 2))
    cases.append({
        'times': times.tolist(),
        'quat': key.as_quat().tolist(),
        'query': tq.tolist(),
        'matrix': np.atleast_3d(mat).reshape(-1, 3, 3).tolist(),
        'rate': rate.tolist(),
        'accel': accel.tolist(),
    })

out = {'module': 'spatial.transform', 'scipy': __import__('scipy').__version__,
       'rotspline_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial_rotspline: {len(cases)} cases')
