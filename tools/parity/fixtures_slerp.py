#!/usr/bin/env python3
"""scipy.spatial.transform.Slerp fixtures for pure-PHP Tessero\\Spatial\\Transform\\Slerp.

Spherical linear interpolation of a key sequence of Rotations. The rotation MATRIX is the
sign-unambiguous invariant, so every interpolated rotation is validated through its matrix. Each
case records the key times + key quaternions, the query times, and the expected interpolated
matrices (including the inclusive endpoints, which must reproduce the keys exactly).

Host-independent: closed-form trig, no pinned environment / BLAS needed."""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

OUT = os.environ.get('SLERP_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'spatial_slerp.json')

rng = np.random.default_rng(20261013)
cases = []


def randrot(n):
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return R.from_quat(q)


# named key sequences of increasing size; random + structured.
specs = []
for n in (2, 3, 4, 5, 7):
    times = np.sort(rng.uniform(0.0, 12.0, n))
    times = np.unique(times)
    specs.append((times, randrot(len(times))))

# small-angle neighbours (nearly-parallel branch) + anti-parallel pair
times = np.array([0.0, 1.0, 2.0])
base = randrot(1).as_quat()[0]
q2 = base + 1e-9 * rng.standard_normal(4)
q3 = -base + 1e-9 * rng.standard_normal(4)
key = R.from_quat(np.array([base, q2 / np.linalg.norm(q2), q3 / np.linalg.norm(q3)]))
specs.append((times, key))

# Euler-defined keyframes (readable, structured angles)
times = np.array([0.0, 10.0, 20.0, 40.0])
ang = [[-10, 20, 30], [0, 15, 40], [-30, 45, 30], [20, 45, 90]]
specs.append((times, R.from_euler('XYZ', ang, degrees=True)))

for (times, key) in specs:
    sl = Slerp(times, key)
    # dense query across the inclusive range, plus the exact keyframe times.
    tq = np.unique(np.concatenate([np.linspace(times[0], times[-1], 11), times]))
    expect = sl(tq).as_matrix()
    cases.append({
        'times': times.tolist(),
        'quat': key.as_quat().tolist(),
        'query': tq.tolist(),
        'matrix': np.atleast_3d(expect).reshape(-1, 3, 3).tolist(),
    })

out = {'module': 'spatial.transform', 'scipy': __import__('scipy').__version__,
       'slerp_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial_slerp: {len(cases)} cases')
