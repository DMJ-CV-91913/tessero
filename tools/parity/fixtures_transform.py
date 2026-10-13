#!/usr/bin/env python3
"""scipy.spatial.transform.Rotation fixtures for the pure-PHP Tessero\\Spatial\\Transform\\Rotation.

The rotation MATRIX is the sign-unambiguous invariant, so every representation is validated through
it: from each source representation we record the expected matrix and the expected values of the
other representations. The run-tests branch:
  * compares matrices / rotvec / mrp / euler / magnitude / apply VALUES to machine precision;
  * compares quaternions UP TO the global q/-q sign (scipy keeps the stored input sign);
  * for Euler, also checks that from_euler(as_euler(.)) reproduces the matrix (covers gimbal lock).

Host-independent: pure trig, no pinned environment / BLAS needed."""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

OUT = os.environ.get('TRANSFORM_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'spatial_transform.json')

rng = np.random.default_rng(20261013)
cases = []

# --- from_quat: record every accessor -------------------------------------
for n in (1, 2, 5, 8):
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    r = R.from_quat(q)
    v = rng.standard_normal((n, 3))
    cases.append({
        'src': 'quat',
        'quat': q.tolist(),
        'matrix': r.as_matrix().tolist(),
        'rotvec': r.as_rotvec().tolist(),
        'rotvec_deg': r.as_rotvec(degrees=True).tolist(),
        'mrp': r.as_mrp().tolist(),
        'euler_xyz': r.as_euler('xyz').tolist(),
        'euler_ZYX': r.as_euler('ZYX').tolist(),
        'euler_zxz': r.as_euler('zxz').tolist(),
        'euler_XYZ': r.as_euler('XYZ').tolist(),
        'magnitude': np.atleast_1d(r.magnitude()).tolist(),
        'apply_vec': v.tolist(),
        'apply': np.atleast_2d(r.apply(v)).tolist(),
        'apply_inv': np.atleast_2d(r.apply(v, inverse=True)).tolist(),
    })

# --- from_euler every order (incl. degrees) -> matrix + euler roundtrip ----
import warnings
for order in ['xyz', 'zyx', 'XYZ', 'ZYX', 'zxz', 'ZXZ', 'xyx', 'ZXY']:
    ang = rng.uniform(-2.5, 2.5, (6, 3))
    r = R.from_euler(order, ang)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        back = r.as_euler(order).tolist()
    cases.append({'src': 'euler', 'order': order, 'angles': ang.tolist(),
                  'matrix': r.as_matrix().tolist(), 'euler_back': back})

# degrees
ang = rng.uniform(-170, 170, (5, 3))
r = R.from_euler('xyz', ang, degrees=True)
cases.append({'src': 'euler_deg', 'order': 'xyz', 'angles': ang.tolist(),
              'matrix': r.as_matrix().tolist()})

# single-axis / single rotation
r = R.from_euler('x', 0.7)
cases.append({'src': 'euler_single', 'order': 'x', 'angles': 0.7, 'matrix': r.as_matrix().tolist()})

# --- from_rotvec / from_mrp -> matrix -------------------------------------
rv = rng.standard_normal((6, 3))
cases.append({'src': 'rotvec', 'rotvec': rv.tolist(), 'matrix': R.from_rotvec(rv).as_matrix().tolist()})
rv_small = rng.standard_normal((4, 3)) * 1e-8
cases.append({'src': 'rotvec', 'rotvec': rv_small.tolist(),
              'matrix': R.from_rotvec(rv_small).as_matrix().tolist()})
mrp = rng.standard_normal((6, 3)) * 0.4
cases.append({'src': 'mrp', 'mrp': mrp.tolist(), 'matrix': R.from_mrp(mrp).as_matrix().tolist()})

# --- gimbal-lock Euler (second angle ~ +-pi/2) ----------------------------
gl = [[0.3, np.pi / 2, 0.7], [0.3, -np.pi / 2, 0.7], [1.1, np.pi / 2, -0.4]]
r = R.from_euler('xyz', gl)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    back = r.as_euler('xyz').tolist()
cases.append({'src': 'euler', 'order': 'xyz', 'angles': gl, 'matrix': r.as_matrix().tolist(),
              'euler_back': back})

# symmetric gimbal (second angle ~ 0 or pi for zxz)
gls = [[0.5, 0.0, 0.9], [0.5, np.pi, 0.9]]
r = R.from_euler('zxz', gls)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    back = r.as_euler('zxz').tolist()
cases.append({'src': 'euler', 'order': 'zxz', 'angles': gls, 'matrix': r.as_matrix().tolist(),
              'euler_back': back})

# --- identity -------------------------------------------------------------
cases.append({'src': 'identity', 'matrix': R.identity().as_matrix().tolist()})

out = {'module': 'spatial.transform', 'scipy': __import__('scipy').__version__,
       'transform_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial_transform: {len(cases)} cases')
