#!/usr/bin/env python3
"""scipy.spatial.transform.RigidTransform fixtures for pure-PHP
Tessero\\Spatial\\Transform\\RigidTransform.

A proper rigid transform in SE(3) stored as a 4x4 homogeneous matrix. Each case records the
construction inputs (translation + key rotation quaternions) and the expected 4x4 matrix,
transformed points (apply, forward + inverse), inverse matrix, and the composed transform's matrix
and applied points. Everything is a closed-form linear-algebra identity; machine-precision parity.

Host-independent: no pinned environment / BLAS needed."""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R, RigidTransform as RT

OUT = os.environ.get('RIGID_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'spatial_rigid.json')

rng = np.random.default_rng(20261013)
cases = []


def randrot(n):
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return R.from_quat(q)


for ci in range(6):
    rot = randrot(1)
    trans = rng.standard_normal(3)
    tf = RT.from_components(trans, rot)
    pts = rng.standard_normal((5, 3))
    # a second transform to compose with (A * B)
    rotB = randrot(1)
    transB = rng.standard_normal(3)
    tfB = RT.from_components(transB, rotB)
    comp = tf * tfB
    cases.append({
        'quat': rot.as_quat()[0].tolist(),
        'trans': trans.tolist(),
        'matrix': tf.as_matrix().tolist(),
        'inv_matrix': tf.inv().as_matrix().tolist(),
        'points': pts.tolist(),
        'apply': np.atleast_2d(tf.apply(pts)).tolist(),
        'apply_inv': np.atleast_2d(tf.apply(pts, inverse=True)).tolist(),
        'quat_b': rotB.as_quat()[0].tolist(),
        'trans_b': transB.tolist(),
        'compose_matrix': comp.as_matrix().tolist(),
        'compose_apply': np.atleast_2d(comp.apply(pts)).tolist(),
    })

# identity + from_translation + from_rotation edge cases
tf = RT.identity()
cases.append({
    'quat': R.identity().as_quat().tolist(),
    'trans': [0.0, 0.0, 0.0],
    'matrix': tf.as_matrix().tolist(),
    'inv_matrix': tf.inv().as_matrix().tolist(),
    'points': [[1.0, 2.0, 3.0]],
    'apply': np.atleast_2d(tf.apply([[1.0, 2.0, 3.0]])).tolist(),
    'apply_inv': np.atleast_2d(tf.apply([[1.0, 2.0, 3.0]], inverse=True)).tolist(),
    'quat_b': R.identity().as_quat().tolist(),
    'trans_b': [0.0, 0.0, 0.0],
    'compose_matrix': (tf * tf).as_matrix().tolist(),
    'compose_apply': np.atleast_2d((tf * tf).apply([[1.0, 2.0, 3.0]])).tolist(),
})

out = {'module': 'spatial.transform', 'scipy': __import__('scipy').__version__,
       'rigid_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial_rigid: {len(cases)} cases')
