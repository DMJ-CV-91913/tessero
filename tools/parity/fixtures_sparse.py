#!/usr/bin/env python3
"""scipy.sparse constructor fixtures: csr/csc/coo/dia _array and _matrix built from a dense matrix. The stored
arrays (data, indices/indptr or row/col or offsets, shape) are integer/float storage with no BLAS, so the values
are host-independent and this generator does not require the pinned environment."""
import json, os
import numpy as np
import scipy.sparse as sp
import fixtures_np as F  # enc

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'sparse.json')
rng = np.random.default_rng(20261001)


def dense(m, n, density=0.4):
    A = rng.uniform(-5.0, 5.0, (m, n))
    A[rng.uniform(0.0, 1.0, (m, n)) > density] = 0.0
    return A


MATS = [dense(3, 4), dense(4, 4), dense(5, 3), dense(2, 6), dense(6, 2),
        np.diag([1.0, 2.0, 3.0, 4.0]),
        np.array([[0., 2., 0., 3.], [0., 0., 0., 0.], [4., 0., 5., 0.]])]

SPECS = [
    ('csr_array', sp.csr_array, ['data', 'indices', 'indptr', 'shape']),
    ('csr_matrix', sp.csr_matrix, ['data', 'indices', 'indptr', 'shape']),
    ('csc_array', sp.csc_array, ['data', 'indices', 'indptr', 'shape']),
    ('csc_matrix', sp.csc_matrix, ['data', 'indices', 'indptr', 'shape']),
    ('coo_array', sp.coo_array, ['data', 'row', 'col', 'shape']),
    ('coo_matrix', sp.coo_matrix, ['data', 'row', 'col', 'shape']),
    ('dia_array', sp.dia_array, ['data', 'offsets', 'shape']),
    ('dia_matrix', sp.dia_matrix, ['data', 'offsets', 'shape']),
]


def attr(obj, k):
    return F.enc(np.array(obj.shape)) if k == 'shape' else F.enc(np.asarray(getattr(obj, k)))


calls = []
for fn, ctor, keys in SPECS:
    cases = []
    for A in MATS:
        obj = ctor(A)
        cases.append({'args': [F.enc(A)], 'kwargs': {},
                      'expect': {'dict': {k: attr(obj, k) for k in keys}}, 'compare': 'tol'})
    calls.append({'fn': fn, 'cases': cases})

def stack_case(fn, mats):
    r = getattr(sp, fn)([sp.csr_array(M) for M in mats]).tocsr()
    r.sort_indices()
    d = {'data': F.enc(r.data), 'indices': F.enc(r.indices), 'indptr': F.enc(r.indptr), 'shape': F.enc(np.array(r.shape))}
    return {'args': [F.enc(M) for M in mats], 'kwargs': {}, 'expect': {'dict': d}, 'compare': 'tol'}


hs = [stack_case('hstack', [dense(3, 2), dense(3, 4)]), stack_case('hstack', [dense(4, 2), dense(4, 3), dense(4, 1)]),
      stack_case('hstack', [dense(2, 3), dense(2, 3)])]
vs = [stack_case('vstack', [dense(2, 4), dense(3, 4)]), stack_case('vstack', [dense(2, 3), dense(1, 3), dense(4, 3)]),
      stack_case('vstack', [dense(3, 2), dense(3, 2)])]
calls.append({'fn': 'hstack', 'cases': hs})
calls.append({'fn': 'vstack', 'cases': vs})

out = {'module': 'sparse', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'sparse: {len(calls)} constructors, {sum(len(c["cases"]) for c in calls)} cases')
