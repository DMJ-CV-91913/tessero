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


def csr_dict(X):
    X = sp.csr_array(X); X.sum_duplicates(); X.sort_indices(); X.eliminate_zeros()
    return {'data': F.enc(X.data), 'indices': F.enc(X.indices), 'indptr': F.enc(X.indptr), 'shape': F.enc(np.array(X.shape))}


# eye / eye_array (same cases) and identity
ec = [{'args': [F.enc(m), F.enc(n), F.enc(k)], 'kwargs': {}, 'expect': {'dict': csr_dict(sp.eye_array(m, n, k=k))}, 'compare': 'tol'}
      for (m, n, k) in [(4, 4, 0), (5, 6, 1), (5, 6, -2), (3, 5, 0), (6, 3, 2)]]
calls.append({'fn': 'eye', 'cases': ec})
calls.append({'fn': 'eye_array', 'cases': ec})
calls.append({'fn': 'identity', 'cases': [{'args': [F.enc(n)], 'kwargs': {}, 'expect': {'dict': csr_dict(sp.identity(n))}, 'compare': 'tol'} for n in (1, 3, 5)]})
# kron (general shapes), kronsum (square)
kr = []
for A, B in [(dense(3, 4), dense(2, 2)), (dense(2, 3), dense(3, 1)), (np.diag([1., 2., 3.]), dense(2, 2))]:
    kr.append({'args': [F.enc(A), F.enc(B)], 'kwargs': {}, 'expect': {'dict': csr_dict(sp.kron(A, B))}, 'compare': 'tol'})
calls.append({'fn': 'kron', 'cases': kr})
ks = []
for A, B in [(dense(3, 3), dense(2, 2)), (np.diag([2., 3.]), dense(3, 3)), (dense(2, 2), np.diag([1., 2., 3., 4.]))]:
    ks.append({'args': [F.enc(A), F.enc(B)], 'kwargs': {}, 'expect': {'dict': csr_dict(sp.kronsum(A, B))}, 'compare': 'tol'})
calls.append({'fn': 'kronsum', 'cases': ks})
# tril / triu
for fn_, fx in (('tril', sp.tril), ('triu', sp.triu)):
    cs = []
    for A in (dense(5, 5), dense(4, 6), dense(6, 4)):
        for k in (-1, 0, 2):
            cs.append({'args': [F.enc(A), F.enc(k)], 'kwargs': {}, 'expect': {'dict': csr_dict(fx(A, k))}, 'compare': 'tol'})
    calls.append({'fn': fn_, 'cases': cs})
# find: row, col, data (row-major order)
fc = []
for A in MATS[:4] + [dense(5, 5)]:
    I, J, V = sp.find(sp.csr_array(A))
    order = np.lexsort((J, I))
    fc.append({'args': [F.enc(A)], 'kwargs': {},
               'expect': {'dict': {'row': F.enc(I[order].astype(np.int32)), 'col': F.enc(J[order].astype(np.int32)), 'data': F.enc(V[order])}}, 'compare': 'tol'})
calls.append({'fn': 'find', 'cases': fc})
# spdiags: set diagonals of an m-by-n matrix from rows of data
sd = []
for (m, n, offs) in [(5, 6, [-1, 0, 2]), (4, 4, [0, 1, -2]), (6, 4, [-3, 0, 1]), (3, 5, [0])]:
    data = rng.uniform(-5, 5, (len(offs), max(m, n)))
    sd.append({'args': [F.enc(data), F.enc(np.array(offs)), F.enc(m), F.enc(n)], 'kwargs': {},
               'expect': {'dict': csr_dict(sp.spdiags(data, offs, m, n))}, 'compare': 'tol'})
calls.append({'fn': 'spdiags', 'cases': sd})
# block_diag: block-diagonal assembly (blocks passed as positional args)
bd = []
for mats in [[dense(2, 3), dense(3, 2), dense(1, 4)], [dense(2, 2), dense(3, 3)], [dense(4, 1), dense(1, 4)]]:
    bd.append({'args': [F.enc(M) for M in mats], 'kwargs': {},
               'expect': {'dict': csr_dict(sp.block_diag(mats))}, 'compare': 'tol'})
calls.append({'fn': 'block_diag', 'cases': bd})
# diags / diags_array: build from per-offset diagonals (correct length each), explicit shape
def diag_len(m, n, k):
    return min(m - max(0, -k), n - max(0, k))
dg = []
for (m, n, offs) in [(5, 5, [-1, 0, 2]), (4, 6, [0, 1]), (6, 4, [-2, 0]), (5, 5, [-1, 1])]:
    diagonals = [rng.uniform(-5, 5, diag_len(m, n, k)) for k in offs]
    expect = {'dict': csr_dict(sp.diags(diagonals, offs, shape=(m, n)))}
    dg.append({'args': [F.enc([np.asarray(d) for d in diagonals]), F.enc(np.array(offs)), F.enc(np.array([m, n]))],
               'kwargs': {}, 'expect': expect, 'compare': 'tol'})
calls.append({'fn': 'diags', 'cases': dg})
calls.append({'fn': 'diags_array', 'cases': dg})

out = {'module': 'sparse', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'sparse: {len(calls)} constructors, {sum(len(c["cases"]) for c in calls)} cases')
