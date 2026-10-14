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

# ===== APPEND to tools/parity/fixtures_sparse.py, before the final out=/json.dump block =====
# (uses the module-level `dense`, `rng`, `F`, `csr_dict`, and `calls` already defined there)

# bmat / block_array: 2-D grid of dense blocks (None allowed), passed as a row-major flat sequence of
# length nrows*ncols plus the grid dimensions. The sequence is encoded with {'list': [...]} so run-tests
# decodeValue() rebuilds it as a PHP list (reaching the routine's `blocks[]` seq slot); a None block is a
# JSON null, which decodeValue passes through unchanged (-> tsr_arg kind 0 in the kernel).
def _bmat_case(nr, nc, blocks):
    import scipy.sparse as _sp
    grid = [[blocks[i * nc + j] for j in range(nc)] for i in range(nr)]
    R = _sp.bmat(grid)
    seq = {'list': [None if b is None else F.enc(b) for b in blocks]}
    return {'args': [seq, F.enc(nr), F.enc(nc)], 'kwargs': {}, 'expect': {'dict': csr_dict(R)}, 'compare': 'tol'}

_bm = [
    _bmat_case(2, 2, [dense(2, 2), dense(2, 3), dense(1, 2), dense(1, 3)]),
    _bmat_case(2, 2, [dense(2, 2), None, None, dense(2, 2)]),
    _bmat_case(3, 1, [dense(2, 3), dense(1, 3), dense(4, 3)]),
    _bmat_case(2, 2, [dense(2, 2), None, dense(3, 2), dense(3, 1)]),
    _bmat_case(2, 3, [dense(2, 2), dense(2, 1), dense(2, 3), None, dense(1, 1), None]),
    _bmat_case(2, 2, [None, dense(2, 3), dense(1, 2), None]),
    _bmat_case(3, 2, [dense(1, 2), dense(1, 2), None, dense(2, 2), dense(1, 2), None]),
    _bmat_case(2, 2, [dense(3, 2), dense(3, 3), dense(2, 2), dense(2, 3)]),
]
calls.append({'fn': 'bmat', 'cases': _bm})
calls.append({'fn': 'block_array', 'cases': _bm})

# permute_dims(A, axes) / swapaxes(A, axis1, axis2) on 2-D sparse: transpose (axes=(0,1) is identity).
import scipy.sparse as _sp2
_pm, _sw = [], []
for A in (dense(3, 4), dense(4, 3), dense(2, 5), dense(5, 2), np.diag([1., 2., 3.]), dense(1, 4), dense(4, 1), dense(3, 3), dense(6, 2)):
    sA = _sp2.csr_array(A)
    _pm.append({'args': [F.enc(A), None], 'kwargs': {}, 'expect': {'dict': csr_dict(_sp2.permute_dims(sA))}, 'compare': 'tol'})
    _pm.append({'args': [F.enc(A), {'list': [F.enc(1), F.enc(0)]}], 'kwargs': {}, 'expect': {'dict': csr_dict(_sp2.permute_dims(sA, (1, 0)))}, 'compare': 'tol'})
    _pm.append({'args': [F.enc(A), {'list': [F.enc(0), F.enc(1)]}], 'kwargs': {}, 'expect': {'dict': csr_dict(_sp2.permute_dims(sA, (0, 1)))}, 'compare': 'tol'})
    _sw.append({'args': [F.enc(A), F.enc(0), F.enc(1)], 'kwargs': {}, 'expect': {'dict': csr_dict(_sp2.swapaxes(sA, 0, 1))}, 'compare': 'tol'})
    _sw.append({'args': [F.enc(A), F.enc(0), F.enc(0)], 'kwargs': {}, 'expect': {'dict': csr_dict(_sp2.swapaxes(sA, 0, 0))}, 'compare': 'tol'})
calls.append({'fn': 'permute_dims', 'cases': _pm})
calls.append({'fn': 'swapaxes', 'cases': _sw})

# ===== APPEND to tools/parity/fixtures_sparse.py, before the final out=/json.dump block =====
# (uses the module-level `dense`, `rng`, `F`, `MATS`, and `calls` already defined there)
#
# lil / dok / bsr constructors from a dense matrix. These three scipy formats are mutable
# containers, but the storage built FROM a dense matrix is deterministic (no BLAS, host-
# independent), so -- exactly like csr/csc/coo/dia_array above -- we compare the canonical
# storage arrays:
#   lil_array(A).data/.rows flatten row-major to CSR data/column-indices (+ indptr row bounds)
#   dok_array(A) items sorted by (row, col) are the row-major coordinate triplets (= coo)
#   bsr_array(A, blocksize) is block-CSR: data (nnz_blocks, R, C), indices, indptr, blocksize

# lil: expose the row-list storage laid out as CSR (data, rows=column of each nnz, indptr, shape)
_lil = []
for A in MATS:
    L = sp.lil_array(A)
    data = np.array([v for rowvals in L.data for v in rowvals], dtype=np.float64)
    cols = np.array([c for rowcols in L.rows for c in rowcols], dtype=np.int32)
    indptr = np.concatenate(([0], np.cumsum([len(r) for r in L.rows]))).astype(np.int32)
    d = {'data': F.enc(data), 'rows': F.enc(cols), 'indptr': F.enc(indptr), 'shape': F.enc(np.array(A.shape))}
    _lil.append({'args': [F.enc(A)], 'kwargs': {}, 'expect': {'dict': d}, 'compare': 'tol'})
calls.append({'fn': 'lil_array', 'cases': _lil})
calls.append({'fn': 'lil_matrix', 'cases': _lil})

# dok: row-major coordinate triplets (sorted (row, col)) -> row, col, data, shape
_dok = []
for A in MATS:
    D = sp.dok_array(A)
    items = sorted(D.items())
    row = np.array([int(k[0]) for k, _ in items], dtype=np.int32)
    col = np.array([int(k[1]) for k, _ in items], dtype=np.int32)
    val = np.array([float(v) for _, v in items], dtype=np.float64)
    d = {'data': F.enc(val), 'row': F.enc(row), 'col': F.enc(col), 'shape': F.enc(np.array(A.shape))}
    _dok.append({'args': [F.enc(A)], 'kwargs': {}, 'expect': {'dict': d}, 'compare': 'tol'})
calls.append({'fn': 'dok_array', 'cases': _dok})
calls.append({'fn': 'dok_matrix', 'cases': _dok})

# bsr: block-CSR. Pass blocksize as two scalar args (R, C); use blocks that divide each matrix.
# shapes in MATS: (3,4),(4,4),(5,3),(2,6),(6,2),(4,4 diag),(3,4). Choose a dividing blocksize per matrix.
_bsr_specs = [
    (dense(4, 4), (2, 2)), (dense(4, 6), (2, 3)), (dense(6, 2), (3, 1)),
    (dense(4, 4), (1, 1)), (dense(6, 4), (2, 2)), (dense(2, 6), (2, 2)),
    (np.diag([1., 2., 3., 4.]), (2, 2)), (dense(6, 6), (3, 3)), (dense(4, 4), (4, 4)),
]
_bsr = []
for A, (R, C) in _bsr_specs:
    B = sp.bsr_array(A, blocksize=(R, C))
    B.sort_indices()  # scipy csr->bsr leaves block-columns unsorted; canonicalise to the sorted form the kernel emits
    d = {'data': F.enc(np.asarray(B.data, dtype=np.float64)),
         'indices': F.enc(np.asarray(B.indices, dtype=np.int32)),
         'indptr': F.enc(np.asarray(B.indptr, dtype=np.int32)),
         'shape': F.enc(np.array(B.shape)),
         'blocksize': F.enc(np.array([R, C]))}
    _bsr.append({'args': [F.enc(A), F.enc(R), F.enc(C)], 'kwargs': {}, 'expect': {'dict': d}, 'compare': 'tol'})
calls.append({'fn': 'bsr_array', 'cases': _bsr})
calls.append({'fn': 'bsr_matrix', 'cases': _bsr})

# ===== APPEND to tools/parity/fixtures_sparse.py, before the final out=/json.dump block =====
# (uses the module-level `dense`, `rng`, `F`, and `calls` already defined there)
#
# scipy.sparse.expand_dims(A, axis): insert a length-1 axis into a 2-D sparse array, producing an
# N-D (ndim==3) coo_array. Tessero's kernel routine sparse.expand_dims emits the canonical
# (C/row-major) coordinate storage: data[k] is the k-th nonzero of the expanded array in C-order,
# coords is (nnz x 3) int32, shape is the 3 expanded dims. Compared as that dict of arrays.
import scipy.sparse as _spED


def _ed_dict(A, axis):
    B = _spED.expand_dims(_spED.coo_array(A), axis=axis)
    coords = np.vstack(B.coords)                      # (ndim, nnz)
    order = np.lexsort(coords[::-1])                  # C-order: last axis varies fastest
    cc = np.ascontiguousarray(coords[:, order].T.astype(np.int32))   # (nnz, ndim)
    dd = np.asarray(B.data)[order]
    return {'data': F.enc(dd), 'coords': F.enc(cc), 'shape': F.enc(np.array(B.shape, dtype=np.int64))}


_ed = []
for A in (dense(3, 4), dense(4, 3), dense(2, 5), dense(5, 2),
          np.diag([1., 2., 3.]), dense(1, 4), dense(4, 1), dense(3, 3), dense(6, 2),
          np.array([[0., 2., 0., 3.], [0., 0., 0., 0.], [4., 0., 5., 0.]])):
    for ax in (0, 1, 2, -1, -2, -3):
        _ed.append({'args': [F.enc(A), F.enc(ax)], 'kwargs': {},
                    'expect': {'dict': _ed_dict(A, ax)}, 'compare': 'tol'})
calls.append({'fn': 'expand_dims', 'cases': _ed})

out = {'module': 'sparse', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'sparse: {len(calls)} constructors, {sum(len(c["cases"]) for c in calls)} cases')
