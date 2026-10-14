#!/usr/bin/env python3
r"""scipy.sparse.save_npz / load_npz round-trip fixtures.

Two directions, both carrying the archive CONTENT as base64 so run-tests.php writes it to a temp file:

  - load cases ('load'): scipy.sparse.save_npz writes the .npz; the fixture carries those bytes and the expected
    dense matrix. Tessero's Io\SparseNpz::load must recover the CSR triple (reconstructed dense compared exactly).
  - save cases ('save'): the fixture carries the CSR triple + shape + is_array; Tessero writes a .npz; the runner
    reads the PHP-written archive back with Io\SparseNpz::load and checks the dense + triple + flags against
    'expect'. (scipy.sparse.load_npz reading PHP's archive is confirmed in VERIFY.py / e2e.php during de-risk.)

Pure-PHP (Io\SparseNpz over Io\Npy + ext-zip); host-independent, no pinned BLAS needed.
"""
import base64
import json
import os
import tempfile

import numpy as np
import scipy
import scipy.sparse as sp

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'sparse_npz.json')


def _npz_b64(spmat):
    f = tempfile.mktemp(suffix='.npz')
    sp.save_npz(f, spmat)
    with open(f, 'rb') as fh:
        b = fh.read()
    os.remove(f)
    return base64.b64encode(b).decode('ascii')


def _expect(A, spmat, is_array):
    return {
        'dense': A.tolist(),
        'data': spmat.data.tolist(),
        'indices': spmat.indices.tolist(),
        'indptr': spmat.indptr.tolist(),
        'shape': list(A.shape),
        'format': 'csr',
        'is_array': bool(is_array),
    }


def load_case(A, is_array):
    A = np.asarray(A, dtype=np.float64)
    spmat = sp.csr_array(A) if is_array else sp.csr_matrix(A)
    return {'dir': 'load', 'npz_b64': _npz_b64(spmat), 'expect': _expect(A, spmat, is_array)}


def save_case(A, is_array):
    A = np.asarray(A, dtype=np.float64)
    spmat = sp.csr_array(A) if is_array else sp.csr_matrix(A)
    return {
        'dir': 'save',
        'csr': {
            'data': spmat.data.tolist(),
            'indices': spmat.indices.tolist(),
            'indptr': spmat.indptr.tolist(),
            'shape': list(A.shape),
        },
        'is_array': bool(is_array),
        'expect': _expect(A, spmat, is_array),
    }


MATS = [
    [[1., 0, 2, 0], [0, 3, 0, 0], [4, 0, 0, 5]],
    [[1., 0, 2], [0, 3, 0], [4, 0, 0]],
    [[0., 0, 0], [0, 0, 0]],              # all-zero (empty CSR)
    [[1.5, -2.25], [0, 0], [3.0, 4e10]],
    np.eye(5).tolist(),                   # identity
    [[1., 2, 3, 4, 5]],                   # single row
    [[1.], [2.], [3.]],                   # single column
    [[-0.0, 1.25, 0], [0, 0, -3.5]],
]

cases = []
for A in MATS:
    cases.append(load_case(A, True))
    cases.append(load_case(A, False))
    cases.append(save_case(A, True))
    cases.append(save_case(A, False))

out = {'module': 'sparse_npz', 'scipy': scipy.__version__, 'sparse_npz_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'sparse_npz: {len(cases)} cases (load + save, csr_array + csr_matrix)')
