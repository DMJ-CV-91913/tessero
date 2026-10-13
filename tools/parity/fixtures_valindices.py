#!/usr/bin/env python3
"""scipy.ndimage.value_indices fixtures.

value_indices(arr, ignore_value=None) returns a dict mapping each distinct integer value in ``arr`` to a
tuple of per-dimension index arrays -- exactly ``numpy.nonzero(arr == v)`` for that value, in C order. The
pure-PHP ``Tessero\\ValueIndices::valueIndices`` reproduces it on both backends. The result is a pure
selection (no arithmetic on the data), so it is bit-exact and host-independent; this generator does not
require the pinned numerical environment.

Encoding (consumed by tools/parity/run-tests.php via the existing {'dict'}+{'list'} comparators):
  expect = {'dict': {str(v): {'list': [enc(idx_dim0), enc(idx_dim1), ...]}}}
with the dict keys in SORTED-ascending order -- the same order Tessero returns -- so the runner's
``array_keys($want['dict']) !== array_keys($got)`` key-order check holds.
"""
import json
import os
import sys

import numpy as np
import scipy.ndimage as nd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'parity'))
import fixtures_np as F  # noqa: E402  -- enc / enc_result

OUT = os.environ.get('VALINDICES_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'ndimage_valindices.json')


def enc_expect(arr, ignore_value):
    """value_indices(arr, ignore_value) -> {'dict': {str(v): {'list': [per-dim idx arrays]}}}, keys sorted."""
    kw = {} if ignore_value is None else {'ignore_value': ignore_value}
    vi = nd.value_indices(arr, **kw)
    d = {}
    for v in sorted(vi.keys()):                      # ascending numeric order (ints, negatives included)
        tup = vi[v]                                  # tuple of per-dimension int64 index arrays
        d[str(int(v))] = {'list': [F.enc(np.asarray(a)) for a in tup]}
    return {'dict': d}


def case(name, arr, ignore_value=None):
    return {
        'name': name,
        'ndim': int(arr.ndim),
        'input': F.enc(arr),
        'ignore_value': ignore_value,
        'expect': enc_expect(arr, ignore_value),
    }


cases = []

# 2-D integer array (the proto's canonical matrix): values {0,1,2,3}
a2 = np.array([[1, 1, 0, 2],
               [1, 0, 0, 2],
               [0, 0, 3, 3]], dtype=np.int64)
cases.append(case('2d_basic', a2, None))
cases.append(case('2d_ignore0', a2, 0))

# 2-D with negative values + ignore
an = np.array([[-1, -1, 2],
               [0, 2, 2]], dtype=np.int64)
cases.append(case('2d_negatives', an, None))
cases.append(case('2d_negatives_ignore_neg1', an, -1))

# 3-D integer array (deterministic): values {0,1,2}
rng = np.random.default_rng(7)
a3 = rng.integers(0, 3, size=(2, 3, 4)).astype(np.int64)
cases.append(case('3d_basic', a3, None))
cases.append(case('3d_ignore0', a3, 0))

# 3-D wider value range, ignore a mid value
a3b = rng.integers(-1, 4, size=(2, 2, 3)).astype(np.int64)
cases.append(case('3d_range', a3b, None))
cases.append(case('3d_range_ignore1', a3b, 1))

out = {'module': 'ndimage', 'scipy': __import__('scipy').__version__, 'valindices_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'ndimage_valindices: {len(cases)} value_indices cases -> {OUT}')
