#!/usr/bin/env python3
"""scipy.ndimage.geometric_transform fixtures (carve-out).

Named, reconstructible coordinate mappings (like the scipy.optimize solver fixtures): each case records a
mapping `kind` plus its params, the input array, and the (order, mode, cval). The expected output is scipy's
own geometric_transform with the equivalent Python mapping -- the PHP side rebuilds the same mapping from the
kind tag, walks the output coordinates, and delegates to map_coordinates (orders 0-5, all modes).

Integer-grid B-spline interpolation; no BLAS -- host-independent, so this generator does not require the
pinned environment (it still records env for provenance, consistent with the other ndimage fixtures)."""
import json, os
import numpy as np
import scipy
import scipy.ndimage as nd
import fixtures_np as F  # enc / enc_result

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'ndimage_geomxform.json')
rng = np.random.default_rng(20261012)


def py_mapping(kind, p):
    """Build the Python mapping callable for a named kind (must mirror the PHP rebuild in run-tests.php)."""
    if kind == 'shift':
        s = p['shift']
        return lambda o: tuple(o[i] - s[i] for i in range(len(o)))
    if kind == 'scale':
        a = p['scale']
        return lambda o: tuple(o[i] / a[i] for i in range(len(o)))
    if kind == 'affine':                      # input = M @ output + b   (M row-major, square ndim x ndim)
        M, b = p['matrix'], p['offset']
        n = len(b)
        return lambda o: tuple(sum(M[i][j] * o[j] for j in range(n)) + b[i] for i in range(n))
    if kind == 'swap':                        # transpose-like axis permutation (2-D)
        return lambda o: (o[1], o[0])
    raise ValueError(f'unknown mapping kind {kind}')


def case(inp, kind, params, order, mode='constant', cval=0.0, output_shape=None):
    mp = py_mapping(kind, params)
    exp = nd.geometric_transform(inp, mp, output_shape=output_shape, order=order, mode=mode, cval=cval)
    c = {'kind': kind, 'params': params, 'input': F.enc(inp), 'order': int(order),
         'mode': mode, 'cval': float(cval), 'expect': F.enc_result(exp, [])}
    if output_shape is not None:
        c['output_shape'] = list(output_shape)
    return c


A = rng.uniform(-5, 5, (5, 6))
B = rng.uniform(-5, 5, (6, 6))
C = np.arange(30.0).reshape(5, 6)

cases = []
# NOTE on orders/modes: orders 0 and 1 are matched on every boundary mode. For the B-spline orders (>=2)
# Tessero's map_coordinates (and hence this carve-out) reproduces scipy exactly under the fold modes
# 'mirror' and 'reflect' -- the prefilter's boundary conditions there are the fold modes. The 'constant',
# 'nearest' and 'wrap' prefilter extensions differ, so high-order cases here use the fold modes only,
# exactly as the existing map_coordinates fixtures (fixtures_ndimage.py) do (they test orders 2-5 only
# with mode='mirror').

# --- shift mapping: input = output - shift  (so the image shifts by +shift) --------------------------
for order in (0, 1):
    for mode in ('constant', 'nearest', 'reflect', 'mirror', 'wrap'):
        cases.append(case(A, 'shift', {'shift': [1.3, -0.7]}, order, mode))
cases.append(case(A, 'shift', {'shift': [1.3, -0.7]}, 3, 'mirror'))
cases.append(case(A, 'shift', {'shift': [0.5, 0.5]}, 1, 'constant', cval=2.5))
cases.append(case(C, 'shift', {'shift': [-1.0, 2.0]}, 3, 'mirror'))

# --- scale mapping: input = output / scale (zoom-like resample onto the same grid) -------------------
for order in (0, 1):
    cases.append(case(A, 'scale', {'scale': [1.5, 0.8]}, order, 'reflect'))
cases.append(case(A, 'scale', {'scale': [1.5, 0.8]}, 3, 'mirror'))
cases.append(case(B, 'scale', {'scale': [0.9, 1.1]}, 3, 'mirror'))

# --- affine mapping: input = M @ output + offset -----------------------------------------------------
Maff = [[0.9, 0.1], [-0.2, 1.1]]
off = [0.37, -0.29]
for order in (0, 1):
    for mode in ('constant', 'nearest', 'reflect', 'mirror', 'wrap'):
        cases.append(case(A, 'affine', {'matrix': Maff, 'offset': off}, order, mode))
cases.append(case(A, 'affine', {'matrix': Maff, 'offset': off}, 3, 'mirror'))
# affine onto a different output shape
cases.append(case(A, 'affine', {'matrix': [[1.0, 0.0], [0.0, 1.0]], 'offset': [0.0, 0.0]}, 1,
                  'constant', output_shape=[3, 4]))

# --- swap (axis transpose) mapping -------------------------------------------------------------------
for order in (0, 1):
    cases.append(case(C, 'swap', {}, order, 'reflect', output_shape=[6, 5]))
cases.append(case(C, 'swap', {}, 3, 'mirror', output_shape=[6, 5]))

doc = {'module': 'ndimage', 'scipy': scipy.__version__,
       'env': {'numpy': np.__version__, 'scipy': scipy.__version__, 'pinned': False},
       'geomxform_cases': cases}
with open(OUT, 'w') as fh:
    json.dump(doc, fh)
print(f'wrote {OUT}: {len(cases)} geomxform cases')
