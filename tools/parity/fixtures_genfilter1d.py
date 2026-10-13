#!/usr/bin/env python3
"""scipy.ndimage.generic_filter1d fixtures.

generic_filter1d(input, function, filter_size, axis=-1, mode='reflect', cval=0.0, origin=0) slides a
user 1-D filter along ``axis``. For each line SciPy builds an *extended* line of length
``n + filter_size - 1`` (left = filter_size//2 + origin real samples folded in before index 0, the rest
after n-1, per the boundary ``mode``) and calls ``function(input_line, output_line)`` once, the callee
filling the length-``n`` output line in place.

The pure-PHP ``Tessero/Ndimage/GenericFilter1d::apply`` reproduces the identical extend-line +
callback-per-line contract (its callable is ``fn(list inbuf_ext, int $outlen): list``), one class on
both the FFI and ext backends. Because the harness cannot ship a PHP closure in JSON, each case names a
STENCIL (``mavg`` / ``fdiff`` / ``wstencil``) that the PHP runner reconstructs identically; we evaluate
the matching numpy stencil here through scipy's real generic_filter1d and dump the expected array.

Encoding (consumed by tools/parity/run-tests.php genfilter1d_cases): each case carries the stencil name +
params, filter_size, axis, mode, cval, origin, the input array and the expected output array.
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import generic_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'parity'))
import fixtures_np as F  # noqa: E402  -- enc

OUT = os.environ.get('GENFILTER1D_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'ndimage_genfilter1d.json')


# --- named line-operations, MIRRORED verbatim by makeStencil() in run-tests.php ---
def mavg(size):
    """moving average of each window of filter_size over the extended line."""
    def f(inbuf, outlen):
        out = np.empty(outlen)
        for i in range(outlen):
            out[i] = inbuf[i:i + size].mean()
        return out
    return f


def fdiff():
    """forward difference (filter_size=2): out[i] = in_ext[i+1] - in_ext[i]."""
    def f(inbuf, outlen):
        out = np.empty(outlen)
        for i in range(outlen):
            out[i] = inbuf[i + 1] - inbuf[i]
        return out
    return f


def wstencil(weights):
    """fixed weighted stencil of length filter_size over the extended line."""
    w = np.array(weights, dtype=float)
    size = len(weights)
    def f(inbuf, outlen):
        out = np.empty(outlen)
        for i in range(outlen):
            out[i] = (inbuf[i:i + size] * w).sum()
        return out
    return f


def build(stencil, fsize):
    kind = stencil['kind']
    if kind == 'mavg':
        return mavg(fsize)
    if kind == 'fdiff':
        return fdiff()
    if kind == 'wstencil':
        return wstencil(stencil['weights'])
    raise ValueError(kind)


def scipy_cb(pyfn):
    def cb(inbuf, outbuf):
        outbuf[:] = pyfn(inbuf, len(outbuf))
    return cb


def case(name, arr, stencil, fsize, axis=-1, mode='reflect', cval=0.0, origin=0):
    a = np.asarray(arr, dtype=float)
    exp = generic_filter1d(a, scipy_cb(build(stencil, fsize)), fsize,
                           axis=axis, mode=mode, cval=cval, origin=origin)
    return {
        'name': name,
        'input': F.enc(a),
        'stencil': stencil,
        'filter_size': int(fsize),
        'axis': int(axis),
        'mode': mode,
        'cval': float(cval),
        'origin': int(origin),
        'expect': F.enc(np.asarray(exp, dtype=float)),
    }


cases = []

# 1-D
cases.append(case('1d_mavg3_reflect', [1., 2., 3., 4., 5.], {'kind': 'mavg'}, 3, -1, 'reflect'))
cases.append(case('1d_fdiff_nearest', [1., 4., 9., 16., 25.], {'kind': 'fdiff'}, 2, -1, 'nearest'))
cases.append(case('1d_wstencil_constant', [2., 0., 1., 3., 7., 4.],
                  {'kind': 'wstencil', 'weights': [0.25, 0.5, 0.25]}, 3, 0, 'constant'))
cases.append(case('1d_mavg4_constant_cval2', [3., 1., 4., 1., 5., 9., 2.],
                  {'kind': 'mavg'}, 4, -1, 'constant', cval=2.0))
cases.append(case('1d_mavg3_wrap', [10., 20., 30., 40.], {'kind': 'mavg'}, 3, 0, 'wrap'))
cases.append(case('1d_mavg3_mirror', [1., 2., 3., 4., 5., 6.], {'kind': 'mavg'}, 3, -1, 'mirror'))

# 2-D along axis 0 / axis 1
cases.append(case('2d_mavg3_axis0_wrap', [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                  {'kind': 'mavg'}, 3, 0, 'wrap'))
cases.append(case('2d_mavg2_axis1_mirror', [[1., 2., 3., 4.], [5., 6., 7., 8.]],
                  {'kind': 'mavg'}, 2, 1, 'mirror'))
cases.append(case('2d_fdiff_axis1_reflect', [[1., 3., 6., 10.], [2., 5., 9., 14.]],
                  {'kind': 'fdiff'}, 2, 1, 'reflect'))
cases.append(case('2d_wstencil_axis0_nearest',
                  [[2., 1.], [4., 3.], [6., 5.], [8., 7.]],
                  {'kind': 'wstencil', 'weights': [1.0, -2.0, 1.0]}, 3, 0, 'nearest'))

out = {'module': 'ndimage', 'scipy': __import__('scipy').__version__, 'genfilter1d_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'ndimage_genfilter1d: {len(cases)} generic_filter1d cases -> {OUT}')
