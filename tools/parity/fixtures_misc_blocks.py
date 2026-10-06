#!/usr/bin/env python3
"""The completeness blocks of the "np" module fixtures: spacing, modf, bitwise_and/or/xor, fromiter and can_cast.

These are integer, ulp (unit-in-the-last-place) and dtype-casting-rule operations: no BLAS and no transcendental
loop, so their reference values are bit-for-bit host-independent. They are kept in literal form (no rng draws) in
their own module so they can be produced two ways that always agree: by a full fixtures_np_numeric run, or by the
splice generator gen_misc.py, which writes them into tests/fixtures/parity/np_numeric.json on a host where a full
run cannot import numpy under the ADR-0012 OpenBLAS pin (e.g. a CPU without AVX512)."""
import numpy as np


def build(call):
    """Return {fn: [case, ...]} for the completeness routines. `call(fn, args, kwargs=None)` records one case."""
    b = {}

    b['spacing'] = [call('spacing', [np.array(x)]) for x in (
        [1.0, -1.0, 0.0, 0.5, 2.0, 1e10, -1e-10, 1e-300],
        [-3.5, 7.25, -100.0, 0.1, 1234.5],
        [np.inf, -np.inf, np.nan],
        [1, 2, 3, 4, 5],                                     # integers -> float64 spacing
    )] + [call('spacing', [np.array(1.0)]), call('spacing', [np.array(-8.0)])]

    b['modf'] = [call('modf', [np.array(x)]) for x in (
        [2.5, -2.5, 0.0, 3.0, -0.25, 1e10, -7.75],
        [0.1, 0.9, -0.1, -0.9, 5.5, -5.5],
        [1, 2, 3, 4],                                        # integers: fractional part is zero
        [-0.0, 0.0],
    )]

    for fn in ('bitwise_and', 'bitwise_or', 'bitwise_xor'):
        b[fn] = [call(fn, [np.array(a), np.array(c)]) for a, c in (
            ([12, 25, 100, 200, 7, 255], [10, 25, 55, 200, 3, 15]),
            ([[1, 2, 3], [255, 128, 64]], [[3, 2, 1], [15, 240, 32]]),
        )] + [
            call(fn, [np.array([1, 2, 4, 8, 16]), 12]),       # array and a scalar
            call(fn, [7, np.array([3, 5, 9, 15, 255])]),      # scalar and an array
            call(fn, [np.array([5, 6, 7]), np.array([3])]),   # broadcast a length-1 operand
            call(fn, [np.array([0xFF, 0x0F, 0xAA]), np.array([0x0F, 0xF0, 0x55])]),
            call(fn, [np.array([-1, -2, 5]), np.array([3, 6, -1])]),   # two's-complement negatives
            call(fn, [np.array([1, 2, 3, 4]), 0]),
            call(fn, [np.array([10, 20, 30]), -1]),
        ]

    b['fromiter'] = [
        call('fromiter', [np.array([1.0, 2.0, 3.0, 4.0]), 'float64']),
        call('fromiter', [np.array([1.5, 2.7, -3.9]), 'int64']),   # float -> int truncates toward zero
        call('fromiter', [np.array([1, 2, 3, 4, 5]), 'float64', 3]),
        call('fromiter', [np.array([1, 2, 3]), 'int32']),
        call('fromiter', [np.array([0.0, 1.0]), 'float64', 0]),
        call('fromiter', [np.array([10, 20, 30, 40]), 'float32', 2]),
        call('fromiter', [np.array([1, 0, 2, 0]), 'bool']),
    ]

    cs = [call('can_cast', [f, t]) for f, t in (
        ('int32', 'int64'), ('int32', 'float64'), ('int32', 'float32'), ('int64', 'float64'), ('uint8', 'int32'),
        ('float32', 'float64'), ('float64', 'complex128'), ('int64', 'complex128'), ('bool', 'int64'),
        ('float64', 'int64'), ('int64', 'int32'), ('float32', 'int32'), ('uint8', 'float32'), ('bool', 'bool'),
        ('float64', 'float64'), ('complex128', 'float64'))]
    cs += [call('can_cast', [f, t], {'casting': c}) for f, t, c in (
        ('int64', 'int32', 'unsafe'), ('int64', 'int32', 'same_kind'), ('int64', 'int32', 'safe'),
        ('float64', 'float32', 'same_kind'), ('int32', 'int64', 'no'), ('int32', 'int64', 'equiv'),
        ('float64', 'float64', 'no'), ('uint8', 'int32', 'safe'), ('float32', 'float64', 'no'))]
    b['can_cast'] = cs

    b['frexp'] = [call('frexp', [np.array(x)]) for x in (
        [1.0, 2.0, 3.0, 8.0, 0.5, 0.75, -4.0, 1024.0],
        [0.0, -0.0, 1e300, 1e-300, -123.5],
        [np.inf, -np.inf, np.nan],
        [1, 2, 4, 8],                                        # integers -> float64 mantissa
    )]

    b['ldexp'] = [
        call('ldexp', [np.array([1.0, 2.0, 3.0, 1.5]), np.array([2, 3, 0, -1])]),
        call('ldexp', [np.array([1.0, -1.0, 0.0, 0.5]), np.array([10, -3, 5, 4])]),
        call('ldexp', [np.array([1.5, 2.5, 3.5]), 2]),       # broadcast a scalar exponent
        call('ldexp', [1.0, np.array([0, 1, 2, 3, 4])]),     # broadcast a scalar mantissa
        call('ldexp', [np.array([[1.0, 2.0], [4.0, 8.0]]), np.array([[1, 2], [3, 4]])]),
    ]

    b['divmod'] = [
        call('divmod', [np.array([7, -7, 8, -8, 0, 15]), np.array([3, 3, -3, -3, 5, 4])]),   # integer, mixed signs
        call('divmod', [np.array([7.5, -7.5, 8.0, -8.25]), np.array([3.0, 3.0, -2.0, 2.5])]),  # float
        call('divmod', [np.array([10, 20, 30, 40]), 7]),      # broadcast an integer scalar
        call('divmod', [np.array([10.0, 20.0, 30.0]), 3.0]),  # broadcast a float scalar
        call('divmod', [np.array([[10, 11], [12, 13]]), np.array([[3, 4], [5, 6]])]),   # equal 2-D shapes
        call('divmod', [np.array([10, 20, 30, 40]), np.array([7])]),   # broadcast a length-1 operand
    ]

    b['bitwise_count'] = [call('bitwise_count', [np.array(x)]) for x in (
        [0, 1, 2, 3, 7, 8, 255, 256, 1023, 1024],
        [-1, -2, -7, -255, -256],                            # absolute value is counted
        [[1, 3, 7], [15, 31, 63]],
    )] + [call('bitwise_count', [np.array([9223372036854775807], dtype=np.int64)])]   # 2**63 - 1 -> 63 bits

    return b
