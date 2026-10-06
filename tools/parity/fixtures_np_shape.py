#!/usr/bin/env python3
"""Generate NumPy reference fixtures for array construction and manipulation (np.* routines in
csrc/src/np_shape.c): flip/roll/rot90/tile/repeat, joining and splitting, axis moves, diagonals and
triangles, the *_like constructors, gathering and index conversion.

    tools/parity/gen-fixtures.sh shape

Writes tests/fixtures/parity/np_shape.json (module "np", read by tools/parity/run-tests.php like np.json). Each
function gets ordinary cases over dtypes (float64, int64, bool), shapes (0-d to 4-d, empty axes) and every
argument form, plus the error cases NumPy raises on. Tuple and list results are encoded as {"list": [...]}.
Compare modes: "tol" (default) and "shape" (numpy.empty_like: only the shape and dtype are specified).
"""
import json, os, warnings

import numpy as np
import fixture_env
from fixtures_np import enc, enc_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'fixtures', 'parity', 'np_shape.json')
rng = np.random.default_rng(20260929)


def enc_out(r):
    """A result: arrays as arrays, 0-d arrays and NumPy scalars as scalars, tuples and lists as lists."""
    if isinstance(r, (tuple, list)):
        return {'list': [enc_out(x) for x in r]}
    if isinstance(r, (bool, int, float, np.generic)):
        return enc_scalar(r)
    r = np.asarray(r)
    if r.ndim == 0:
        return enc_scalar(r[()])
    return enc(r)


def call(fn, args, kwargs=None, compare='tol', f=None):
    kwargs = kwargs or {}
    f = f or getattr(np, fn)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            expect = enc_out(f(*args, **kwargs))
        except Exception as e:  # noqa: BLE001 - the exception class is the expectation
            expect = {'error': type(e).__name__}
    return {'args': [enc_arg(a) for a in args], 'kwargs': {k: enc_arg(v) for k, v in kwargs.items()},
            'expect': expect, 'compare': compare}


def enc_arg(a):
    if isinstance(a, tuple):
        return {'list': [enc_arg(x) for x in a]}
    if isinstance(a, list):
        return {'list': [enc_arg(x) for x in a]}
    return enc(a)


def F(*shape, lo=-10.0, hi=10.0):
    return rng.uniform(lo, hi, size=shape)


def I(*shape, lo=-9, hi=10):
    return rng.integers(lo, hi, size=shape).astype(np.int64)


def B(*shape):
    return rng.uniform(size=shape) < 0.5


ARRS = [F(5), I(7), B(4), F(3, 4), I(2, 3, 4), B(2, 3), F(1, 5), F(2, 1, 3, 2), np.array(3.5), np.array(7),
        np.zeros((0, 3)), np.zeros((2, 0), dtype=np.int64)]


def main():
    blocks = {}

    # ---- flip family, rot90, roll
    cs = []
    for a in ARRS:
        cs.append(call('flip', [a]))
        for ax in range(-a.ndim, a.ndim):
            cs.append(call('flip', [a], {'axis': ax}))
        if a.ndim >= 2:
            cs.append(call('flip', [a], {'axis': (0, -1)}))
            cs.append(call('flip', [a], {'axis': (0, 0)}))           # repeated axis
    cs += [call('flip', [F(3)], {'axis': 1}), call('flip', [np.array(2.0)], {'axis': 0})]
    blocks['flip'] = cs
    blocks['fliplr'] = [call('fliplr', [a]) for a in ARRS]
    blocks['flipud'] = [call('flipud', [a]) for a in ARRS]
    cs = []
    for a in (F(3, 4), I(2, 3, 4), B(2, 5), F(2, 3, 1, 4), np.zeros((0, 2))):
        for k in (-5, -1, 0, 1, 2, 3, 4, 7):
            cs.append(call('rot90', [a], {'k': k}))
        for axes in ((1, 0), (0, 2), (-1, -2), (2, 0)):
            cs.append(call('rot90', [a], {'k': 1, 'axes': axes}))
    cs += [call('rot90', [F(4)]), call('rot90', [F(2, 2)], {'axes': (0, 0)}), call('rot90', [F(2, 2)], {'axes': (0, 1, 2)}),
           call('rot90', [F(2, 2)], {'axes': (0, 2)}), call('rot90', [F(2, 2, 2)], {'axes': (0, -3)})]
    blocks['rot90'] = cs
    cs = []
    for a in ARRS:
        for sh in (0, 1, -1, 3, 100, -101):
            cs.append(call('roll', [a, sh]))
            if a.ndim >= 1:
                cs.append(call('roll', [a, sh], {'axis': -1}))
        if a.ndim >= 2:
            cs.append(call('roll', [a, (1, -2)], {'axis': (0, 1)}))
            cs.append(call('roll', [a, 2], {'axis': (0, 1)}))
            cs.append(call('roll', [a, (1, 2)], {'axis': 0}))       # shifts on one axis accumulate
            cs.append(call('roll', [a, (1, 2, 3)], {'axis': (0, 1)}))   # mismatched lengths
    cs += [call('roll', [F(6), 1.7]), call('roll', [F(6), (1, 2)]), call('roll', [F(3, 2), 1], {'axis': 2}),
           call('roll', [I(5), np.iinfo(np.int64).max])]
    blocks['roll'] = cs

    # ---- tile, repeat, resize
    cs = []
    for a in (F(3), I(2, 3), B(2, 2), np.array(4.0), np.zeros((0, 2))):
        for reps in (1, 2, 0, (2, 1), (1, 3), (2, 2, 2), (), (3,), (1, 1)):
            cs.append(call('tile', [a, reps]))
    cs += [call('tile', [F(3), -1]), call('tile', [F(3), (2, -1)]), call('tile', [F(2), 2.5])]
    blocks['tile'] = cs
    cs = []
    for a in (F(4), I(2, 3), B(3, 2), np.array(5)):
        for r in (0, 1, 3):
            cs.append(call('repeat', [a, r]))
            for ax in range(-a.ndim, a.ndim):
                cs.append(call('repeat', [a, r], {'axis': ax}))
    cs += [call('repeat', [I(2, 3), np.array([1, 0, 2])], {'axis': 1}), call('repeat', [F(3), np.array([2, 0, 1])]),
           call('repeat', [F(3), np.array([1, 2])]), call('repeat', [F(3), -1]), call('repeat', [F(3), 1.5]),
           call('repeat', [F(2, 2), np.array([1, 2])], {'axis': 2}), call('repeat', [F(3), np.array([[1, 2, 3]])])]
    blocks['repeat'] = cs
    cs = []
    for a in (F(5), I(2, 3), B(4), np.zeros(0)):
        for ns in (0, 3, 7, (2, 5), (3, 0), (1, 2, 3), ()):
            cs.append(call('resize', [a, ns]))
    cs += [call('resize', [F(3), (2, -1)]), call('resize', [np.array(3.0), (2, 2)])]
    blocks['resize'] = cs

    # ---- axes
    cs = []
    for a in (F(3), I(2, 3), np.array(1.5), B(2, 1, 2)):
        for ax in (0, -1, 1, (0, 1), (0, -1), (1, 1), 5, ()):
            cs.append(call('expand_dims', [a, ax]))
    blocks['expand_dims'] = cs
    cs = []
    for a in (F(1, 3, 1), I(1), B(2, 1), np.array(2.0), F(3, 4), np.zeros((1, 0, 1))):
        cs.append(call('squeeze', [a]))
        for ax in (0, -1, 2, (0, 2)):
            cs.append(call('squeeze', [a], {'axis': ax}))
    blocks['squeeze'] = cs
    cs = []
    for a in (F(2, 3), I(2, 3, 4), B(3, 1, 2)):
        for i, j in ((0, 1), (1, 0), (-1, 0), (0, 0), (0, 5)):
            cs.append(call('swapaxes', [a, i, j]))
    blocks['swapaxes'] = cs
    cs = []
    a4 = F(2, 3, 4, 5)
    for s, d in ((0, -1), (-1, 0), ((0, 1), (-1, -2)), ((0, 1, 2), (2, 1, 0)), ([], []), (1, 1), ((0, 1), (2,)),
                 ((0, 0), (1, 2)), (4, 0), ((0, 3), (3, 0))):
        cs.append(call('moveaxis', [a4, s, d]))
    cs += [call('moveaxis', [I(2, 3), 0, 1]), call('moveaxis', [np.array(1.0), (), ()])]
    blocks['moveaxis'] = cs
    cs = []
    for ax in range(-4, 5):
        for st in (-5, -4, -1, 0, 1, 3, 4, 5):
            cs.append(call('rollaxis', [a4, ax], {'start': st}))
    cs.append(call('rollaxis', [I(2, 3), 1]))
    blocks['rollaxis'] = cs
    blocks['matrix_transpose'] = [call('matrix_transpose', [a]) for a in (F(2, 3), I(2, 3, 4), B(1, 2), F(3), np.array(1.0))]
    for fn in ('atleast_1d', 'atleast_2d', 'atleast_3d'):
        f = getattr(np, fn)
        blocks[fn] = [call(fn, [a]) for a in (np.array(2.0), F(3), I(2, 3), B(2, 3, 4), F(1, 2, 3, 4))] + \
            [call(fn, [np.array(1), F(2), I(2, 2)]), call(fn, [3]), call(fn, [2.5, True]), call(fn, [], f=f)]
    cs = []
    for a in (F(3), I(1, 3), np.array(2.0), B(2, 1)):
        for shp in ((3,), (2, 3), (4, 2, 3), (), (0, 3), (3, 1), 3, (-1, 3)):
            cs.append(call('broadcast_to', [a, shp]))
    blocks['broadcast_to'] = cs
    blocks['broadcast_arrays'] = [call('broadcast_arrays', list(p)) for p in (
        (F(3), I(2, 1)), (np.array(1.0), F(2, 3), B(1, 3)), (F(2),), (F(3), F(4)), (I(2, 1, 4), F(3, 1), np.array(True)), ())]
    blocks['broadcast_shapes'] = [call('broadcast_shapes', list(p)) for p in (
        ((2, 1), (3,)), (3, (4, 1)), ((1, 2), (3, 1), (5, 1, 1)), ((2,), (3,)), (), ((0, 3), (1, 3)), ((2, 3),), ((-1,),))]

    # ---- joining
    cs = []
    pairs = [(F(2, 3), F(4, 3), 0), (F(2, 3), I(2, 5), 1), (I(3), B(2), 0), (B(2, 2), B(2, 2), -1), (F(2, 3, 4), F(2, 1, 4), 1),
             (F(2, 3), F(2, 3), None), (F(3), F(3, 1), 0), (np.array(1.0), np.array(2.0), 0), (F(2, 3), F(3, 3), 1),
             (np.zeros((0, 3)), F(2, 3), 0), (F(2, 3), F(2, 3), 2), (I(2, 2), I(3), None)]
    for a, b, ax in pairs:
        cs.append(call('concatenate', [[a, b]], {'axis': ax}))
    cs += [call('concatenate', [[F(2), F(3), I(1)]]), call('concatenate', [[]]), call('concatenate', [F(3, 2)]),
           call('concatenate', [[I(2), I(3)]], {'dtype': 'float64'}), call('concatenate', [[F(2), F(3)]], {'dtype': 'int64'}),
           call('concatenate', [[F(2), F(3)]], {'dtype': 'int64', 'casting': 'unsafe'}),
           call('concatenate', [[I(2), I(3)]], {'dtype': 'bool'}), call('concatenate', [[I(2), F(3)]], {'casting': 'safe'}),
           call('concatenate', [[I(2), F(3)]], {'casting': 'no'}), call('concatenate', [[B(2), I(3)]], {'casting': 'equiv'})]
    blocks['concatenate'] = cs
    blocks['concat'] = [call('concat', [[a, b]], {'axis': ax}) for a, b, ax in pairs[:6]]
    cs = []
    for arrs in ([F(3), F(3)], [I(2, 3), I(2, 3), I(2, 3)], [B(2), I(2)], [np.array(1.0), np.array(2)], [F(2, 0), F(2, 0)]):
        for ax in (0, -1, 1, -2):
            cs.append(call('stack', [arrs], {'axis': ax}))
    cs += [call('stack', [[F(2), F(3)]]), call('stack', [[]]), call('stack', [[F(2), F(2)]], {'axis': 2}),
           call('stack', [[I(2), I(2)]], {'dtype': 'float64'}), call('stack', [[F(2), F(2)]], {'dtype': 'int64'}),
           call('stack', [F(3, 2)])]
    blocks['stack'] = cs
    tups = [[F(3), F(3)], [I(2, 3), F(1, 3)], [np.array(1.0), F(1)], [B(2, 2, 2), I(2, 2, 2)], [F(3), F(4)], [F(2, 3), F(2, 4)],
            [F(2, 1), F(2, 3)], [F(4)], []]
    blocks['vstack'] = [call('vstack', [t]) for t in tups] + [call('vstack', [[I(2), I(2)]], {'dtype': 'float64'}),
                                                             call('vstack', [[F(2), F(2)]], {'dtype': 'int64'})]
    blocks['hstack'] = [call('hstack', [t]) for t in tups] + [call('hstack', [[I(2), I(2)]], {'dtype': 'float64'}),
                                                             call('hstack', [[F(2), F(2)]], {'dtype': 'int64', 'casting': 'unsafe'})]
    blocks['dstack'] = [call('dstack', [t]) for t in tups]
    blocks['column_stack'] = [call('column_stack', [t]) for t in tups] + [call('column_stack', [[F(3), I(3, 2)]])]
    cs = []
    for a in (F(3), I(2, 3), B(2, 3, 4), np.zeros((0, 2)), np.array(1.0)):
        for ax in range(-a.ndim, a.ndim):
            cs.append(call('unstack', [a], {'axis': ax}))
        cs.append(call('unstack', [a]))
    cs.append(call('unstack', [F(2, 2)], {'axis': 2}))
    blocks['unstack'] = cs

    # ---- splitting
    cs = []
    for a in (F(9), I(6, 4), B(2, 6, 3), np.zeros((0, 2))):
        for s in (1, 2, 3, 4, 5, 0, -1, [2, 5], [0], [], [3, 1], [-2, 20], [1, 1, 2]):
            for ax in (0, -1, 1):
                if ax >= a.ndim:
                    continue
                cs.append(call('split', [a, s], {'axis': ax}))
    cs += [call('split', [F(6), 2.0]), call('split', [F(6), 2.5]), call('split', [F(6), np.array([1, 4])]),
           call('split', [np.array(2.0), 1]), call('split', [F(6), 3], {'axis': 2})]
    blocks['split'] = cs
    cs = []
    for a in (F(10), I(5, 3), B(3, 4, 2), np.zeros(0)):
        for s in (1, 2, 3, 4, 7, 11, 0, -2, [2, 5], [], [4, 1], [-3], [100], [1.0]):
            cs.append(call('array_split', [a, s]))
            if a.ndim > 1:
                cs.append(call('array_split', [a, s], {'axis': -1}))
    cs += [call('array_split', [F(10), 3.7]), call('array_split', [F(10), 3], {'axis': 1})]
    blocks['array_split'] = cs
    for fn, arrs in (('hsplit', (F(6), I(3, 4), B(2, 6, 2), np.array(1.0))), ('vsplit', (F(6), I(4, 3), B(6, 2, 2))),
                     ('dsplit', (F(6), I(4, 3), B(2, 3, 4), F(1, 2, 6, 2)))):
        blocks[fn] = [call(fn, [a, s]) for a in arrs for s in (2, [1, 3], 3, [])]

    # ---- append, insert, delete, trim_zeros
    cs = []
    for a, v, ax in ((F(3), F(2), None), (I(2, 3), I(2, 3), None), (I(2, 3), I(1, 3), 0), (I(2, 3), F(2, 2), 1),
                     (F(3), 5, None), (I(3), 2.5, None), (I(3), True, None), (B(2), I(2), None), (F(2, 3), F(3), 0),
                     (F(2, 3), F(2, 3), 2), (np.array(1.0), np.array(2.0), None), (I(2, 2), I(2, 2), -1)):
        kw = {} if ax is None else {'axis': ax}
        cs.append(call('append', [a, v], kw))
    blocks['append'] = cs
    cs = []
    for a in (I(5), F(2, 3), B(4)):
        for obj in (0, 2, -1, 5, 10, -10, [1, 3], [0, 0], [3, 1, 2], [], np.array([True, False, True, False])):
            for v in (7, 1.5, [8, 9], np.array([[1, 2, 3]])):
                cs.append(call('insert', [a, obj, v]))
            if a.ndim > 1:
                cs.append(call('insert', [a, obj, 0], {'axis': 1}))
                cs.append(call('insert', [a, obj, [5, 6]], {'axis': 0}))
                cs.append(call('insert', [a, obj, np.array([[5], [6]])], {'axis': 1}))
    cs += [call('insert', [I(3), 1, np.nan]), call('insert', [F(2, 2), 1, 9.0], {'axis': 2}),
           call('insert', [I(3), np.array([[0, 1]]), 5]), call('insert', [I(4), 1.5, 5]), call('insert', [I(3), [1], [4, 5]])]
    blocks['insert'] = cs
    cs = []
    for a in (I(6), F(3, 4), B(5), F(2, 3, 2)):
        for obj in (0, 2, -1, 6, -7, [0, 2], [1, 1], [], [4, -1], np.array([True, False, True, False, True, False])):
            cs.append(call('delete', [a, obj]))
            for ax in range(a.ndim):
                cs.append(call('delete', [a, obj], {'axis': ax}))
    cs += [call('delete', [F(3), 1.5]), call('delete', [F(3), True]), call('delete', [F(3), 1], {'axis': 1})]
    blocks['delete'] = cs
    cs = []
    for a in (np.array([0, 0, 1, 2, 0, 3, 0, 0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.zeros(0),
              np.array([[0, 0, 0], [0, 1, 0], [0, 0, 2], [0, 0, 0]]), np.array([False, True, False])):
        for trim in ('fb', 'f', 'b', 'BF', 'F', 'x', 'fbf'):
            cs.append(call('trim_zeros', [a], {'trim': trim}))
        if a.ndim > 1:
            for ax in (0, 1, -1, (0, 1), ()):
                cs.append(call('trim_zeros', [a], {'axis': ax}))
    cs.append(call('trim_zeros', [np.array(0.0)]))
    blocks['trim_zeros'] = cs

    # ---- diagonals and triangles
    cs = []
    for v in (F(3), I(4), B(2), np.zeros(0), F(3, 4), I(4, 2), B(3, 3), F(2, 2, 2), np.array(1.0)):
        for k in (0, 1, -1, 2, -3, 5):
            cs.append(call('diag', [v], {'k': k}))
    blocks['diag'] = cs
    blocks['diagflat'] = [call('diagflat', [v], {'k': k}) for v in (F(3), I(2, 2), B(3), np.array(2.0)) for k in (0, 1, -2)]
    cs = []
    for a in (F(3, 4), I(4, 3, 2), B(2, 2, 2), F(2, 3, 4, 5)):
        for off in (0, 1, -1, 3, -5):
            for a1, a2 in ((0, 1), (1, 0), (0, -1), (-1, -2), (1, 1), (0, 5)):
                cs.append(call('diagonal', [a], {'offset': off, 'axis1': a1, 'axis2': a2}))
    cs.append(call('diagonal', [F(3)]))
    blocks['diagonal'] = cs
    cs = []
    for a in (F(4, 4), F(3, 5), I(3, 3), B(3, 3), F(2, 3, 4), I(3, 2, 2), F(40, 40), F(200, 200)):
        for off in (0, 1, -1, 3):
            cs.append(call('trace', [a], {'offset': off}))
            if a.ndim > 2:
                cs.append(call('trace', [a], {'offset': off, 'axis1': 1, 'axis2': 2}))
    big = np.array([[np.iinfo(np.int64).max, 0], [0, 5]])
    cs += [call('trace', [F(3)]), call('trace', [big]), call('trace', [I(3, 3)], {'dtype': 'float64'}),
           call('trace', [F(3, 3)], {'dtype': 'int64'})]
    blocks['trace'] = cs
    cs = []
    for n, m, k in ((3, None, 0), (3, 5, 0), (4, 2, 1), (3, 3, -1), (0, 3, 0), (3, 0, 0), (2, 2, 5), (2, 2, -5)):
        kw = {'k': k}
        if m is not None:
            kw['M'] = m
        cs.append(call('tri', [n], kw))
        cs.append(call('tri', [n], dict(kw, dtype='int64')))
        cs.append(call('tri', [n], dict(kw, dtype='bool')))
    cs.append(call('tri', [-1]))
    blocks['tri'] = cs
    for fn in ('tril', 'triu'):
        blocks[fn] = [call(fn, [a], {'k': k}) for a in (F(3, 4), I(4, 3), B(2, 3, 3), F(4), np.zeros((0, 2)), F(1, 2, 3, 4))
                      for k in (0, 1, -1, 3, -4)] + [call(fn, [np.array(1.0)])]
    for fn in ('tril_indices', 'triu_indices'):
        blocks[fn] = [call(fn, [n], {'k': k, 'm': m}) for n in (0, 1, 3, 4) for k in (0, 1, -1, 5) for m in (None, 2, 5)] + \
            [call(fn, [-1])]
    for fn in ('tril_indices_from', 'triu_indices_from'):
        blocks[fn] = [call(fn, [a], {'k': k}) for a in (F(3, 3), F(2, 4), I(4, 2), F(3), F(2, 2, 2)) for k in (0, 1, -2)]
    blocks['diag_indices'] = [call('diag_indices', [n], {'ndim': d}) for n in (0, 1, 4) for d in (1, 2, 3)] + \
        [call('diag_indices', [3])]
    blocks['diag_indices_from'] = [call('diag_indices_from', [a]) for a in (F(3, 3), F(2, 2, 2), F(2, 3), F(3), I(4, 4))]
    blocks['identity'] = [call('identity', [n], kw) for n in (0, 1, 4) for kw in ({}, {'dtype': 'int64'}, {'dtype': 'bool'})] + \
        [call('identity', [-1])]

    # ---- *_like
    for fn in ('zeros_like', 'ones_like'):
        cs = []
        for a in (F(3), I(2, 3), B(2), np.array(2.5), np.zeros((0, 2))):
            cs.append(call(fn, [a]))
            cs.append(call(fn, [a], {'dtype': 'int64'}))
            cs.append(call(fn, [a], {'dtype': 'bool'}))
            cs.append(call(fn, [a], {'shape': (2, 2)}))
            cs.append(call(fn, [a], {'shape': 3}))
            cs.append(call(fn, [a], {'shape': ()}))
        blocks[fn] = cs
    blocks['empty_like'] = [call('empty_like', [a], kw, compare='shape') for a in (F(3), I(2, 3), B(2)) for kw in ({}, {'dtype': 'bool'}, {'shape': (3, 1)})]
    cs = []
    for a in (F(3), I(2, 3), B(2), np.array(2.5), np.array(3)):
        for fv in (0, 7, -2.7, 2.7, True, np.nan, np.inf, np.array([1.0, 2.0, 3.0])):
            cs.append(call('full_like', [a, fv]))
        cs.append(call('full_like', [a, 5], {'dtype': 'float64'}))
        cs.append(call('full_like', [a, 1.5], {'shape': (2, 2)}))
    cs += [call('full_like', [I(2), 2 ** 62]), call('full_like', [F(2, 3), np.array([1.0, 2.0])])]
    blocks['full_like'] = cs

    # ---- gathering
    cs = []
    for a in (F(5), I(3, 4), B(2, 3, 2)):
        for idx in (0, -1, 2, [0, 2, 1], [[0, 1], [1, 0]], [], 10, -10):
            cs.append(call('take', [a, idx]))
            for ax in range(a.ndim):
                cs.append(call('take', [a, idx], {'axis': ax}))
            for mode in ('wrap', 'clip'):
                cs.append(call('take', [a, idx], {'mode': mode}))
    cs += [call('take', [F(3), [1.0]]), call('take', [F(3), 1], {'mode': 'bad'}), call('take', [np.zeros(0), [0]], {'mode': 'wrap'})]
    blocks['take'] = cs
    cs = []
    a = F(3, 4)
    for ax in (0, 1, -1):
        cs.append(call('take_along_axis', [a, np.argsort(a, axis=ax)], {'axis': ax}))
        cs.append(call('take_along_axis', [a, np.argmax(a, axis=ax, keepdims=True)], {'axis': ax}))
    b3 = I(2, 3, 4)
    cs += [call('take_along_axis', [b3, np.argsort(b3, axis=1)], {'axis': 1}),
           call('take_along_axis', [b3, np.zeros((1, 2, 1), dtype=np.int64)], {'axis': 1}),
           call('take_along_axis', [b3, np.array([[[-1]]])], {'axis': 2}),
           call('take_along_axis', [F(5), np.array([4, 0, 2])], {'axis': None}),
           call('take_along_axis', [F(2, 2), np.array([[3]])], {'axis': None}),
           call('take_along_axis', [F(3), np.array([5])]), call('take_along_axis', [F(3), np.array([1.0])]),
           call('take_along_axis', [F(3, 2), np.array([1])]), call('take_along_axis', [F(3, 2), np.zeros((2, 2), dtype=np.int64)], {'axis': 0}),
           call('take_along_axis', [F(3, 2), np.zeros((2, 3), dtype=np.int64)], {'axis': 0})]
    blocks['take_along_axis'] = cs
    nz = [np.array([0.0, 1.5, 0.0, -2.0]), I(3, 4), B(2, 3, 2), np.zeros((2, 0)), np.array(3.0), np.array(0), np.array([np.nan, 0.0])]
    blocks['nonzero'] = [call('nonzero', [a]) for a in nz]
    blocks['argwhere'] = [call('argwhere', [a]) for a in nz]
    blocks['flatnonzero'] = [call('flatnonzero', [a]) for a in nz]
    cs = []
    for a in (F(5), I(3, 4), B(2, 3)):
        for c in ([True, False, True], [1, 0, 1, 1, 0], [False] * 7, [0, 0, 0, 0, 0, 0, 1], [], np.array([2.5, 0.0, 1.0])):
            cs.append(call('compress', [c, a]))
            for ax in range(a.ndim):
                cs.append(call('compress', [c, a], {'axis': ax}))
    cs.append(call('compress', [np.array([[True]]), F(3)]))
    blocks['compress'] = cs
    blocks['extract'] = [call('extract', [c, a]) for a in (F(5), I(3, 4)) for c in
                         (B(5), B(3, 4), np.array([1, 0, 2]), np.ones(20, dtype=bool), np.zeros(20, dtype=bool), I(3, 4))]

    # ---- index conversion
    cs = []
    for idx in (0, 5, 22, 23, 24, -1, [1, 7, 11], [[0, 3], [5, 23]], np.array([], dtype=np.int64)):
        for shape in ((2, 3, 4), (24,), (4, 6)):
            for order in ('C', 'F'):
                cs.append(call('unravel_index', [idx, shape], {'order': order}))
    cs += [call('unravel_index', [1.0, (2, 3)]), call('unravel_index', [0, ()]), call('unravel_index', [3, (2, -1)]),
           call('unravel_index', [0, (0,)]), call('unravel_index', [1, (2, 3)], {'order': 'X'})]
    blocks['unravel_index'] = cs
    cs = []
    for mi, dims in ((([1, 2], [0, 3]), (3, 4)), (([0, 1, 1], [2, 0, 1], [3, 3, 0]), (2, 3, 4)), ((5, 7), (6, 8)),
                     (([3, -1], [1, 9]), (3, 4)), ((np.array([[1], [2]]), np.array([0, 1, 2])), (3, 4))):
        for mode in ('raise', 'wrap', 'clip', ('clip', 'wrap')):
            for order in ('C', 'F'):
                cs.append(call('ravel_multi_index', [mi, dims], {'mode': mode, 'order': order}))
    cs += [call('ravel_multi_index', [([1], [2]), (3,)]), call('ravel_multi_index', [([1.0], [2]), (3, 4)]),
           call('ravel_multi_index', [([1], [2]), (3, 4)], {'mode': 'bad'})]
    blocks['ravel_multi_index'] = cs
    blocks['indices'] = [call('indices', [d], kw) for d in ((2, 3), (3,), (), (2, 0, 2), (1, 2, 3))
                         for kw in ({}, {'sparse': True}, {'dtype': 'float64'}, {'dtype': 'float64', 'sparse': True})]
    blocks['ix_'] = [call('ix_', list(a)) for a in (([0, 1], [2, 4]), (np.array([True, False, True]), [1, 2, 3]), ([1, 2, 3],),
                                                    ([0], [1], [2]), ([],), (F(2, 2),), (np.array([2.5, 1.0]), [0]), ())]
    cs = []
    xs = (F(3), I(2), B(2))
    for n in (1, 2, 3):
        for idx in ('xy', 'ij'):
            for sp in (False, True):
                cs.append(call('meshgrid', list(xs[:n]), {'indexing': idx, 'sparse': sp}))
    cs += [call('meshgrid', [F(2, 2), F(3)]), call('meshgrid', [F(2)], {'indexing': 'xx'}), call('meshgrid', []),
           call('meshgrid', [np.array(2.0), F(3)]), call('meshgrid', [F(3), F(2)], {'copy': False})]
    blocks['meshgrid'] = cs

    # ---- comparisons and metadata
    x = F(3, 4)
    xn = x.copy()
    xn[1, 2] = np.nan
    cs = []
    for a, b in ((x, x.copy()), (x, x + 1e-16), (I(3), I(3)), (np.array([1, 2]), np.array([1.0, 2.0])), (x, x[:2]),
                 (xn, xn.copy()), (B(3), B(3)), (np.array([2 ** 60 + 1]), np.array([2 ** 60])),
                 (np.array([np.nan]), np.array([np.nan])), (np.zeros(0), np.zeros((0, 1))), (np.array(1.0), 1.0)):
        cs.append(call('array_equal', [a, b]))
        cs.append(call('array_equal', [a, b], {'equal_nan': True}))
    blocks['array_equal'] = cs
    blocks['array_equiv'] = [call('array_equiv', [a, b]) for a, b in (
        (F(3), F(3)), (np.array([1, 2]), np.array([[1, 2], [1, 2]])), (np.array([1, 2]), np.array([[1, 2], [1, 3]])),
        (np.array([1, 2]), np.array([1, 2, 3])), (np.array(5), np.array([5, 5])), (np.array([np.nan]), np.array([np.nan])))]
    blocks['ndim'] = [call('ndim', [a]) for a in ARRS[:9]]
    blocks['shape'] = [call('shape', [a]) for a in ARRS]
    blocks['size'] = [call('size', [a], kw) for a in (F(3, 4), I(2, 3, 4), np.array(1.0), np.zeros((0, 2)))
                      for kw in ({}, {'axis': 0}, {'axis': -1}, {'axis': (0, 1)}, {'axis': 5})]

    # ---- select, choose
    c1, c2 = B(3, 4), B(3, 4)
    cs = [call('select', [[c1, c2], [F(3, 4), F(3, 4)]]), call('select', [[c1, c2], [I(3, 4), I(3, 4)]], {'default': -1}),
          call('select', [[c1, c2], [I(3, 4), I(3, 4)]], {'default': 0.5}), call('select', [[c1], [B(3, 4)]]),
          call('select', [[c1, c2], [1, 2.5]]), call('select', [[c1, c2], [1, 2]], {'default': np.nan}),
          call('select', [[c1, c2], [I(4), F(3, 1)]]), call('select', [[np.array(True)], [F(2)]]),
          call('select', [[c1], [F(3, 4), F(3, 4)]]), call('select', [[], []]),
          call('select', [[I(3, 4)], [F(3, 4)]]), call('select', [[c1, c2], [True, False]]),
          call('select', [[np.array(False)], [7]], {'default': 3}), call('select', [[c1], [F(3, 4)]], {'default': F(3, 4)})]
    blocks['select'] = cs
    cs = []
    ch = [F(2, 3), F(2, 3), F(2, 3)]
    for a in (np.array([[0, 1, 2], [2, 1, 0]]), np.array([0, 2, 1]), np.array(1), np.array([[3, -1, 5], [0, 1, 2]])):
        for mode in ('raise', 'wrap', 'clip'):
            cs.append(call('choose', [a, ch], {'mode': mode}))
    cs += [call('choose', [np.array([0, 1]), [I(2), F(2)]]), call('choose', [np.array([0, 1, 1]), [B(3), B(3)]]),
           call('choose', [np.array([0.0, 1.0]), [F(2), F(2)]]), call('choose', [np.array([0, 1]), []]),
           call('choose', [np.array([[0], [1]]), [F(3), I(3)]])]
    blocks['choose'] = cs

    # PAR-2.3 verification: float k (NumPy's arange arithmetic), int64 edges, offsets beyond a C int, exact 0-d ints
    ks = [0.5, -0.5, 1.5, -1.5, 2.7, -2.7, 1e20, -1e20, float('nan'), float('inf'), 2**62, -(2**62), -(2**63), 2**63 - 1,
          -(2**63) + 1, -(2**63) + 2, 3.0]
    blocks['tri'] += [call('tri', [3, 4, k], {'dtype': 'int64'}) for k in ks]
    for fn in ('tril', 'triu'):
        blocks[fn] += [call(fn, [F(3, 4)], {'k': k}) for k in ks] + [call(fn, [F(3, 1)], {'k': k}) for k in (1e20, 0.5)]
    for fn in ('tril_indices', 'triu_indices'):
        blocks[fn] += [call(fn, [3, k, 4]) for k in ks]
    for fn in ('tril_indices_from', 'triu_indices_from'):
        blocks[fn] += [call(fn, [F(3, 4), k]) for k in ks]
    blocks['rot90'] += [call('rot90', [I(2, 3)], {'k': k}) for k in (1.5, 4.0, -1.0, 7.0, float('nan'), -2.5, 2.0, 0.25)]
    for fn in ('trace', 'diagonal'):
        blocks[fn] += [call(fn, [F(3, 4)], {'offset': o}) for o in (2**31 - 1, 2**31, -(2**31) + 1, 2**40, -(2**40))]
    blocks['trace'] += [call('trace', [np.array([[2**62 + 1, 1], [2, 2**60 + 3]], dtype=np.int64)])]
    blocks['ravel_multi_index'] += [call('ravel_multi_index', [(2**31 + 5, 3), (2**32, 7)])]
    blocks['concatenate'] += [call('concatenate', [[np.array([1.0]), np.array([[1.0]])]])]

    # second verification pass: dims that only count elements, empty triangles at k = -2^63, float offsets
    blocks['ravel_multi_index'] += [call('ravel_multi_index', [[1], [2**62]]), call('ravel_multi_index', [[2**31 - 1, 2**31 - 1], [2**31, 2**31]]),
                                    call('ravel_multi_index', [[1, 1], [2**62, 4]])]
    blocks['unravel_index'] += [call('unravel_index', [5, [2**62]]), call('unravel_index', [2**62 + 7, [2**31, 2**31, 2]]),
                                call('unravel_index', [5, [2**62, 4]])]
    blocks['tril_indices'] += [call('tril_indices', [0, -(2**63)]), call('tril_indices', [2, -(2**63), 0]), call('tril_indices', [0, -(2**63), 4])]
    blocks['triu_indices'] += [call('triu_indices', [0, -(2**63)]), call('triu_indices', [0, -(2**63) + 1])]
    blocks['trace'] += [call('trace', [F(3, 3)], {'offset': 1.0})]

    # PAR-4.1b: ascontiguousarray / asfortranarray (contiguous copy, ndim >= 1, optional dtype cast)
    for fn in ('ascontiguousarray', 'asfortranarray'):
        blocks[fn] = [call(fn, [F(7)]), call(fn, [F(3, 4)]), call(fn, [F(2, 3, 4)]), call(fn, [I(5)]),
                      call(fn, [F(3, 4)], {'dtype': 'int64'}), call(fn, [I(4)], {'dtype': 'float64'}),
                      call(fn, [F(2, 3)], {'dtype': 'float32'}), call(fn, [5.0]), call(fn, [I(1)])]

    out = {'module': 'np', 'numpy': np.__version__, 'env': fixture_env.env(),
           'calls': [{'fn': k, 'cases': v} for k, v in blocks.items()]}
    fixture_env.require_pinned()
    with open(OUT, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'np_shape: {len(blocks)} functions, {sum(len(v) for v in blocks.values())} calls')


if __name__ == '__main__':
    main()
