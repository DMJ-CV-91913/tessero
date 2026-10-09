#!/usr/bin/env python3
"""Generate NumPy reference fixtures for the numerical array routines (np.* routines in csrc/src/np_numeric.c):
products, differences and integrals, interp, logspace/geomspace, vander and the polynomial helpers, windows,
sinc and i0, pad, nan_to_num and the type predicates, and the in-place writers.

    tools/parity/gen-fixtures.sh numeric

Writes tests/fixtures/parity/np_numeric.json (module "np"). Compare modes: "tol" (default rtol 1e-12), "shape"
(pad mode 'empty': the padded area is uninitialised in NumPy) and "inplace" (numpy.put, place, putmask, copyto,
fill_diagonal, put_along_axis return None; the expectation is the first argument after the call).

Per-call tolerances derived from the call's data (tools/parity/run-tests.php takes max(default, the call's)):
NumPy computes inner, vdot, tensordot and polymul on floats with BLAS dot products, whose summation order is
the BLAS kernel's; Tessero sums in index order. Both results lie within gamma_n * sum|a_k b_k| of the exact
dot product (gamma_n = n eps / (1 - n eps)), so they agree within 2 gamma_n * sum|a_k b_k| / |result|; that
bound (at least 1e-12) is the call's rtol, recorded with its reason.
"""
import json, os, warnings

import numpy as np
import fixture_env
from fixtures_np import enc, enc_scalar
from fixtures_np_shape import enc_out, enc_arg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'fixtures', 'parity', 'np_numeric.json')
rng = np.random.default_rng(20260930)
EPS = np.finfo(np.float64).eps


def call(fn, args, kwargs=None, compare='tol', f=None, tol=None):
    kwargs = kwargs or {}
    f = f or getattr(np, fn)
    enc_args = [enc_arg(a) for a in args]
    enc_kwargs = {k: enc_arg(v) for k, v in kwargs.items()}
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            if compare == 'inplace':
                target = args[0].copy()
                r = f(target, *args[1:], **kwargs)
                assert r is None
                expect = enc(target)
            else:
                expect = enc_out(f(*args, **kwargs))
        except Exception as e:  # noqa: BLE001 - the exception class is the expectation
            expect = {'error': type(e).__name__}
    c = {'args': enc_args, 'kwargs': enc_kwargs, 'expect': expect, 'compare': compare}
    if tol is not None:
        c['tol'] = tol
    return c


def contract_tol(result, abs_result, n):
    """rtol for a contraction of n products: 2 gamma_n sum|a_k b_k| / |result| over the outputs"""
    n = max(int(n), 1)
    g = n * EPS / (1 - n * EPS)
    r = np.abs(np.asarray(result, dtype=np.float64))
    s = np.asarray(abs_result, dtype=np.float64)
    with np.errstate(all='ignore'):
        q = np.where(r > 0, 2 * g * s / r, 0.0)
    rt = float(np.max(q)) if q.size else 0.0
    if not np.isfinite(rt) or rt <= 1e-12:
        return None
    return {'rtol': rt, 'why': 'BLAS dot product: summation order differs; bound 2 gamma_n sum|a_k b_k| / |result|'}


def F(*shape, lo=-10.0, hi=10.0):
    return rng.uniform(lo, hi, size=shape)


def I(*shape, lo=-9, hi=10):
    return rng.integers(lo, hi, size=shape).astype(np.int64)


def B(*shape):
    return rng.uniform(size=shape) < 0.5


def main():
    blocks = {}

    # ---- products
    blocks['outer'] = [call('outer', [a, b]) for a, b in (
        (F(3), F(4)), (I(2, 2), I(3)), (B(3), B(2)), (np.array(2.0), F(3)), (F(2), I(2)), (np.zeros(0), F(2)),
        (np.array([2 ** 62, 3]), np.array([4, -5])))]
    cs = []
    for a, b in ((F(5), F(5)), (F(2, 3), F(4, 3)), (I(3, 4), I(2, 4)), (B(4), B(4)), (F(2, 3, 2), I(3, 2)), (np.array(2.5), F(3)),
                 (F(3), F(4)), (np.zeros((2, 0)), np.zeros((3, 0))), (F(40), F(40)), (np.array([2 ** 62, 7]), np.array([4, 3]))):
        t = None
        if a.dtype.kind == 'f' and a.ndim and b.ndim and a.shape[-1] == b.shape[-1]:
            t = contract_tol(np.inner(a, b), np.inner(np.abs(a), np.abs(b)), a.shape[-1])
        cs.append(call('inner', [a, b], tol=t))
    blocks['inner'] = cs
    cs = []
    for a, b in ((F(5), F(5)), (F(2, 3), F(3, 2)), (I(6), I(6)), (B(4), B(4)), (F(3), I(3)), (F(3), F(4)), (F(100), F(100)),
                 (np.array([2 ** 62, 7]), np.array([4, 3]))):
        t = contract_tol(np.vdot(a, b), np.vdot(np.abs(a), np.abs(b)), a.size) if a.size == b.size and a.dtype.kind == 'f' else None
        cs.append(call('vdot', [a, b], tol=t))
    blocks['vdot'] = cs
    blocks['kron'] = [call('kron', [a, b]) for a, b in (
        (F(2, 2), F(3, 3)), (I(3), I(2)), (F(2), F(2, 3)), (B(2, 2), B(2)), (np.array(3.0), F(2, 2)), (F(2, 2, 2), I(2)),
        (np.zeros((0, 2)), F(2)))]
    cs = []
    for a, b, kw in ((F(3), F(3), {}), (F(4, 3), F(4, 3), {}), (I(3), I(3), {}), (F(2), F(2), {}), (F(2), F(3), {}),
                     (F(3), F(2), {}), (F(3, 4), F(3, 4), {'axis': 0}), (F(3, 2), F(3, 2), {'axisa': 0, 'axisb': 0}),
                     (F(4, 3), F(3), {'axisc': 0}), (F(4), F(4), {}), (B(3), B(3), {}), (np.array(1.0), F(3), {}),
                     (F(2, 3), F(3, 2), {'axisa': 1, 'axisb': 0, 'axisc': 1}), (F(3), F(3), {'axisc': 2})):
        cs.append(call('cross', [a, b], kw))
    blocks['cross'] = cs
    cs = []
    for a, b, ax in ((F(3, 4), F(4, 5), 1), (F(2, 3, 4), F(3, 4, 2), 2), (I(2, 3), I(3, 2), 1), (F(3), F(4), 0),
                     (F(2, 3, 4), F(4, 3, 2), ([1, 2], [1, 0])), (F(3, 3), F(3, 3), ([0, 1], [0, 1])),
                     (F(3, 4), F(4, 3), ([1], [0])), (F(2, 2), F(2, 2), 2), (F(3, 4), F(5, 4), 1), (B(2, 2), B(2, 2), 1),
                     (F(3, 4), F(4, 3), ([0, 0], [1, 1])), (F(3, 4), F(4, 5), -1)):
        t = None
        if a.dtype.kind == 'f':
            try:
                axa = list(range(a.ndim - ax, a.ndim)) if isinstance(ax, int) else list(ax[0])
                nsum = int(np.prod([a.shape[k] for k in axa])) if not isinstance(ax, int) or ax > 0 else 1
                t = contract_tol(np.tensordot(a, b, axes=ax), np.tensordot(np.abs(a), np.abs(b), axes=ax), nsum)
            except Exception:  # noqa: BLE001 - NumPy raises: no tolerance needed
                t = None
        cs.append(call('tensordot', [a, b], {'axes': ax}, tol=t))
    blocks['tensordot'] = cs

    cs = []
    for a, b in ((F(5), F(5)), (F(3), F(3)), (F(40), F(40))):   # vecdot: 1-D . 1-D
        cs.append(call('vecdot', [a, b], tol=contract_tol(np.vecdot(a, b), np.vecdot(np.abs(a), np.abs(b)), a.shape[-1])))
    blocks['vecdot'] = cs
    cs = []
    for a, b in ((F(3, 4), F(4)), (F(5, 5), F(5)), (F(2, 3, 4), F(2, 4))):   # matvec: (...,M,N) x (...,N)
        cs.append(call('matvec', [a, b], tol=contract_tol(np.matvec(a, b), np.matvec(np.abs(a), np.abs(b)), a.shape[-1])))
    blocks['matvec'] = cs
    cs = []
    for a, b in ((F(4), F(4, 3)), (F(5), F(5, 5)), (F(2, 3), F(2, 3, 4))):   # vecmat: (...,N) x (...,N,M)
        cs.append(call('vecmat', [a, b], tol=contract_tol(np.vecmat(a, b), np.vecmat(np.abs(a), np.abs(b)), a.shape[-1])))
    blocks['vecmat'] = cs
    blocks['poly'] = [call('poly', [np.array([1.0, 2.0, 3.0])]), call('poly', [np.array([-1.0, 0.5, 2.0, -3.0])]), call('poly', [rng.uniform(-2, 2, size=5)])]
    blocks['sort_complex'] = [call('sort_complex', [np.array([3.0, 1.0, 2.0, -1.0, 0.5])]), call('sort_complex', [rng.uniform(-3, 3, size=6)])]

    # ---- differences and integrals
    cs = []
    for a in (F(8), I(3, 5), B(6), F(2, 3, 4), np.array([2 ** 62, -(2 ** 62), 5])):
        for n in (0, 1, 2, 3, 7):
            cs.append(call('diff', [a], {'n': n}))
        for ax in range(-a.ndim, a.ndim):
            cs.append(call('diff', [a], {'axis': ax}))
        cs.append(call('diff', [a], {'prepend': 0, 'append': 5}))
        if a.ndim == 1:
            cs.append(call('diff', [a], {'prepend': np.array([1, 2]), 'n': 2}))
    cs += [call('diff', [np.array(3.0)]), call('diff', [F(3)], {'n': -1}), call('diff', [F(3, 2)], {'axis': 2}),
           call('diff', [F(3, 2)], {'prepend': F(3, 1), 'axis': 1}), call('diff', [F(3, 2)], {'append': F(2, 1)})]
    blocks['diff'] = cs
    cs = []
    for a in (F(6), I(2, 3), np.array([1.0]), np.zeros(0), I(4)):
        cs.append(call('ediff1d', [a]))
        cs.append(call('ediff1d', [a], {'to_begin': -99, 'to_end': [88, 77]}))
        cs.append(call('ediff1d', [a], {'to_end': np.array([1.5])}))
        cs.append(call('ediff1d', [a], {'to_begin': 2.5}))
    cs += [call('ediff1d', [B(4)]), call('ediff1d', [I(3)], {'to_begin': np.array([1.5])})]
    blocks['ediff1d'] = cs
    cs = []
    x_nonuni = np.cumsum(rng.uniform(0.2, 2.0, size=7))
    for f in (F(7), I(7), F(5, 6), F(3, 4, 5), np.array([1.0, 4.0, 9.0])):
        cs.append(call('gradient', [f]))
        for eo in (1, 2):
            cs.append(call('gradient', [f], {'edge_order': eo}))
            cs.append(call('gradient', [f, 0.5], {'edge_order': eo}))
            if f.ndim == 1 and f.shape[0] == 7:
                cs.append(call('gradient', [f, x_nonuni], {'edge_order': eo}))
                cs.append(call('gradient', [f, np.arange(7) * 2], {'edge_order': eo}))
        for ax in range(f.ndim):
            cs.append(call('gradient', [f], {'axis': ax}))
            cs.append(call('gradient', [f, 2], {'axis': ax, 'edge_order': 2}))
        if f.ndim == 2:
            cs.append(call('gradient', [f, 1.0, np.cumsum(rng.uniform(0.2, 2, size=f.shape[1]))]))
            cs.append(call('gradient', [f], {'axis': (1, 0)}))
    cs += [call('gradient', [F(2)], {'edge_order': 2}), call('gradient', [F(4)], {'edge_order': 3}), call('gradient', [B(4)]),
           call('gradient', [F(4), 1.0, 2.0]), call('gradient', [F(4), np.array([1.0, 2.0])]), call('gradient', [np.array(3.0)]),
           call('gradient', [F(2, 3), np.array([1.0, 2.0])])]
    blocks['gradient'] = cs
    cs = []
    for y in (F(9), I(6), F(4, 5), F(3, 4, 5), np.array([2.0])):
        for ax in range(-y.ndim, y.ndim):
            cs.append(call('trapezoid', [y], {'axis': ax}))
            cs.append(call('trapezoid', [y], {'axis': ax, 'dx': 0.25}))
            cs.append(call('trapezoid', [y], {'axis': ax, 'x': np.sort(F(y.shape[ax]))}))
        if y.ndim == 2:
            cs.append(call('trapezoid', [y], {'x': np.sort(F(*y.shape), axis=-1)}))
    cs += [call('trapezoid', [B(4)]), call('trapezoid', [F(4)], {'x': F(3)}), call('trapezoid', [F(4)], {'dx': 2})]
    blocks['trapezoid'] = cs
    cs = []
    for p in (np.cumsum(F(12, lo=-2, hi=2) * 2), F(3, 8), I(10, lo=-20, hi=20) * 3, np.array([0.0, np.pi, 2 * np.pi, 3 * np.pi])):
        cs.append(call('unwrap', [p]))
        cs.append(call('unwrap', [p], {'discont': 4.0}))
        cs.append(call('unwrap', [p], {'period': 5.0}))
        if p.ndim == 2:
            cs.append(call('unwrap', [p], {'axis': 0}))
    cs += [call('unwrap', [I(10, lo=-30, hi=30)], {'period': 7}), call('unwrap', [I(10, lo=-30, hi=30)], {'period': 8}),
           call('unwrap', [np.array([0, 4, 8, 12, 6])], {'period': 8})]
    blocks['unwrap'] = cs

    # ---- interp
    cs = []
    xp = np.sort(F(12))
    fp = F(12)
    for x in (F(20, lo=-12, hi=12), np.array(0.3), np.array([np.nan, xp[0], xp[-1], xp[5]]), F(3, 4), I(5)):
        cs.append(call('interp', [x, xp, fp]))
        cs.append(call('interp', [x, xp, fp], {'left': -100.0, 'right': 100.0}))
        cs.append(call('interp', [x, xp, fp], {'period': 7.5}))
    cs += [call('interp', [F(5), np.array([1.0]), np.array([3.0])]), call('interp', [F(5), np.zeros(0), np.zeros(0)]),
           call('interp', [F(5), xp, fp[:5]]), call('interp', [F(5), np.array([0.0, 1.0, 1.0, 2.0]), np.array([0.0, 1.0, 5.0, 6.0])]),
           call('interp', [F(5), xp, fp], {'period': 0}), call('interp', [F(200, lo=-12, hi=12), xp, fp]),
           call('interp', [np.array([0.5]), np.array([0.0, 1.0]), np.array([np.inf, np.inf])]),
           call('interp', [np.array([0.5, 1e308]), np.array([0.0, 1e308]), np.array([-1e308, 1e308])])]
    blocks['interp'] = cs

    # ---- sequences
    cs = []
    for st, sp in ((0, 2), (1.5, -3), (np.array([0.0, 1.0]), 3.0), (2.0, np.array([[1.0], [4.0]]))):
        for num in (0, 1, 5, 50):
            for ep in (True, False):
                cs.append(call('logspace', [st, sp], {'num': num, 'endpoint': ep}))
        cs.append(call('logspace', [st, sp], {'num': 4, 'base': 2.0}))
        cs.append(call('logspace', [st, sp], {'num': 4, 'dtype': 'int64'}))
    cs += [call('logspace', [np.array([0.0, 1.0]), 2.0], {'num': 3, 'axis': 1}), call('logspace', [0, 1], {'num': 3, 'base': np.array([2.0, 10.0])}),
           call('logspace', [0, 1], {'num': -1}), call('logspace', [np.array([0.0, 1.0]), 2.0], {'num': 3, 'axis': -1, 'base': np.array([2.0, 3.0])})]
    blocks['logspace'] = cs
    cs = []
    for st, sp in ((1, 1000), (1.0, 2.0), (-1, -1000), (2.0, 8.0), (np.array([1.0, 10.0]), 1000.0), (5.0, np.array([[1.0], [-2.0]]))):
        for num in (0, 1, 4, 7):
            for ep in (True, False):
                cs.append(call('geomspace', [st, sp], {'num': num, 'endpoint': ep}))
        cs.append(call('geomspace', [st, sp], {'num': 5, 'dtype': 'int64'}))
    cs += [call('geomspace', [0, 10]), call('geomspace', [1.0, 10.0], {'num': 3, 'axis': 0}),
           call('geomspace', [np.array([1.0, 2.0]), 16.0], {'num': 3, 'axis': 1}), call('geomspace', [1, -8], {'num': 4})]
    blocks['geomspace'] = cs
    blocks['vander'] = [call('vander', [x], kw) for x in (F(4), I(5, lo=-3, hi=4), B(3), np.array([2 ** 20, 3]), np.zeros(0))
                        for kw in ({}, {'N': 3}, {'increasing': True}, {'N': 0}, {'N': 6, 'increasing': True})] + \
        [call('vander', [F(2, 2)]), call('vander', [F(3)], {'N': -1})]

    # ---- polynomials
    ps = [F(4), I(3), np.array([0.0, 0.0, 2.0, -1.0]), np.array([5.0]), F(2, 3), np.zeros(0), B(3)]
    blocks['polyval'] = [call('polyval', [p, x]) for p in ps for x in (F(6), 2.5, I(2, 3), np.array(1.5), 3)] + \
        [call('polyval', [np.array(2.0), F(3)])]
    for fn in ('polyadd', 'polysub'):
        blocks[fn] = [call(fn, [a, b]) for a, b in ((F(3), F(5)), (I(4), I(2)), (F(3), F(3)), (np.array(2.0), F(3)), (F(2, 3), F(3)),
                                                     (B(2), B(3)), (F(2, 3), F(2, 3)), (I(2), F(4)))]
    cs = []
    for a, b in ((F(3), F(4)), (I(3), I(5)), (np.array([0.0, 0.0, 1.0, 2.0]), F(3)), (np.zeros(3), F(2)), (np.array(2.0), F(3)),
                 (F(2, 2), F(2)), (B(3), B(2)), (F(12), F(9))):
        t = None
        if a.dtype.kind == 'f' and b.dtype.kind == 'f' and a.ndim <= 1 and b.ndim <= 1:
            a1, b1 = np.trim_zeros(np.atleast_1d(a), 'f'), np.trim_zeros(np.atleast_1d(b), 'f')
            if a1.size and b1.size:
                s = np.convolve(np.abs(a1), np.abs(b1))
                r = np.polymul(a, b)
                n = min(a1.size, b1.size)
                g = n * EPS / (1 - n * EPS)
                with np.errstate(all='ignore'):
                    q = np.where(np.abs(r) > 0, 2 * g * s / np.abs(r), 0.0)
                rt = float(np.max(q))
                t = {'rtol': rt, 'why': 'convolution by dot products: summation order differs; bound 2 gamma_n sum|a_k b_k| / |result|'} if rt > 1e-12 else None
        cs.append(call('polymul', [a, b], tol=t))
    blocks['polymul'] = cs
    blocks['polyder'] = [call('polyder', [p], {'m': m}) for p in (F(5), I(4), np.array([3.0]), B(3), np.array([2 ** 62, 1, 1]))
                         for m in (0, 1, 2, 5, 2.7)] + [call('polyder', [F(3)], {'m': -1}), call('polyder', [np.array(3.0)])]
    blocks['polyint'] = [call('polyint', [p], kw) for p in (F(4), I(3), np.array([2.0]), B(2))
                         for kw in ({}, {'m': 2}, {'m': 3, 'k': 1.5}, {'m': 2, 'k': [1, 2]}, {'m': 0}, {'k': np.array([7.0])})] + \
        [call('polyint', [F(3)], {'m': -1}), call('polyint', [F(3)], {'m': 3, 'k': [1, 2]})]
    blocks['polydiv'] = [call('polydiv', [u, v]) for u, v in (
        (F(6), F(3)), (np.array([1.0, -3.0, 2.0]), np.array([1.0, -1.0])), (I(4), I(2, lo=1)), (F(2), F(4)), (np.array(3.0), np.array(2.0)),
        (np.array([1.0, 2.0, 1.0]), np.array([1.0, 1.0])), (F(5), np.array([0.0, 1.0])), (np.array([1e-10, 1.0]), np.array([1.0])))]

    # ---- windows, sinc, i0
    for fn in ('bartlett', 'blackman', 'hamming', 'hanning'):
        blocks[fn] = [call(fn, [m]) for m in (-3, 0, 1, 2, 3, 5, 12, 51, 5.5, True)]
    blocks['kaiser'] = [call('kaiser', [m, b]) for m in (0, 1, 2, 5, 12, 5.5) for b in (0.0, 5.0, 14.0, -2.0)]
    blocks['i0'] = [call('i0', [x]) for x in (F(30, lo=-30, hi=30), np.array([0.0, 8.0, -8.0, 7.999999, 8.000001, np.inf, np.nan]),
                                              I(5), B(3), np.array(2.5), F(2, 3, lo=0, hi=700))]
    blocks['sinc'] = [call('sinc', [x]) for x in (F(20), np.array([0.0, -0.0, 1.0, 0.5, 1e-300, np.inf, np.nan]), I(6), np.array(0.25), B(3))]

    # ---- nan_to_num and predicates
    xs = [np.array([np.nan, np.inf, -np.inf, 1.5, -0.0]), F(2, 3), I(4), B(3), np.array(np.nan), np.array(np.inf)]
    blocks['nan_to_num'] = [call('nan_to_num', [x], kw) for x in xs for kw in (
        {}, {'nan': -1.0}, {'posinf': 9.0, 'neginf': -9.0}, {'nan': 2, 'posinf': 1e300}, {'copy': False})]
    for fn in ('isposinf', 'isneginf', 'iscomplex', 'isreal', 'iscomplexobj', 'isrealobj', 'real_if_close'):
        blocks[fn] = [call(fn, [x]) for x in xs + [np.array(2.0), np.zeros((2, 0))]]

    # ---- pad
    cs = []
    arrs = [np.arange(1.0, 6.0), I(3, 4), F(2, 3, 2), B(3), np.array([7.0]), np.zeros((0, 3))]
    widths = [1, (2, 3), ((1, 2), (0, 3)), ((2,), (1,)), 0, 7, ((3, 0), (0, 0), (1, 1))]
    for a in arrs:
        for w in widths:
            for mode, kw in (('constant', {}), ('constant', {'constant_values': (-1, 9)}), ('edge', {}), ('wrap', {}),
                             ('reflect', {}), ('reflect', {'reflect_type': 'odd'}), ('symmetric', {}), ('symmetric', {'reflect_type': 'odd'}),
                             ('linear_ramp', {}), ('linear_ramp', {'end_values': (5, -2)}), ('maximum', {}), ('minimum', {}),
                             ('mean', {}), ('median', {}), ('mean', {'stat_length': 2}), ('median', {'stat_length': ((1, 3),)}),
                             ('empty', {})):
                cs.append(call('pad', [a, w], dict(kw, mode=mode), compare='shape' if mode == 'empty' else 'tol'))
    cs += [call('pad', [F(3), 1.5]), call('pad', [F(3), -1]), call('pad', [F(3), 1], {'mode': 'bogus'}),
           call('pad', [F(3), 1], {'mode': 'edge', 'constant_values': 1}), call('pad', [F(3), 1], {'mode': 'maximum', 'stat_length': 0}),
           call('pad', [np.zeros((0, 2)), 1], {'mode': 'edge'}), call('pad', [F(2, 2), ((1, 2), (3, 4), (5, 6))]),
           call('pad', [I(4), 3], {'mode': 'constant', 'constant_values': 2.7}), call('pad', [I(5), 2], {'mode': 'mean'}),
           call('pad', [np.array([1, 2, 3, 4]), 3], {'mode': 'median'}), call('pad', [np.array(5.0), 2]),
           call('pad', [F(3), [1, 2]], {'mode': 'wrap'}), call('pad', [F(2), 9], {'mode': 'reflect'}),
           call('pad', [F(2), 9], {'mode': 'symmetric', 'reflect_type': 'odd'})]
    blocks['pad'] = cs

    # ---- in-place writers
    cs = []
    for a in (I(6), F(2, 3), B(5)):
        for ind, v, mode in (([0, 2], [9, 8], 'raise'), ([1], [2.7], 'raise'), ([0, 1, 2], [5], 'raise'), ([-1, -6], [4, 3], 'raise'),
                             ([7, -8], [1, 2], 'wrap'), ([7, -8], [1, 2], 'clip'), ([10], [1], 'raise'), ([], [], 'raise'),
                             ([1, 2], [], 'raise'), (np.array([[0, 1], [2, 3]]), np.array([[5, 6], [7, 8]]), 'raise'), ([1.5], [3], 'raise')):
            cs.append(call('put', [a, ind, v], {'mode': mode}, compare='inplace'))
    cs.append(call('put', [F(3), np.array([1.0]), [2]], compare='inplace'))
    blocks['put'] = cs
    cs = []
    for fn in ('putmask', 'place'):
        cs = []
        for a in (I(6), F(2, 3), B(4)):
            m = rng.uniform(size=a.shape) < 0.5
            for v in ([10, 20], [7], 2.5, [], np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])):
                cs.append(call(fn, [a, m, v], compare='inplace'))
            cs.append(call(fn, [a, np.ones(a.shape, dtype=bool), [1, 2, 3]], compare='inplace'))
            cs.append(call(fn, [a, np.zeros(a.shape, dtype=bool), []], compare='inplace'))
            cs.append(call(fn, [a, np.array([True]), [1]], compare='inplace'))
        blocks[fn] = cs
    cs = []
    for dst in (I(2, 3), F(2, 3), B(3)):
        for src, kw in ((5, {}), (2.5, {}), (2.5, {'casting': 'unsafe'}), (True, {}), (F(2, 3) if dst.ndim == 2 else F(3), {}),
                        (I(3), {}), (I(3), {'casting': 'no'}), (F(3), {'where': np.array([True, False, True])}),
                        (7, {'where': np.array([False, True, True])}), (F(4), {}), (B(3), {'casting': 'safe'}),
                        (I(3), {'casting': 'safe'}), (np.array(1.5), {'casting': 'same_kind'})):
            cs.append(call('copyto', [dst, src], kw, compare='inplace'))
    blocks['copyto'] = cs
    blocks['fill_diagonal'] = [call('fill_diagonal', [a, v], kw, compare='inplace') for a in (np.zeros((3, 3)), np.zeros((5, 3)), np.zeros((3, 5), dtype=np.int64),
                                                                                          np.zeros((2, 2, 2)), np.zeros(3), np.zeros((2, 3, 2)))
                               for v, kw in ((7, {}), ([1, 2], {}), (2.9, {}), (5, {'wrap': True}), (np.array([4.0, 5.0, 6.0]), {'wrap': True}))]
    cs = []
    a = I(2, 3)
    for ax, idx, v in ((1, np.array([[0], [2]]), 99), (1, np.array([[0], [2]]), 9.9), (0, np.array([[1, 0, 1]]), np.array([[5, 6, 7]])),
                       (1, np.array([[0, 0], [1, 1]]), np.array([[1, 2], [3, 4]])), (-1, np.array([[3], [0]]), 1), (1, np.array([[-1], [-3]]), 4),
                       (None, np.array([0, 1]), 5), (1, np.array([[0.0], [1.0]]), 1), (1, np.array([0, 1]), 1), (2, np.array([[0], [1]]), 1)):
        cs.append(call('put_along_axis', [a, idx, v, ax], compare='inplace'))
    blocks['put_along_axis'] = cs

    # PAR-2.3 verification: Python floats and ints written into integer arrays (int() and range checks)
    pv = [2.7, -1.5, float('nan'), float('inf'), 1e10, 1e19, 300, -1, 2**40, [2.5, 1e10], [2.5, -3.5]]
    for fn in ('put', 'putmask', 'place', 'fill_diagonal', 'put_along_axis'):
        for dt in ('int64', 'bool', 'float64'):
            for v in pv:
                zero = np.zeros(4, dtype=dt) if fn in ('put', 'putmask', 'place') else np.zeros((2, 2), dtype=dt)
                if fn == 'put':
                    args = [zero, [0, 1], v]
                elif fn in ('putmask', 'place'):
                    args = [zero, np.array([True, False, True, False]), v]
                elif fn == 'fill_diagonal':
                    args = [zero, v]
                else:
                    args = [zero, np.array([[0], [1]]), v if not isinstance(v, list) else [[v[0]], [v[1]]], 1]
                blocks[fn].append(call(fn, args, compare='inplace'))
    blocks['tensordot'] += [call('tensordot', [F(2, 3), F(4)], {'axes': ([], [])}), call('tensordot', [F(2, 3), F(4)], {'axes': [[], []]})]
    blocks['interp'] += [call('interp', [F(3), np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]])]), call('interp', [1.0, 5.0, 5.0])]

    for fn in ('put', 'putmask', 'place'):
        zero = np.zeros(4, dtype=np.int64)
        extra = [[0, 1]] if fn == 'put' else [np.array([True, False, True, False])]
        blocks[fn] += [call(fn, [zero] + extra + [v], compare='inplace') for v in (-9.223372036854776e18, 9.223372036854776e18)]
    blocks['tensordot'] += [call('tensordot', [F(2, 3), F(3)], {'axes': ([5], [0])})]

    # cumulative_trapezoid (scipy.integrate): the running trapezoid integral, kept last among the rng-drawing
    # blocks so its draws do not shift the fixtures above.
    from scipy.integrate import cumulative_trapezoid as _ctrap
    cs = []
    for y in (F(9), I(6), F(4, 5), F(3, 4, 5)):
        for ax in range(-y.ndim, y.ndim):
            cs.append(call('cumulative_trapezoid', [y], {'axis': ax}, f=_ctrap))
            cs.append(call('cumulative_trapezoid', [y], {'axis': ax, 'dx': 0.25}, f=_ctrap))
            cs.append(call('cumulative_trapezoid', [y], {'axis': ax, 'initial': 0}, f=_ctrap))
            cs.append(call('cumulative_trapezoid', [y], {'axis': ax, 'x': np.sort(F(y.shape[ax]))}, f=_ctrap))
    cs += [call('cumulative_trapezoid', [F(4)], {'dx': 2}, f=_ctrap),
           call('cumulative_trapezoid', [F(5)], {'initial': 1.5}, f=_ctrap),
           call('cumulative_trapezoid', [B(6)], f=_ctrap)]
    blocks['cumulative_trapezoid'] = cs
    # simpson (scipy.integrate): both parities of N exercise the Cartwright correction for odd interval counts.
    from scipy.integrate import simpson as _simp
    cs = []
    for y in (F(7), F(8), F(4, 5), F(3, 6), F(2, 3, 4), I(9)):
        for ax in range(-y.ndim, y.ndim):
            cs.append(call('simpson', [y], {'axis': ax}, f=_simp))
            cs.append(call('simpson', [y], {'axis': ax, 'dx': 0.3}, f=_simp))
            cs.append(call('simpson', [y], {'axis': ax, 'x': np.sort(F(y.shape[ax]))}, f=_simp))
    cs += [call('simpson', [F(2)], f=_simp), call('simpson', [F(3)], f=_simp),
           call('simpson', [F(5)], {'x': np.sort(F(5))}, f=_simp),
           call('simpson', [F(6)], {'dx': 2.0}, f=_simp), call('simpson', [B(7)], f=_simp)]
    blocks['simpson'] = cs
    # cumulative_simpson (scipy.integrate): running Simpson integral; both N parities, uniform/irregular spacing,
    # and the `initial` (prepend + offset) path. 1-/2-point lanes fall back to the running trapezoid.
    from scipy.integrate import cumulative_simpson as _csimp
    cs = []
    for y in (F(7), F(8), F(4, 5), F(3, 6), F(2, 3, 4), I(9)):
        for ax in range(-y.ndim, y.ndim):
            cs.append(call('cumulative_simpson', [y], {'axis': ax}, f=_csimp))
            cs.append(call('cumulative_simpson', [y], {'axis': ax, 'dx': 0.3}, f=_csimp))
            cs.append(call('cumulative_simpson', [y], {'axis': ax, 'initial': 0.0}, f=_csimp))
            cs.append(call('cumulative_simpson', [y], {'axis': ax, 'x': np.sort(F(y.shape[ax]))}, f=_csimp))
    cs += [call('cumulative_simpson', [F(5)], {'initial': 1.5}, f=_csimp),
           call('cumulative_simpson', [F(6)], {'dx': 2.0}, f=_csimp),
           call('cumulative_simpson', [F(2)], {'initial': 0.0}, f=_csimp), call('cumulative_simpson', [B(7)], f=_csimp)]
    blocks['cumulative_simpson'] = cs
    # romb (scipy.integrate): Romberg integration of 2**k+1 equally-spaced samples
    from scipy.integrate import romb as _romb, newton_cotes as _nc
    cs = []
    for npow in (2, 3, 4, 5):
        y = F(2 ** npow + 1)
        cs.append(call('romb', [y], f=_romb))
        cs.append(call('romb', [y], {'dx': 0.4}, f=_romb))
    blocks['romb'] = cs
    # newton_cotes (scipy.integrate): equally-spaced quadrature weights (the weights array)
    blocks['newton_cotes'] = [call('newton_cotes', [n], f=lambda nn: _nc(nn)[0]) for n in range(1, 7)]

    # ---- completeness: spacing, modf, bitwise_and/or/xor, fromiter, can_cast (literal-only, host-independent:
    # these are integer/ulp/type-rule ops with no BLAS or transcendental loop, shared with the splice generator)
    import fixtures_misc_blocks
    blocks.update(fixtures_misc_blocks.build(call))

    out = {'module': 'np', 'numpy': np.__version__, 'env': fixture_env.env(),
           'calls': [{'fn': k, 'cases': v} for k, v in blocks.items()]}
    fixture_env.require_pinned()
    with open(OUT, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'np_numeric: {len(blocks)} functions, {sum(len(v) for v in blocks.values())} calls')


if __name__ == '__main__':
    main()
