#!/usr/bin/env python3
"""Generate NumPy reference fixtures for the kernel's numpy functions (np.* in the registry, ADR 0011).

    python3 tools/parity/fixtures_np.py

Writes tests/fixtures/parity/np.json: for every function, a list of calls (positional arguments, keyword
arguments) and what the pinned NumPy (tools/parity/requirements.txt) returns for them, or the exception it
raises. tools/parity/run-tests.php replays each call against Tessero\\Np (FFI) or Tessero\\Ext\\Np and compares.

Values are encoded as JSON:
  number / "nan" / "inf" / "-inf"           a float (or int) scalar
  true / false / null                        a bool / None
  {"s": "..."}                               a string
  {"shape": [...], "dtype": "...", "data": [...]}   an ndarray (flat, C order; dtype float64/int64/bool)
  {"list": [...]}                            a Python list or tuple of values (lexsort keys, range)
  {"dict": {"name": value, ...}}             a multi-output result, keyed by the registry's output names
  {"error": "ValueError"}                    the call raises
Each call carries a `compare` mode: "tol" (rtol/atol, NaN == NaN, the default), "sorted" (both sides sorted
first: NumPy leaves the order unspecified) or "partition" (for partition/argpartition: the kth elements and the
partition invariant, as NumPy does not fix the order).
"""
import json, math, os, warnings

import numpy as np
import fixture_env

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'fixtures', 'parity', 'np.json')
rng = np.random.default_rng(20260928)


def enc_scalar(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    if math.isnan(v):
        return 'nan'
    if math.isinf(v):
        return 'inf' if v > 0 else '-inf'
    return v


def enc(v):
    if v is None:
        return None
    if isinstance(v, str):
        return {'s': v}
    if isinstance(v, np.ndarray):
        if v.dtype.kind == 'c':                         # complex128: interleaved [re, im, ...] (as NDArray toList)
            data = []
            for x in v.ravel():
                data += [enc_scalar(x.real), enc_scalar(x.imag)]
            return {'shape': list(v.shape), 'dtype': 'complex128', 'data': data}
        kind = {'f': 'float64', 'i': 'int64', 'u': 'int64', 'b': 'bool'}[v.dtype.kind]
        return {'shape': list(v.shape), 'dtype': kind, 'data': [enc_scalar(x) for x in v.ravel()]}
    if isinstance(v, (list, tuple)):
        return {'list': [enc(x) for x in v]}
    return enc_scalar(v)


def enc_result(r, names):
    if isinstance(r, tuple):
        return {'dict': {n: enc_result_one(x) for n, x in zip(names, r)}}
    return enc_result_one(r)


def enc_result_one(r):
    r = np.asarray(r)
    if r.ndim == 0:
        return enc_scalar(r[()])
    return enc(r)


def call(fn, args, kwargs=None, names=None, compare='tol'):
    kwargs = kwargs or {}
    f = getattr(np, fn)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            r = f(*args, **kwargs)
            if hasattr(r, '_fields'):                  # the unique_* namedtuples
                r = tuple(r)
            expect = enc_result(r, names or [])
        except Exception as e:  # noqa: BLE001 - the exception class is the expectation
            expect = {'error': type(e).__name__}
    return {'args': [enc(a) for a in args], 'kwargs': {k: enc(v) for k, v in kwargs.items()}, 'expect': expect,
            'compare': compare}


def data(shape, nan=0.0, ints=False, lo=-10.0, hi=10.0, ties=False):
    if ints:
        return rng.integers(-5, 6, size=shape).astype(np.int64)
    if ties:
        a = rng.integers(0, 6, size=shape).astype(np.float64) / 2
    else:
        a = rng.uniform(lo, hi, size=shape)
    if nan:
        a = np.where(rng.uniform(size=shape) < nan, np.nan, a)
    return a


SHAPES = [(1,), (2,), (7,), (100,), (1001,), (3, 4), (5, 1), (2, 3, 4)]


def axes_for(shape):
    out = [None]
    for ax in range(len(shape)):
        out += [ax]
    if len(shape) > 1:
        out += [-1]
    if len(shape) == 3:
        out += [(0, 2)]
    return out


def reductions(fn, nan=0.0, extra=None, int_data=True):
    calls = []
    for shp in SHAPES:
        for ax in axes_for(shp):
            for kd in (False, True):
                kw = dict(extra or {})
                if ax is not None:
                    kw['axis'] = ax
                if kd:
                    kw['keepdims'] = True
                calls.append(call(fn, [data(shp, nan=nan)], kw))
    # ties, all-NaN slices, empty-ish and special values
    calls.append(call(fn, [data((9,), ties=True)], dict(extra or {})))
    calls.append(call(fn, [np.array([1.0, np.inf, -np.inf, 2.0])], dict(extra or {})))
    calls.append(call(fn, [np.array([np.nan, np.nan, 1.0, 2.0])], dict(extra or {})))
    calls.append(call(fn, [np.array([[np.nan, np.nan], [1.0, 3.0]])], dict(extra or {}, axis=1)))
    if int_data:
        calls.append(call(fn, [data((3, 5), ints=True)], dict(extra or {}, axis=0)))
    return calls


def main():
    blocks = {}
    for fn in ('median', 'ptp', 'count_nonzero'):
        blocks[fn] = reductions(fn)
    for fn in ('nanmedian', 'nansum', 'nanprod', 'nanmin', 'nanmax', 'nanmean', 'nanargmin', 'nanargmax',
               'nancumsum', 'nancumprod'):
        blocks[fn] = reductions(fn, nan=0.2)
    for fn in ('nanvar', 'nanstd'):
        blocks[fn] = reductions(fn, nan=0.2) + reductions(fn, nan=0.2, extra={'ddof': 1})[::5] \
            + [call(fn, [np.array([1.0, np.nan, 3.0])], {'ddof': 2}), call(fn, [np.array([1.0, np.nan, 3.0])], {'ddof': 5})]
    for fn in ('cumulative_sum', 'cumulative_prod'):
        cs = []
        for shp in [(1,), (7,), (3, 4), (2, 3, 4)]:
            for ax in ([None] if len(shp) == 1 else list(range(len(shp)))):
                for inc in (False, True):
                    kw = {'include_initial': inc}
                    if ax is not None:
                        kw['axis'] = ax
                    cs.append(call(fn, [data(shp, lo=0.5, hi=1.5)], kw))
        cs.append(call(fn, [data((3, 4))], {}))            # 2-D without axis raises
        blocks[fn] = cs

    methods = ['linear', 'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf',
               'hazen', 'weibull', 'median_unbiased', 'normal_unbiased', 'lower', 'higher', 'midpoint', 'nearest']
    for fn, scale, nan in (('quantile', 1.0, 0.0), ('percentile', 100.0, 0.0), ('nanquantile', 1.0, 0.2),
                           ('nanpercentile', 100.0, 0.2)):
        cs = []
        for m in methods:
            for shp in [(1,), (2,), (7,), (10,), (3, 4), (2, 3, 4)]:
                for q in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
                    cs.append(call(fn, [data(shp, nan=nan), q * scale], {'method': m}))
                cs.append(call(fn, [data(shp, nan=nan), np.array([0.0, 0.3, 0.5, 0.77, 1.0]) * scale], {'method': m}))
                if len(shp) > 1:
                    cs.append(call(fn, [data(shp, nan=nan), 0.4 * scale], {'method': m, 'axis': 0}))
                    cs.append(call(fn, [data(shp, nan=nan), np.array([0.2, 0.8]) * scale], {'method': m, 'axis': 1, 'keepdims': True}))
            cs.append(call(fn, [data((9,), ties=True), 0.35 * scale], {'method': m}))
        cs.append(call(fn, [data((5,)), 1.5 * scale], {}))   # q out of range raises
        blocks[fn] = cs

    cs = []
    for shp in [(1,), (7,), (3, 4), (2, 3, 4)]:
        for ax in axes_for(shp):
            kw = {} if ax is None else {'axis': ax}
            cs.append(call('average', [data(shp)], kw))
            w = rng.uniform(0.1, 2.0, size=shp if ax is None or isinstance(ax, tuple) else (shp[ax],))
            cs.append(call('average', [data(shp)], dict(kw, weights=w)))
    cs.append(call('average', [np.array([1.0, 2.0])], {'weights': np.array([1.0, -1.0])}))   # zero weight sum raises
    blocks['average'] = cs

    srt = np.sort(data((12,), ties=True))
    blocks['searchsorted'] = [call('searchsorted', [srt, data(s, ties=True)], {'side': side})
                              for s in [(1,), (9,), (3, 4)] for side in ('left', 'right')] + \
        [call('searchsorted', [srt, 1.0]), call('searchsorted', [np.array([1.0, 2.0, np.nan]), np.array([np.nan, 2.0, 5.0])])]
    bins_up = np.array([-5.0, -1.0, 0.0, 2.5, 7.0])
    blocks['digitize'] = [call('digitize', [data(s), b], {'right': r}) for s in [(1,), (20,), (3, 4)]
                          for b in (bins_up, bins_up[::-1].copy()) for r in (False, True)] + \
        [call('digitize', [np.array([0.0, 2.5, np.nan]), bins_up])]
    blocks['isin'] = [call('isin', [data(s, ints=True), np.array([0, 2, 3, -5])], {'invert': inv})
                      for s in [(1,), (15,), (3, 4)] for inv in (False, True)] + \
        [call('isin', [np.array([1.0, np.nan, 2.5]), np.array([np.nan, 2.5])])]
    cs = []
    for fn in ('partition', 'argpartition'):
        cs = []
        for shp in [(1,), (7,), (20,), (3, 5), (2, 3, 4)]:
            for kth in (0, shp[-1] // 2, shp[-1] - 1):
                cs.append(call(fn, [data(shp, ties=True), kth], compare='partition'))
            if len(shp) > 1:
                cs.append(call(fn, [data(shp), 1], {'axis': 0}, compare='partition'))
        cs.append(call(fn, [np.array([3.0, np.nan, 1.0, 2.0]), 1], compare='partition'))
        blocks[fn] = cs

    # routines
    U = ['values', 'indices', 'inverse', 'counts']
    cs = []
    for a in (data((20,), ties=True), data((3, 4), ints=True), np.array([2.0, np.nan, 1.0, np.nan, 2.0]), np.array([5.0])):
        for ri in (False, True):
            for rv in (False, True):
                for rc in (False, True):
                    kw = {'return_index': ri, 'return_inverse': rv, 'return_counts': rc}
                    names = ['values'] + [n for n, f in zip(U[1:], (ri, rv, rc)) if f]
                    cs.append(call('unique', [a], kw, names))
    cs.append(call('unique', [np.array([2.0, np.nan, np.nan])], {'equal_nan': False}, ['values']))
    blocks['unique'] = cs
    ua = [data((15,), ties=True), data((2, 5), ints=True), np.array([np.nan, 1.0, np.nan])]
    # NumPy >= 2.3 hashes integers in unique_values, so its order is unspecified: compare as sets
    blocks['unique_values'] = [call('unique_values', [a], compare='sorted') for a in ua]
    blocks['unique_counts'] = [call('unique_counts', [a], names=['values', 'counts']) for a in ua]
    blocks['unique_inverse'] = [call('unique_inverse', [a], names=['values', 'inverse_indices']) for a in ua]
    blocks['unique_all'] = [call('unique_all', [a], names=['values', 'indices', 'inverse_indices', 'counts']) for a in ua]
    pairs = [(data((12,), ties=True), data((9,), ties=True)), (data((3, 4), ints=True), data((5,), ints=True)),
             (np.array([1.0, 2.0]), np.array([3.0]))]
    blocks['intersect1d'] = [call('intersect1d', [a, b], {'return_indices': ri}, ['intersect1d', 'comm1', 'comm2'])
                             for a, b in pairs for ri in (False, True)] + \
        [call('intersect1d', [np.unique(a), np.unique(b)], {'assume_unique': True}) for a, b in pairs]
    blocks['union1d'] = [call('union1d', [a, b]) for a, b in pairs]
    blocks['setdiff1d'] = [call('setdiff1d', [a, b]) for a, b in pairs] + \
        [call('setdiff1d', [np.unique(a), np.unique(b)], {'assume_unique': True}) for a, b in pairs]
    blocks['setxor1d'] = [call('setxor1d', [a, b]) for a, b in pairs] + \
        [call('setxor1d', [np.unique(a), np.unique(b)], {'assume_unique': True}) for a, b in pairs]
    xs = [rng.integers(0, 8, size=30), np.array([0, 1, 1, 3]), np.array([2])]
    blocks['bincount'] = [call('bincount', [x]) for x in xs] + \
        [call('bincount', [x], {'weights': rng.uniform(size=len(x))}) for x in xs] + \
        [call('bincount', [x], {'minlength': 10}) for x in xs] + [call('bincount', [np.array([1, -1])])]
    hs = []
    for a in (data((50,)), data((200,), lo=0, hi=3), np.array([1.0, 1.0, 1.0]), data((7,), ties=True)):
        for bins in (1, 3, 10, 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt',
                     np.array([-10.0, -2.0, 0.0, 5.0, 10.0])):
            hs.append(call('histogram', [a], {'bins': bins}, ['hist', 'bin_edges']))
        hs.append(call('histogram', [a], {'bins': 4, 'range': (-1.0, 1.0)}, ['hist', 'bin_edges']))
        hs.append(call('histogram', [a], {'bins': 5, 'density': True}, ['hist', 'bin_edges']))
        hs.append(call('histogram', [a], {'bins': 5, 'weights': rng.uniform(size=a.shape)}, ['hist', 'bin_edges']))
    blocks['histogram'] = hs
    blocks['histogram_bin_edges'] = [call('histogram_bin_edges', [a], {'bins': b})
                                     for a in (data((50,)), data((9,), ties=True)) for b in (4, 'auto', 'fd', 'sturges')]
    cv = []
    for m in (data((3, 10)), data((2, 5)), data((10, 3)), data((6,))):
        cv.append(call('cov', [m]))
        cv.append(call('cov', [m], {'rowvar': False}))
        cv.append(call('cov', [m], {'bias': True}))
        cv.append(call('cov', [m], {'ddof': 0}))
    x2 = data((10,))
    cv += [call('cov', [x2, data((10,))]), call('cov', [x2], {'fweights': rng.integers(1, 4, size=10)}),
           call('cov', [x2], {'aweights': rng.uniform(0.1, 1, size=10)})]
    blocks['cov'] = cv
    blocks['corrcoef'] = [call('corrcoef', [m]) for m in (data((3, 10)), data((2, 5)))] + \
        [call('corrcoef', [data((10, 3))], {'rowvar': False}), call('corrcoef', [x2, data((10,))])]
    cc = []
    for fn in ('correlate', 'convolve'):
        cc = []
        for n, m in ((5, 3), (3, 5), (1, 1), (10, 4), (4, 4)):
            for mode in ('full', 'same', 'valid'):
                cc.append(call(fn, [data((n,)), data((m,))], {'mode': mode}))
        blocks[fn] = cc
    blocks['lexsort'] = [call('lexsort', [[data((10,), ties=True), data((10,), ties=True)]]),
                         call('lexsort', [[data((8,), ties=True)]]),
                         call('lexsort', [[data((6,), ties=True), data((6,), ties=True), data((6,), ties=True)]])]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # ---- integer semantics (exact int64 arithmetic with NumPy's wrap-around) and input validation: found by the
    # independent verification and the sanitizer fuzz pass; recorded here so they cannot regress
    M = np.iinfo(np.int64).max
    big = np.array([2 ** 53 + 1, 0, -5, 2 ** 60 + 7], dtype=np.int64)
    ext = np.array([M, 0, -M, 1], dtype=np.int64)
    blocks['ptp'] += [call('ptp', [big]), call('ptp', [ext]), call('ptp', [np.array([M, -M], dtype=np.int64)]),
                      call('ptp', [np.array([[M, 0], [1, -M]], dtype=np.int64)], {'axis': 0}),
                      call('ptp', [np.array([True, False])])]
    for fn in ('nansum', 'nanprod', 'nanmin', 'nanmax', 'nancumsum', 'nancumprod'):
        blocks[fn] += [call(fn, [big]), call(fn, [ext]), call(fn, [np.array([[M, 1], [2, 3]], dtype=np.int64)], {'axis': 1})]
    for fn in ('cumulative_sum', 'cumulative_prod'):
        blocks[fn] += [call(fn, [ext]), call(fn, [big], {'include_initial': True}),
                       call(fn, [np.array([1.0, 2.0, 3.0])], {'include_initial': 2})]
    blocks['partition'] += [call('partition', [big, 1], compare='partition'), call('partition', [ext, 2], compare='partition')]
    for fn in ('correlate', 'convolve'):
        blocks[fn] += [call(fn, [np.array([1073741825], dtype=np.int64), np.array([1073741827], dtype=np.int64)]),
                       call(fn, [np.array([M, 2], dtype=np.int64), np.array([2, 1], dtype=np.int64)], {'mode': 'full'}),
                       call(fn, [np.array([3, -4, 5, 2 ** 40], dtype=np.int64), np.array([2 ** 30, 7], dtype=np.int64)], {'mode': 'same'}),
                       call(fn, [np.array([1, 2, 3], dtype=np.int64), np.array([4, 5], dtype=np.int64)])]
    a22 = np.array([[1.0, 2.0], [3.0, 4.0]])
    blocks['average'] += [call('average', [a22], {'weights': np.array([[1.0, 2.0]]), 'axis': 0}),
                          call('average', [a22], {'weights': np.array([1.0, 2.0, 3.0, 4.0]), 'axis': (0, 1)}),
                          call('average', [a22], {'weights': np.array([1.0, 2.0]), 'axis': 1}),
                          call('average', [a22], {'weights': np.array([1.0, 2.0, 3.0]), 'axis': 0}),
                          call('average', [a22], {'weights': np.array([1.0, 2.0])})]
    blocks['bincount'] += [call('bincount', [np.array([1.0, 2.0, 2.0])]), call('bincount', [np.array([0, 1, 2, 1])], {'minlength': 2.5}),
                           call('bincount', [np.array([0, 1, 2, 1])], {'minlength': 6}), call('bincount', [np.array([3, 3, 0])])]

    # PAR-2.3 verification: exact int64 above 2^53, Python scalars and lists, method-dependent dtypes
    big = np.array([2**62 + 1, 2**62, 2**62 + 1, 5, -(2**61) - 3, 2**62], dtype=np.int64)
    big2 = np.array([2**62, 7, 2**62 + 3, -(2**61) - 3], dtype=np.int64)
    blocks['unique'] += [call('unique', [big], {'return_index': True, 'return_inverse': True, 'return_counts': True},
                              ['values', 'indices', 'inverse', 'counts']),
                         call('unique', [5]), call('unique', [2.5]), call('unique', [True])]
    blocks['unique_counts'] += [call('unique_counts', [big], names=['values', 'counts'])]
    blocks['intersect1d'] += [call('intersect1d', [big, big2], {'return_indices': True}, ['intersect1d', 'comm1', 'comm2']),
                              call('intersect1d', [5, [5, 6]])]
    blocks['union1d'] += [call('union1d', [big, big2])]
    blocks['setdiff1d'] += [call('setdiff1d', [big, big2]), call('setdiff1d', [np.unique(big), np.unique(big2)], {'assume_unique': True})]
    blocks['setxor1d'] += [call('setxor1d', [big, big2])]
    blocks['bincount'] += [call('bincount', [[1.5, 2.2]]), call('bincount', [[-0.5, 1.0]]), call('bincount', [[-1.5]]),
                           call('bincount', [[float('nan')]]), call('bincount', [[float('inf')]]), call('bincount', [[1e30]]),
                           call('bincount', [[True, False, True]])]
    blocks['searchsorted'] += [call('searchsorted', [np.array([[1.0, 2.0], [3.0, 4.0]]), 3.0]), call('searchsorted', [5.0, 3.0])]
    qm = ['inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull',
          'linear', 'median_unbiased', 'normal_unbiased', 'lower', 'higher', 'midpoint', 'nearest']
    qi = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64)
    qb = np.array([True, False, True, True])
    qbig = np.array([2**60 + 3, 2**60 + 1, 2**60 + 2, 7], dtype=np.int64)
    for fn, q in (('quantile', np.array([0.0, 0.4, 0.5, 1.0])), ('percentile', np.array([0.0, 40.0, 100.0])),
                  ('nanquantile', 0.4), ('nanpercentile', 40.0)):
        blocks[fn] += [call(fn, [a, q], {'method': m}) for a in (qi, qb) for m in qm]
        blocks[fn] += [call(fn, [qbig, q], {'method': m}) for m in ('lower', 'higher', 'nearest', 'inverted_cdf', 'closest_observation')]
        blocks[fn] += [call(fn, [qi.reshape(2, 4), q], {'method': 'lower', 'axis': 1})]

    # second verification pass: promotion of bool with int64, int64 against float, empty samples, minlength
    bb = np.array([True, False])
    blocks['union1d'] += [call('union1d', [bb, big]), call('union1d', [True, 1])]
    blocks['intersect1d'] += [call('intersect1d', [True, 1]), call('intersect1d', [bb, big2])]
    blocks['setxor1d'] += [call('setxor1d', [bb, big])]
    blocks['setdiff1d'] += [call('setdiff1d', [np.array([2**63 - 1, 2**53 + 1, 5], dtype=np.int64), np.array([1.5, 5.0])]),
                            call('setdiff1d', [np.array([2**63 - 1, 2**53 + 1, 5], dtype=np.int64), np.array([5.0])], {'assume_unique': True})]
    for fn, q in (('nanquantile', 0.4), ('nanpercentile', 40.0)):
        blocks[fn] += [call(fn, [np.array([], dtype=np.int64), q], {'method': m}) for m in ('lower', 'linear', 'nearest')]
        blocks[fn] += [call(fn, [np.array([], dtype=bool), q], {'method': m}) for m in ('lower', 'linear')]
    blocks['bincount'] += [call('bincount', [np.array([0, 1])], {'minlength': 2.0}), call('bincount', [np.array([], dtype=np.float64)]),
                           call('bincount', [np.array([], dtype=np.int64)], {'weights': np.array([])})]

    # PAR-4.1: round, isclose, allclose as kernel registry routines (both backends)
    rblk = []
    for shp in [(1,), (7,), (3, 4), (2, 3, 4)]:
        for dec in (0, 1, 2, -1):
            rblk.append(call('round', [data(shp), dec]))
    rblk += [call('round', [data((3, 5), ints=True)]), call('round', [data((3, 5), ints=True), 0]),
             call('round', [np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5])]),
             call('round', [np.array([0.5, 1.5, 2.5, -0.5, -1.5]), 0]),
             call('round', [np.array([1.23456, -1.23456, 123.456, -987.654]), 2]),
             call('round', [np.array([12.0, 15.0, 25.0, -35.0]), -1]),
             call('round', [np.array([np.nan, np.inf, -np.inf, 1.5, -2.5])]),
             call('round', [1.5]), call('round', [2.675, 2])]
    blocks['round'] = rblk

    a7 = data((7,))
    blocks['isclose'] = [
        call('isclose', [a7, a7]), call('isclose', [a7, a7 + 1e-9]), call('isclose', [a7, a7 + 1.0]),
        call('isclose', [data((3, 4)), data((3, 4))]),
        call('isclose', [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.00000001, 3.1])]),
        call('isclose', [np.array([1e10, 1e-8]), np.array([1.00001e10, 1e-9])]),
        call('isclose', [np.array([np.inf, -np.inf, np.nan, 1.0]), np.array([np.inf, -np.inf, np.nan, 1.0])]),
        call('isclose', [np.array([np.inf, np.nan, 1.0]), np.array([-np.inf, np.nan, 1.0])], {'equal_nan': True}),
        call('isclose', [np.array([1.0, 2.0]), np.array([1.1, 2.1])], {'rtol': 0.2}),
        call('isclose', [np.array([1.0, 2.0]), np.array([1.05, 2.05])], {'atol': 0.1}),
        call('isclose', [data((3, 1)), data((1, 4))]),
        call('isclose', [np.array([1.0, 2.0, 3.0]), 2.0]),
        call('isclose', [np.array([1, 2, 3]), np.array([1, 2, 4])]),
        call('isclose', [1.0, 1.0 + 1e-9])]

    blocks['allclose'] = [
        call('allclose', [a7, a7]), call('allclose', [a7, a7 + 1e-9]), call('allclose', [a7, a7 + 1.0]),
        call('allclose', [data((3, 4)), data((3, 4))]),
        call('allclose', [np.array([1.0, 2.0]), np.array([1.0, 2.0000001])]),
        call('allclose', [np.array([np.nan]), np.array([np.nan])]),
        call('allclose', [np.array([np.nan]), np.array([np.nan])], {'equal_nan': True}),
        call('allclose', [np.array([np.inf]), np.array([np.inf])]),
        call('allclose', [np.array([np.inf]), np.array([-np.inf])]),
        call('allclose', [np.array([1.0, 2.0]), np.array([1.1, 2.1])], {'rtol': 0.2}),
        call('allclose', [data((3, 1)), data((1, 4))])]

    # result_type / promote_types: return a dtype, encoded as its name string (compared as a string)
    def dtcall(fn, cargs):
        try:
            expect = {'s': np.dtype(getattr(np, fn)(*cargs)).name}
        except Exception as e:  # noqa: BLE001
            expect = {'error': type(e).__name__}
        return {'args': [enc(a) for a in cargs], 'kwargs': {}, 'expect': expect, 'compare': 'tol'}
    DT = ['float64', 'float32', 'int64', 'int32', 'uint8', 'bool', 'complex128']
    blocks['promote_types'] = [dtcall('promote_types', [a, b]) for a in DT for b in DT]
    rt = [dtcall('result_type', [a, b]) for a in DT for b in DT]
    rt += [dtcall('result_type', [data((3,), ints=True), 'float32']),
           dtcall('result_type', [data((2,)), data((2,), ints=True)]),
           dtcall('result_type', ['int64']), dtcall('result_type', ['float32', 'float32', 'int64']),
           dtcall('result_type', [data((2,)), 'complex128'])]
    blocks['result_type'] = rt

    # finfo / iinfo: machine limits of a dtype, returned as a dict of fields (same key order as the registry)
    def finfo_case(name):
        f = np.finfo(name)
        d = {'eps': float(f.eps), 'epsneg': float(f.epsneg), 'max': float(f.max), 'min': float(f.min),
             'tiny': float(f.tiny), 'smallest_normal': float(f.smallest_normal), 'resolution': float(f.resolution),
             'precision': int(f.precision), 'bits': int(f.bits), 'nmant': int(f.nmant), 'nexp': int(f.nexp),
             'maxexp': int(f.maxexp), 'minexp': int(f.minexp)}
        return {'args': [enc(name)], 'kwargs': {}, 'expect': {'dict': d}, 'compare': 'tol'}
    blocks['finfo'] = [finfo_case(n) for n in ['float64', 'float32', 'complex128']]

    def iinfo_case(name):
        f = np.iinfo(name)
        return {'args': [enc(name)], 'kwargs': {},
                'expect': {'dict': {'min': int(f.min), 'max': int(f.max), 'bits': int(f.bits)}}, 'compare': 'tol'}
    blocks['iinfo'] = [iinfo_case(n) for n in ['int64', 'int32', 'uint8']]

    # einsum: Einstein summation over float64 operands (matmul, trace, diagonal, outer, batch, ellipsis, ...)
    blocks['einsum'] = [
        call('einsum', ['ij,jk->ik', data((3, 4)), data((4, 2))]),
        call('einsum', ['ij->ji', data((3, 4))]),
        call('einsum', ['ii->i', data((4, 4))]),
        call('einsum', ['ii->', data((5, 5))]),
        call('einsum', ['ij->', data((3, 4))]),
        call('einsum', ['ij->i', data((3, 4))]),
        call('einsum', ['ij->j', data((3, 4))]),
        call('einsum', ['i,i->', data((6,)), data((6,))]),
        call('einsum', ['i,j->ij', data((3,)), data((4,))]),
        call('einsum', ['ij,ij->ij', data((3, 4)), data((3, 4))]),
        call('einsum', ['ij,ij->', data((3, 4)), data((3, 4))]),
        call('einsum', ['bij,bjk->bik', data((2, 3, 4)), data((2, 4, 5))]),
        call('einsum', ['ij,jk,kl->il', data((2, 3)), data((3, 4)), data((4, 2))]),
        call('einsum', ['i,ij,j->', data((3,)), data((3, 4)), data((4,))]),
        call('einsum', ['...ij->...ji', data((2, 3, 4))]),
        call('einsum', ['...ij,...jk->...ik', data((2, 3, 4)), data((2, 4, 5))]),
        call('einsum', ['...ij,...jk->...ik', data((1, 3, 4)), data((2, 4, 5))]),  # broadcast batch
        call('einsum', ['ij,jk', data((3, 4)), data((4, 2))]),                      # implicit output 'ik'
        call('einsum', ['ii', data((4, 4))]),                                       # implicit trace (scalar)
        call('einsum', ['ij', data((3, 4))]),                                       # implicit identity
    ]

    # around: an alias of round; fabs: element-wise |x| as float64
    blocks['around'] = [
        call('around', [data((3, 4))]),
        call('around', [data((2, 3)) * 7.321, 2]),
        call('around', [data((5,)), 1]),
        call('around', [data((2, 2)) * 100.0, -1]),
    ]
    blocks['fabs'] = [
        call('fabs', [data((3, 4))]),
        call('fabs', [data((5,), lo=-100.0, hi=100.0)]),
        call('fabs', [np.array([-1.5, 2.5, -0.0, np.inf, -np.inf, np.nan])]),
        call('fabs', [data((2, 3), ints=True)]),
    ]

    # polynomials: polyfit (least-squares), polyval (Horner), roots (companion eig, sorted complex)
    def pcase(args, expectval, rtol, atol=0.0):
        return {'args': [enc(a) for a in args], 'kwargs': {}, 'expect': enc_result(expectval, []), 'compare': 'tol', 'tol': {'rtol': rtol, 'atol': atol}}
    xf = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    blocks['polyfit'] = [
        pcase([xf, np.polyval([1.0, 0.0, 1.0], xf), 2], np.polyfit(xf, np.polyval([1.0, 0.0, 1.0], xf), 2), 1e-6),
        pcase([xf, np.polyval([2.0, -3.0], xf), 1], np.polyfit(xf, np.polyval([2.0, -3.0], xf), 1), 1e-6),
        pcase([xf, np.polyval([1.0, -2.0, 0.5, 3.0], xf), 3], np.polyfit(xf, np.polyval([1.0, -2.0, 0.5, 3.0], xf), 3), 1e-5),
    ]
    blocks['roots'] = [
        pcase([np.array([1.0, -3.0, 2.0])], np.sort_complex(np.roots([1.0, -3.0, 2.0])), 1e-6),
        pcase([np.array([1.0, 0.0, 1.0])], np.sort_complex(np.roots([1.0, 0.0, 1.0])), 1e-6),
        pcase([np.array([1.0, -6.0, 11.0, -6.0])], np.sort_complex(np.roots([1.0, -6.0, 11.0, -6.0])), 1e-6),
        pcase([np.array([1.0, 0.0, 0.0, -1.0])], np.sort_complex(np.roots([1.0, 0.0, 0.0, -1.0])), 1e-6),
    ]

    out = {'module': 'np', 'numpy': np.__version__, 'env': fixture_env.env(), 'calls': [{'fn': k, 'cases': v} for k, v in blocks.items()]}
    fixture_env.require_pinned()
    with open(OUT, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'np: {len(blocks)} functions, {sum(len(v) for v in blocks.values())} calls')


if __name__ == '__main__':
    main()
