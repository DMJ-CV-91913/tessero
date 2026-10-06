#!/usr/bin/env python3
"""scipy.stats descriptive statistics fixtures (group "desc"): tests/fixtures/parity/stats_desc.json.

    (cd tools/parity && python3 fixtures_stats_desc.py)

Each call is recorded against the pinned SciPy with fixtures_stats_fn.call(). Registry conventions that differ
from SciPy's Python signature are applied here, where the call is recorded:
  - gufunc parameters after `axis` are passed as keywords (the façade puts axis right after the arrays);
  - describe's `minmax` tuple is recorded as one array stacking (min, max) (a routine returns arrays, not tuples);
  - trimboth / trim1 record SciPy's result sorted along the axis (np.partition leaves the order of the kept
    values unspecified; Tessero returns them sorted);
  - the tuple defaults of tmean/tvar/tstd/tsem (inclusive) and iqr (rng) are None in the registry and mean
    SciPy's defaults; explicit tuples are passed as lists.
"""
import warnings

import numpy as np
import scipy.stats as st

import fixtures_np as F
from fixtures_stats_fn import call, data, write


def call_describe(args, kwargs=None):
    """describe with minmax stacked into one array (see the module docstring)."""
    kwargs = kwargs or {}
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            r = st.describe(*args, **kwargs)
            vals = {'nobs': r.nobs, 'minmax': np.stack([np.asarray(r.minmax[0]), np.asarray(r.minmax[1])]),
                    'mean': r.mean, 'variance': r.variance, 'skewness': r.skewness, 'kurtosis': r.kurtosis}
            expect = {'dict': {k: F.enc_result_one(np.ma.getdata(v) if isinstance(v, np.ma.MaskedArray) else v)
                               for k, v in vals.items()}}
        except Exception as e:  # noqa: BLE001
            expect = {'error': type(e).__name__}
    return {'args': [F.enc(a) for a in args], 'kwargs': {k: F.enc(v) for k, v in kwargs.items()},
            'expect': expect, 'compare': 'tol'}


def call_trim(fn, args, kwargs=None):
    """trimboth / trim1: np.partition leaves the order of the kept values unspecified; Tessero returns them
    sorted, so the expectation is SciPy's result sorted along the axis (same values per lane)."""
    kwargs = kwargs or {}
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            r = np.asarray(getattr(st, fn)(*args, **kwargs))
            ax = kwargs.get('axis', 0)
            r = np.sort(r, axis=-1 if ax is None else ax) if r.size else r
            expect = F.enc_result_one(r)
        except Exception as e:  # noqa: BLE001
            expect = {'error': type(e).__name__}
    return {'args': [F.enc(a) for a in args], 'kwargs': {k: F.enc(v) for k, v in kwargs.items()},
            'expect': expect, 'compare': 'tol'}


def battery(fn, kw=None, axis_default=0, keep=True, nan=True, ints=True, small=True, empty=True, names=None, pos=()):
    """The standard cases of a reduction: 1-D, 2-D along each axis and None, keepdims, integers, NaNs, tiny samples."""
    kw = dict(kw or {})
    pos = list(pos)
    c = []
    c.append(call(fn, [data(12)] + pos, dict(kw), names=names))
    x = data((5, 7))
    for ax in (0, 1, -1, None):
        c.append(call(fn, [x] + pos, dict(kw, axis=ax), names=names))
    c.append(call(fn, [data((3, 4, 5))] + pos, dict(kw, axis=1), names=names))
    if keep:
        c.append(call(fn, [x] + pos, dict(kw, axis=1, keepdims=True), names=names))
        c.append(call(fn, [x] + pos, dict(kw, axis=None, keepdims=True), names=names))
    if ints:
        c.append(call(fn, [data(9, ints=True)] + pos, dict(kw), names=names))
        c.append(call(fn, [data((4, 6), ints=True)] + pos, dict(kw, axis=1), names=names))
    c.append(call(fn, [data((6, 5), ties=True)] + pos, dict(kw, axis=0), names=names))
    big = data((300, 3))
    c.append(call(fn, [big[:, 0].copy()] + pos, dict(kw), names=names))
    c.append(call(fn, [big] + pos, dict(kw, axis=0), names=names))
    c.append(call(fn, [big.T.copy()] + pos, dict(kw, axis=1), names=names))
    if nan:
        xn = data((4, 11), nan=0.2)
        xn[0, 3] = np.nan
        for pol in ('propagate', 'omit', 'raise'):
            c.append(call(fn, [xn] + pos, dict(kw, axis=1, nan_policy=pol), names=names))
            c.append(call(fn, [xn[0]] + pos, dict(kw, nan_policy=pol), names=names))
        c.append(call(fn, [xn] + pos, dict(kw, axis=0, nan_policy='omit'), names=names))
    if small:
        for n in (1, 2, 3, 4):
            c.append(call(fn, [data(n)] + pos, dict(kw), names=names))
    if empty:
        c.append(call(fn, [np.array([])] + pos, dict(kw), names=names))
    return c


def main():
    B = {}
    pos_data = data(15, lo=0.5, hi=20)
    posm = data((4, 6), lo=0.1, hi=9)

    # ---- generalised means
    for fn in ('gmean', 'hmean'):
        c = battery(fn, nan=False)
        c = [x for x in c]
        c += [call(fn, [pos_data]), call(fn, [posm], {'axis': 1}), call(fn, [posm], {'axis': None, 'keepdims': True})]
        w = data(15, lo=0.1, hi=3)
        c.append(call(fn, [pos_data], {'weights': w}))
        c.append(call(fn, [posm], {'axis': 1, 'weights': data((4, 6), lo=0, hi=2)}))
        c.append(call(fn, [posm], {'axis': 0, 'weights': data((4, 6), lo=0, hi=2)}))
        pn = pos_data.copy(); pn[[2, 7]] = np.nan
        wn = w.copy(); wn[4] = np.nan
        for pol in ('propagate', 'omit', 'raise'):
            c.append(call(fn, [pn], {'nan_policy': pol}))
            c.append(call(fn, [pn], {'weights': wn, 'nan_policy': pol}))
        c.append(call(fn, [np.array([1.0, 0.0, 3.0])]))
        c.append(call(fn, [np.array([2, 8, 4])]))
        c.append(call(fn, [np.array([])]))
        B[fn] = c
    c = []
    for p in (2, 3, -1, 0.5, 1, 0, -2.5, 1.5):
        c.append(call('pmean', [pos_data, p]))
    c.append(call('pmean', [posm, 2], {'axis': 1}))
    c.append(call('pmean', [posm, 3], {'axis': 0, 'keepdims': True}))
    c.append(call('pmean', [posm, -1], {'axis': None}))
    c.append(call('pmean', [pos_data, 2], {'weights': data(15, lo=0.1, hi=3)}))
    c.append(call('pmean', [data(9, ints=True), 2]))
    c.append(call('pmean', [np.array([1.0, -2.0, 3.0]), 2]))
    c.append(call('pmean', [pos_data, np.inf]))
    pn = pos_data.copy(); pn[3] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('pmean', [pn, 2], {'nan_policy': pol}))
    B['pmean'] = c

    # ---- moments
    B['skew'] = battery('skew') + battery('skew', {'bias': False}, small=True, nan=False, empty=False) + [
        call('skew', [np.ones(5)]), call('skew', [np.full(4, 3.3)], {'bias': False})]
    B['kurtosis'] = (battery('kurtosis') + battery('kurtosis', {'bias': False}, nan=False, empty=False)
                     + battery('kurtosis', {'fisher': False}, nan=False, empty=False, small=False)
                     + [call('kurtosis', [np.ones(6)]), call('kurtosis', [data(30)], {'fisher': False, 'bias': False})])
    c = []
    for o in (0, 1, 2, 3, 4, 5, 6, 7):
        c.append(call('moment', [data(13), o]))
    c += battery('moment', {'order': 3})
    c.append(call('moment', [data(10)], {'order': [1, 2, 3]}))
    c.append(call('moment', [data((5, 6))], {'order': [0, 2, 4], 'axis': 1}))
    c.append(call('moment', [data(10)], {'order': 2, 'center': 1.5}))
    c.append(call('moment', [data(10)], {'order': 1, 'center': 0.25}))
    c.append(call('moment', [data(10)], {'order': [1, 3], 'center': -1.0}))
    c.append(call('moment', [data(10)], {'order': 2.5}))
    c.append(call('moment', [data((4, 7))], {'order': [2, 3], 'axis': 1, 'keepdims': True}))
    c.append(call('moment', [data((4, 7))], {'order': [1, 2], 'axis': None}))
    B['moment'] = c
    B['variation'] = (battery('variation') + battery('variation', {'ddof': 1}, nan=False, empty=False)
                      + [call('variation', [np.array([1.0, 2.0])], {'ddof': 2}),
                         call('variation', [np.array([3.0, 3.0])], {'ddof': 2}),
                         call('variation', [np.array([-1.0, 2.0, -4.0])]),
                         call('variation', [np.array([1.0, -1.0])])])
    B['sem'] = battery('sem') + battery('sem', {'ddof': 0}, nan=False, empty=False, small=False) + [
        call('sem', [data(6)], {'ddof': 2})]

    # ---- z-scores and gstd
    def zcases(fn, pre=lambda x: x):
        c = [call(fn, [pre(data(12))])]
        x = pre(data((5, 7)))
        for ax in (0, 1, None):
            c.append(call(fn, [x], {'axis': ax}))
            c.append(call(fn, [x], {'axis': ax, 'ddof': 1}))
        c.append(call(fn, [pre(data((3, 4, 5)))], {'axis': 1}))
        xn = pre(data((4, 9)))
        xn[0, 2] = np.nan; xn[2, 5] = np.nan
        for pol in ('propagate', 'omit', 'raise'):
            c.append(call(fn, [xn], {'axis': 1, 'nan_policy': pol}))
            c.append(call(fn, [xn], {'axis': 0, 'nan_policy': pol, 'ddof': 1}))
        c.append(call(fn, [np.full(4, 2.5)]))
        c.append(call(fn, [np.array([1.0])]))
        return c
    B['zscore'] = zcases('zscore') + [call('zscore', [data(8, ints=True)])]
    B['gzscore'] = zcases('gzscore', lambda x: np.abs(x) + 0.1)
    c = [call('zmap', [data(6), data(10)]), call('zmap', [data((4, 5)), data((7, 5))], {'axis': 0}),
         call('zmap', [data((4, 5)), data((4, 9))], {'axis': 1, 'ddof': 1}),
         call('zmap', [data((3, 4)), data(8)], {'axis': 0}), call('zmap', [data((3, 4)), data((2, 6))], {'axis': None})]
    cn = data(10); cn[3] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('zmap', [data(5), cn], {'nan_policy': pol}))
    c.append(call('zmap', [np.array([1.0, np.nan]), data(6)]))
    c.append(call('zmap', [data(4), np.full(5, 1.0)]))
    B['zmap'] = c
    c = [call('gstd', [pos_data]), call('gstd', [posm], {'axis': 1}), call('gstd', [posm], {'axis': 0, 'ddof': 0}),
         call('gstd', [posm], {'axis': None}), call('gstd', [posm], {'axis': 1, 'keepdims': True}),
         call('gstd', [np.array([2, 4, 8, 3])]), call('gstd', [np.array([1.5])]), call('gstd', [np.array([1.0, -2.0, 3.0])])]
    pn = posm.copy(); pn[1, 2] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('gstd', [pn], {'axis': 1, 'nan_policy': pol}))
    B['gstd'] = c

    # ---- mode
    c = [call('mode', [data(12, ties=True)], names=['mode', 'count']),
         call('mode', [data((6, 5), ties=True)], {'axis': 0}, names=['mode', 'count']),
         call('mode', [data((6, 5), ties=True)], {'axis': 1, 'keepdims': True}, names=['mode', 'count']),
         call('mode', [data((6, 5), ties=True)], {'axis': None}, names=['mode', 'count']),
         call('mode', [data(15, ints=True)], names=['mode', 'count']),
         call('mode', [data((5, 7), ints=True)], {'axis': 1}, names=['mode', 'count']),
         call('mode', [data(5)], names=['mode', 'count'])]
    xn = np.array([1.0, np.nan, 2.0, np.nan, 2.0, np.nan, 3.0])
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('mode', [xn], {'nan_policy': pol}, names=['mode', 'count']))
    xn2 = data((4, 8), ties=True); xn2[0, :3] = np.nan; xn2[2, 1] = np.nan
    for pol in ('propagate', 'omit'):
        c.append(call('mode', [xn2], {'axis': 1, 'nan_policy': pol}, names=['mode', 'count']))
    M = ['mode', 'count']
    c += [call('mode', [np.array([])], names=M), call('mode', [np.zeros((0, 3))], names=M),
          call('mode', [np.array([np.nan, np.nan])], {'nan_policy': 'omit'}, names=M),
          call('mode', [np.array([[1.0, np.nan], [np.nan, np.nan]])], {'axis': 1, 'nan_policy': 'omit'}, names=M),
          call('mode', [xn2], {'axis': None, 'nan_policy': 'omit'}, names=M),
          call('mode', [data(400, ties=True)], names=M)]
    B['mode'] = c

    # ---- trimmed statistics
    for fn in ('tmean', 'tvar', 'tstd', 'tsem'):
        c = battery(fn, empty=False, small=False)
        c += [call(fn, [data(20)], {'limits': [-4.0, 6.0]}),
              call(fn, [data(20)], {'limits': [-4.0, 6.0], 'inclusive': [False, True]}),
              call(fn, [np.arange(10.0)], {'limits': [2.0, 7.0], 'inclusive': [False, False]}),
              call(fn, [np.arange(10.0)], {'limits': [2.0, 7.0]}),
              call(fn, [data((5, 8))], {'limits': [-5.0, 5.0], 'axis': 1}),
              call(fn, [data((5, 8))], {'limits': [-5.0, 5.0], 'axis': 0, 'keepdims': True}),
              call(fn, [data(10)], {'limits': [100.0, 200.0]}),
              call(fn, [np.arange(12)], {'limits': [3, 9]}),
              call(fn, [np.array([1.0, 2.0, 3.0])], {'limits': [1.5, 2.5]})]
        if fn != 'tmean':
            c.append(call(fn, [data(15)], {'limits': [-5.0, 5.0], 'ddof': 0}))
        B[fn] = c
    for fn, lk in (('tmin', 'lowerlimit'), ('tmax', 'upperlimit')):
        c = battery(fn, empty=False, small=False)
        c += [call(fn, [data(20)], {lk: 1.0}), call(fn, [data(20)], {lk: 1.0, 'inclusive': False}),
              call(fn, [np.arange(10)], {lk: 4}), call(fn, [np.arange(10)], {lk: 4, 'inclusive': False}),
              call(fn, [data((5, 8))], {lk: 0.0, 'axis': 1}), call(fn, [data((5, 8))], {lk: 0.0, 'axis': None}),
              call(fn, [np.array([[1, 9], [2, 3]])], {lk: 5, 'axis': 1}),
              call(fn, [data(10)], {lk: 100.0 if fn == 'tmin' else -100.0})]
        B[fn] = c
    c = []
    for p in (0.0, 0.1, 0.25, 0.4, 0.5):
        c.append(call('trim_mean', [data(20), p]))
    x = data((6, 9))
    c += [call('trim_mean', [x, 0.2], {'axis': 0}), call('trim_mean', [x, 0.2], {'axis': 1}),
          call('trim_mean', [x, 0.2], {'axis': None}), call('trim_mean', [x, 0.3], {'axis': 1, 'keepdims': True}),
          call('trim_mean', [data(11, ints=True), 0.1]), call('trim_mean', [data(4), 0.6]),
          call('trim_mean', [np.array([]), 0.1])]
    xn = data(12); xn[4] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('trim_mean', [xn, 0.1], {'nan_policy': pol}))
    B['trim_mean'] = c
    c = []
    for p in (0.0, 0.1, 0.2, 0.34):
        c.append(call_trim('trimboth', [data(20), p]))
    c += [call_trim('trimboth', [data((6, 10)), 0.2], {'axis': 1}),
          call_trim('trimboth', [data((10, 3)), 0.2]),
          call_trim('trimboth', [data((4, 5)), 0.2], {'axis': None}),
          call_trim('trimboth', [data(10, ints=True), 0.2]),
          call_trim('trimboth', [data(4), 0.5]), call_trim('trimboth', [np.array([]), 0.1])]
    B['trimboth'] = c
    c = []
    for p, tail in ((0.1, 'right'), (0.25, 'left'), (0.0, 'right'), (0.5, 'left'), (1.0, 'right')):
        c.append(call_trim('trim1', [data(20), p], {'tail': tail}))
    c += [call_trim('trim1', [data((6, 10)), 0.2], {'axis': 1}),
          call_trim('trim1', [data((10, 3)), 0.3], {'tail': 'left'}),
          call_trim('trim1', [data((4, 5)), 0.2], {'axis': None}),
          call_trim('trim1', [data(10, ints=True), 0.2]),
          call_trim('trim1', [data(10), 0.2], {'tail': 'middle'})]
    B['trim1'] = c
    x = data(40)
    x[[3, 9]] = [60.0, -55.0]
    B['sigmaclip'] = [call('sigmaclip', [x], names=['clipped', 'lower', 'upper']),
                      call('sigmaclip', [x, 1.5, 2.0], names=['clipped', 'lower', 'upper']),
                      call('sigmaclip', [data((5, 6))], {'low': 1.0, 'high': 1.0}, names=['clipped', 'lower', 'upper']),
                      call('sigmaclip', [data(12, ints=True), 1.0, 1.0], names=['clipped', 'lower', 'upper']),
                      call('sigmaclip', [np.array([2.0, 2.0, 2.0])], names=['clipped', 'lower', 'upper'])]

    # ---- percentiles, ranks
    c = []
    a = data(17)
    for per in (0, 10, 25, 50, 63.5, 100):
        c.append(call('scoreatpercentile', [a, per]))
    c += [call('scoreatpercentile', [a, [10, 50, 90]]), call('scoreatpercentile', [a, 30], {'interpolation_method': 'lower'}),
          call('scoreatpercentile', [a, 30], {'interpolation_method': 'higher'}),
          call('scoreatpercentile', [a, 30], {'interpolation_method': 'nearest'}),
          call('scoreatpercentile', [a, 30], {'limit': [-5.0, 5.0]}),
          call('scoreatpercentile', [data((6, 5)), 40], {'axis': 0}), call('scoreatpercentile', [data((6, 5)), 40], {'axis': 1}),
          call('scoreatpercentile', [data((6, 5)), [20, 80]], {'axis': 1}), call('scoreatpercentile', [data((6, 5)), 40]),
          call('scoreatpercentile', [data(9, ints=True), 35]), call('scoreatpercentile', [a, 101]),
          call('scoreatpercentile', [np.array([1.0, 2.0, 3.0]), 0], {'limit': [10.0, 20.0]}),
          call('scoreatpercentile', [np.array([1.0, 2.0, 3.0]), 50], {'limit': [10.0, 20.0]}),
          call('scoreatpercentile', [np.array([]), 50]), call('scoreatpercentile', [np.array([]), [10, 20]])]
    B['scoreatpercentile'] = c
    c = []
    a = data(20, ties=True)
    for kind in ('rank', 'weak', 'strict', 'mean'):
        c.append(call('percentileofscore', [a, 1.5], {'kind': kind}))
        c.append(call('percentileofscore', [a, np.array([0.0, 1.5, 2.2, 9.0])], {'kind': kind}))
    an = a.copy(); an[2] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('percentileofscore', [an, 1.5], {'nan_policy': pol}))
        c.append(call('percentileofscore', [a, np.array([1.0, np.nan])], {'nan_policy': pol}))
    c += [call('percentileofscore', [a, 1.5], {'kind': 'bogus'}), call('percentileofscore', [data((3, 3)), 1.0]),
          call('percentileofscore', [np.array([]), 1.0]), call('percentileofscore', [data(10, ints=True), 2]),
          call('percentileofscore', [a, np.array([[1.0, 2.0], [3.0, 0.5]])])]
    B['percentileofscore'] = c
    c = []
    a = data(14, ties=True)
    for m in ('average', 'min', 'max', 'dense', 'ordinal'):
        c.append(call('rankdata', [a], {'method': m}))
        c.append(call('rankdata', [data((4, 6), ties=True)], {'method': m, 'axis': 1}))
    x = data((5, 4), ties=True)
    c += [call('rankdata', [x]), call('rankdata', [x], {'axis': 0}), call('rankdata', [x], {'axis': -1}),
          call('rankdata', [data(8, ints=True)]), call('rankdata', [a], {'method': 'bogus'}), call('rankdata', [np.array([])]),
          call('rankdata', [np.array([])], {'method': 'min'})]
    xn = data((3, 6), ties=True); xn[1, 2] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('rankdata', [xn], {'axis': 1, 'nan_policy': pol}))
        c.append(call('rankdata', [xn], {'axis': 1, 'nan_policy': pol, 'method': 'min'}))
    B['rankdata'] = c
    B['tiecorrect'] = [call('tiecorrect', [st.rankdata(data(15, ties=True))]), call('tiecorrect', [np.array([1.0, 2.5, 2.5, 4.0])]),
                       call('tiecorrect', [np.arange(1.0, 7.0)]), call('tiecorrect', [np.array([3.0])]),
                       call('tiecorrect', [np.array([2.0, 2.0, 2.0])]), call('tiecorrect', [np.array([1, 1, 3, 3, 3])])]

    # ---- entropy
    c = []
    pk = data(8, lo=0, hi=1)
    qk = data(8, lo=0.01, hi=1)
    c += [call('entropy', [pk]), call('entropy', [pk, qk]), call('entropy', [pk], {'base': 2}),
          call('entropy', [pk, qk], {'base': 10.0}), call('entropy', [np.array([0.5, 0.0, 0.5])]),
          call('entropy', [np.array([1, 2, 3])]), call('entropy', [pk], {'base': -1.0})]
    P = data((4, 6), lo=0, hi=1)
    Q = data((4, 6), lo=0.01, hi=1)
    for ax in (0, 1, None):
        c.append(call('entropy', [P], {'axis': ax}))
        c.append(call('entropy', [P, Q], {'axis': ax}))
    c.append(call('entropy', [P], {'axis': 1, 'keepdims': True}))
    pn = pk.copy(); pn[2] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('entropy', [pn], {'nan_policy': pol}))
        c.append(call('entropy', [pk, np.where(np.arange(8) == 5, np.nan, qk)], {'nan_policy': pol}))
    c.append(call('entropy', [np.array([])]))
    c.append(call('entropy', [np.array([1.0])]))
    B['entropy'] = c
    c = []
    for n in (8, 25, 60):
        v = data(n)
        for m in ('auto', 'vasicek', 'van es', 'correa', 'ebrahimi'):
            c.append(call('differential_entropy', [v], {'method': m}))
    v = data(30)
    c += [call('differential_entropy', [v], {'window_length': 3}), call('differential_entropy', [v], {'base': 2.0}),
          call('differential_entropy', [v], {'window_length': 3, 'method': 'correa'}),
          call('differential_entropy', [data((3, 20))], {'axis': 1}),
          call('differential_entropy', [data((20, 3))], {'axis': 0, 'method': 'vasicek'}),
          call('differential_entropy', [data((3, 20))], {'axis': 1, 'keepdims': True}),
          call('differential_entropy', [data(3)]), call('differential_entropy', [v], {'window_length': 20}),
          call('differential_entropy', [data(12, ints=True)]), call('differential_entropy', [v], {'base': -2.0})]
    vn = data(20); vn[5] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('differential_entropy', [vn], {'nan_policy': pol}))
    B['differential_entropy'] = c

    # ---- frequency tables
    for fn, names in (('cumfreq', ['cumcount', 'lowerlimit', 'binsize', 'extrapoints']),
                      ('relfreq', ['frequency', 'lowerlimit', 'binsize', 'extrapoints'])):
        a = data(30)
        B[fn] = [call(fn, [a], names=names), call(fn, [a], {'numbins': 4}, names=names),
                 call(fn, [a], {'numbins': 5, 'defaultreallimits': [-5.0, 5.0]}, names=names),
                 call(fn, [a], {'numbins': 5, 'defaultreallimits': [-5, 5]}, names=names),
                 call(fn, [a], {'numbins': 6, 'weights': data(30, lo=0, hi=2)}, names=names),
                 call(fn, [data((4, 5))], {'numbins': 3}, names=names),
                 call(fn, [data(20, ints=True)], names=names), call(fn, [np.array([2.0, 2.0, 2.0])], {'numbins': 3}, names=names),
                 call(fn, [np.array([])], {'numbins': 4}, names=names),
                 call(fn, [np.array([1.0, np.nan, 3.0])], names=names)]

    # ---- obrientransform
    B['obrientransform'] = [call('obrientransform', [data(8)]), call('obrientransform', [data(6), data(6), data(6)]),
                            call('obrientransform', [data(5, ints=True), data(5)]), call('obrientransform', [data(4), data(5)])]

    # ---- k-statistics
    for fn, ns in (('kstat', (1, 2, 3, 4)), ('kstatvar', (1, 2))):
        c = []
        for k in ns:
            c.append(call(fn, [data(15), k]))
            c.append(call(fn, [data(15, ints=True), k]))
        x = data((5, 8))
        c += [call(fn, [x], {'axis': 0}), call(fn, [x], {'axis': 1}), call(fn, [x, 3 if fn == 'kstat' else 1], {'axis': 1}),
              call(fn, [x]), call(fn, [x], {'axis': 1, 'keepdims': True}), call(fn, [data(15), 5]), call(fn, [data(15), 0]),
              call(fn, [np.array([])]), call(fn, [data(1)]), call(fn, [data(3)])]
        xn = data(12); xn[2] = np.nan
        for pol in ('propagate', 'omit', 'raise'):
            c.append(call(fn, [xn], {'nan_policy': pol}))
        B[fn] = c

    # ---- L-moments
    c = [call('lmoment', [data(12)]), call('lmoment', [data(12)], {'order': 2}), call('lmoment', [data(12)], {'order': [1, 3]}),
         call('lmoment', [data(12)], {'standardize': False}), call('lmoment', [np.sort(data(12))], {'sorted': True}),
         call('lmoment', [data((4, 9))], {'axis': 1}), call('lmoment', [data((9, 4))], {'axis': 0, 'order': 3}),
         call('lmoment', [data((4, 9))], {'axis': None}), call('lmoment', [data((4, 9))], {'axis': 1, 'keepdims': True, 'order': 2}),
         call('lmoment', [data(3)]), call('lmoment', [data(1)], {'order': 1}), call('lmoment', [data(8)], {'order': [5, 6]}),
         call('lmoment', [data(8)], {'order': 0}), call('lmoment', [data(8, ints=True)]), call('lmoment', [np.array([])])]
    xn = data(10); xn[1] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('lmoment', [xn], {'nan_policy': pol}))
    B['lmoment'] = c

    # ---- expectile
    a = data(25)
    B['expectile'] = [call('expectile', [a]), call('expectile', [a, 0.1]), call('expectile', [a, 0.9]), call('expectile', [a, 0.0]),
                      call('expectile', [a, 1.0]), call('expectile', [a, 0.3], {'weights': data(25, lo=0, hi=2)}),
                      call('expectile', [data((3, 4)), 0.7]), call('expectile', [np.array([2.0, 2.0])]),
                      call('expectile', [data(10, ints=True), 0.25]), call('expectile', [a, 1.5])]

    # ---- circular statistics
    for fn in ('circmean', 'circvar', 'circstd'):
        a = data(15, lo=0, hi=6)
        c = [call(fn, [a]), call(fn, [a, 3.0, -3.0]), call(fn, [data(15, lo=0, hi=360)], {'high': 360}),
             call(fn, [data(15, lo=-180, hi=180)], {'high': 180, 'low': -180}), call(fn, [data((4, 7), lo=0, hi=6)], {'axis': 0}),
             call(fn, [data((4, 7), lo=0, hi=6)], {'axis': 1}), call(fn, [data((4, 7), lo=0, hi=6)]),
             call(fn, [data((4, 7), lo=0, hi=6)], {'axis': 1, 'keepdims': True}), call(fn, [data(7, ints=True) + 5]),
             call(fn, [np.array([])]), call(fn, [np.array([1.0])]), call(fn, [np.array([0.1, 0.1 + np.pi])])]
        an = data(10, lo=0, hi=6); an[4] = np.nan
        for pol in ('propagate', 'omit', 'raise'):
            c.append(call(fn, [an], {'nan_policy': pol}))
        if fn == 'circstd':
            c.append(call(fn, [a], {'normalize': True}))
            c.append(call(fn, [data(15, lo=0, hi=360)], {'high': 360, 'normalize': True}))
        B[fn] = c

    # ---- FDR control
    ps = data(12, lo=0, hi=0.2)
    B['false_discovery_control'] = [
        call('false_discovery_control', [ps]), call('false_discovery_control', [ps], {'method': 'by'}),
        call('false_discovery_control', [np.array([0.01, 0.02, 0.02, 0.04, 0.3, 0.8])]),
        call('false_discovery_control', [data((3, 6), lo=0, hi=0.5)], {'axis': 1}),
        call('false_discovery_control', [data((6, 3), lo=0, hi=0.5)], {'axis': 0, 'method': 'by'}),
        call('false_discovery_control', [data((3, 4), lo=0, hi=0.5)], {'axis': None}),
        call('false_discovery_control', [np.array([0.3])]), call('false_discovery_control', [np.array([0.1, 1.2])]),
        call('false_discovery_control', [np.array([0.1, np.nan])]), call('false_discovery_control', [np.array([1.0, 1.0, 0.99])])]

    # ---- Box-Cox / Yeo-Johnson log-likelihoods
    for fn in ('boxcox_llf', 'yeojohnson_llf'):
        d = data(20, lo=0.1, hi=10) if fn == 'boxcox_llf' else data(20)
        d2 = data((4, 8), lo=0.1, hi=10) if fn == 'boxcox_llf' else data((4, 8))
        c = []
        for lmb in (0.0, 0.5, 1.0, 2.0, -1.3, 1e-20):
            c.append(call(fn, [lmb, d]))
        c += [call(fn, [0.7, d2]), call(fn, [0.7, d2], {'axis': 1}), call(fn, [0.7, d2], {'axis': None}),
              call(fn, [0.7, d2], {'axis': 1, 'keepdims': True}), call(fn, [1.5, np.array([])]), call(fn, [0.3, np.array([2.0])]),
              call(fn, [0.3, np.abs(data(10, ints=True)) + 1])]
        dn = d.copy(); dn[3] = np.nan
        for pol in ('propagate', 'omit', 'raise'):
            c.append(call(fn, [0.4, dn], {'nan_policy': pol}))
        B[fn] = c

    # ---- CDF distances
    for fn in ('energy_distance', 'wasserstein_distance'):
        u, v = data(10), data(14)
        B[fn] = [call(fn, [u, v]), call(fn, [u, v, data(10, lo=0, hi=2), data(14, lo=0, hi=2)]),
                 call(fn, [u, v, data(10, lo=0, hi=2)]), call(fn, [np.array([0.0, 1.0, 3.0]), np.array([5.0, 6.0, 8.0])]),
                 call(fn, [np.array([0, 1, 1, 2]), np.array([1, 1, 3])]), call(fn, [data(8, ties=True), data(8, ties=True)]),
                 call(fn, [u, np.array([])]), call(fn, [u, v, -data(10, lo=0, hi=1)]), call(fn, [u, v, np.zeros(10)]),
                 call(fn, [u, v, np.ones(3)])]

    # ---- describe
    c = [call_describe([data(12)]), call_describe([data((5, 7))]), call_describe([data((5, 7))], {'axis': 1}),
         call_describe([data((5, 7))], {'axis': None}), call_describe([data(12)], {'ddof': 0, 'bias': False}),
         call_describe([data(15, ints=True)]), call_describe([data((4, 6), ints=True)], {'axis': 1}),
         call_describe([np.array([1.0])]), call_describe([np.array([2.0, 5.0])]), call_describe([np.array([])]),
         call_describe([data((3, 4, 5))], {'axis': 2})]
    xn = data((4, 8)); xn[1, 3] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call_describe([xn], {'axis': 1, 'nan_policy': pol}))
        c.append(call_describe([xn[1]], {'nan_policy': pol}))
    c.append(call_describe([xn], {'axis': None, 'nan_policy': 'omit'}))
    c.append(call_describe([xn], {'axis': 0, 'nan_policy': 'omit', 'bias': False}))
    B['describe'] = c

    # ---- iqr, median_abs_deviation
    c = battery('iqr', axis_default=None)
    a = data(21)
    for interp in ('linear', 'lower', 'higher', 'midpoint', 'nearest', 'hazen', 'weibull', 'median_unbiased',
                   'normal_unbiased', 'inverted_cdf', 'averaged_inverted_cdf', 'closest_observation',
                   'interpolated_inverted_cdf'):
        c.append(call('iqr', [a], {'interpolation': interp}))
        c.append(call('iqr', [data(8)], {'interpolation': interp, 'rng': [10, 90]}))
    c += [call('iqr', [a], {'rng': [10, 90]}), call('iqr', [a], {'rng': [80, 30]}), call('iqr', [a], {'scale': 'normal'}),
          call('iqr', [a], {'scale': 2.0}), call('iqr', [a], {'scale': 'bogus'}), call('iqr', [a], {'rng': [-1, 50]}),
          call('iqr', [a], {'interpolation': 'bogus'}), call('iqr', [data((5, 6))], {'axis': 0, 'rng': [5, 95], 'scale': 'normal'})]
    B['iqr'] = c
    c = battery('median_abs_deviation')
    c += [call('median_abs_deviation', [data(13)], {'scale': 'normal'}), call('median_abs_deviation', [data(13)], {'scale': 2.5}),
          call('median_abs_deviation', [data((4, 6))], {'axis': 1, 'scale': 'normal'}),
          call('median_abs_deviation', [data(10)], {'scale': 'bogus'})]
    B['median_abs_deviation'] = c

    # ---- adversarial parameters (independent verification and sanitizer fuzz pass): SciPy raises for negative
    # cuts (np.partition kth out of bounds) and NaN ones (int(nan)); negative cuts down to -n are valid for trim1 left
    for p in (-0.1, -0.5, -40.0, float('nan'), -0.05):
        B['trim_mean'].append(call('trim_mean', [data(10), p]))
        B['trimboth'].append(call_trim('trimboth', [data(10), p]))
    for p, tail in ((-0.1, 'left'), (-0.5, 'left'), (-1.0, 'left'), (-1.5, 'left'), (-0.1, 'right'), (float('nan'), 'left')):
        B['trim1'].append(call_trim('trim1', [np.arange(10.0), p], {'tail': tail}))
    B['kstat'] += [call('kstat', [data(6), float('nan')]), call('kstat', [data(6), 2.5]), call('kstat', [data(6), 4.9])]
    B['kstatvar'] += [call('kstatvar', [data(6), float('nan')]), call('kstatvar', [data(6), 2.5])]
    B['lmoment'] += [call('lmoment', [data(6)], {'order': 3e9})]
    write('desc', B)


if __name__ == '__main__':
    main()
