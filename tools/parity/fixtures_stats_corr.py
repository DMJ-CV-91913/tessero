#!/usr/bin/env python3
"""Fixtures for the scipy.stats correlation / regression group (Tessero registry, csrc/cxx/wip/stats_fn_corr.cpp).

Writes tests/fixtures/parity/stats_corr.json (or $FIXTURE_OUT). Seeded data only.

Inputs are C-contiguous, as PHP arrays always are (NumPy's summation and BLAS paths depend on the layout).

Representation notes (what Tessero returns where SciPy returns a Python object the registry cannot):
  - binned_statistic_dd: `bin_edges` (a list of D edge arrays) is recorded as one (D, k) array
    (np.asarray of the list); every case here has the same number of edges per dimension.
"""
import warnings

import numpy as np
import scipy.stats as st

import fixtures_np as F
from fixtures_stats_fn import call, write, data, rng

SIG = ['statistic', 'pvalue']
LIN = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr', 'intercept_stderr']
THEIL = ['slope', 'intercept', 'low_slope', 'high_slope']
SIEGEL = ['slope', 'intercept']
SOMERS = ['statistic', 'pvalue', 'table']
B1 = ['statistic', 'bin_edges', 'binnumber']
B2 = ['statistic', 'x_edge', 'y_edge', 'binnumber']
ALTS = ['two-sided', 'less', 'greater']


def u(shape, lo=-10.0, hi=10.0):
    return data(shape, lo=lo, hi=hi)


def corr_pair(n, rho=0.5):
    x = rng.normal(size=n)
    y = rho * x + rng.normal(size=n)
    return x, y


def ties(shape, k=4):
    return (rng.integers(0, k, size=shape) / 2.0)


def call_dd(args, kwargs):
    """binned_statistic_dd with bin_edges stacked into one array (see the module docstring)."""
    kwargs = kwargs or {}
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            r = st.binned_statistic_dd(*args, **kwargs)
            expect = {'dict': {'statistic': F.enc_result_one(r.statistic),
                               'bin_edges': F.enc_result_one(np.asarray(r.bin_edges)),
                               'binnumber': F.enc_result_one(r.binnumber)}}
        except Exception as e:  # noqa: BLE001
            expect = {'error': type(e).__name__}
    return {'args': [F.enc(a) for a in args], 'kwargs': {k: F.enc(v) for k, v in kwargs.items()},
            'expect': expect, 'compare': 'tol'}


def pearsonr_cases():
    c = []
    for n in (2, 3, 10, 16, 17, 33, 40, 100, 257):
        x, y = corr_pair(n)
        c.append(call('pearsonr', [x, y], names=SIG))
    x, y = corr_pair(50)
    for a in ALTS:
        c.append(call('pearsonr', [x, y], {'alternative': a}, names=SIG))
    c.append(call('pearsonr', [x, -x + 1e-3 * y], names=SIG))
    c.append(call('pearsonr', [x, np.full(50, 3.0)], names=SIG))                     # constant
    c.append(call('pearsonr', [data(12, ints=True), data(12, ints=True)], names=SIG))
    c.append(call('pearsonr', [x[:2], y[:2]], {'alternative': 'less'}, names=SIG))
    X, Y = rng.normal(size=(6, 40)), rng.normal(size=(6, 40))
    Y = Y + 0.3 * X
    for ax in (0, 1, -1, None):
        c.append(call('pearsonr', [X, Y], {'axis': ax}, names=SIG))
    X, Y = rng.normal(size=(37, 5)), rng.normal(size=(37, 5))
    c.append(call('pearsonr', [X, Y + X], {'axis': 0}, names=SIG))
    c.append(call('pearsonr', [X, Y], {'axis': 0, 'alternative': 'greater'}, names=SIG))
    X3, Y3 = rng.normal(size=(3, 20, 4)), rng.normal(size=(3, 20, 4))
    c.append(call('pearsonr', [X3, Y3 - X3], {'axis': 1}, names=SIG))
    xn = x.copy(); xn[3] = np.nan
    c.append(call('pearsonr', [xn, y], names=SIG))
    c.append(call('pearsonr', [x[:1], y[:1]], names=SIG))                               # error
    c.append(call('pearsonr', [x[:5], y[:6]], names=SIG))                               # error
    return c


def paired_axis_cases(fn, names, extra=None, nmax=30):
    """2-D axis / keepdims / nan_policy cases for a decorated paired-sample function."""
    extra = extra or {}
    c = []
    X, Y = rng.normal(size=(5, nmax)), rng.normal(size=(5, nmax))
    Y = Y + 0.5 * X
    for ax in (0, 1, None):
        c.append(call(fn, [X, Y], {'axis': ax, **extra}, names=names))
    c.append(call(fn, [X, Y], {'axis': 1, 'keepdims': True, **extra}, names=names))
    Xn = X.copy(); Xn[1, 3] = np.nan; Xn[3, 0] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call(fn, [Xn, Y], {'axis': 1, 'nan_policy': pol, **extra}, names=names))
    x, y = X[0], Y[0]
    xn = x.copy(); xn[[2, 7]] = np.nan
    for pol in ('propagate', 'omit'):
        c.append(call(fn, [xn, y], {'nan_policy': pol, **extra}, names=names))
    c.append(call(fn, [x[:1], y[:1]], extra, names=names))                              # too small
    return c


def kendall_cases():
    c = []
    for n in (2, 3, 5, 10, 20, 33, 34, 50):
        x, y = corr_pair(n, 0.4)
        c.append(call('kendalltau', [x, y], names=SIG))
    x, y = corr_pair(25, 0.3)
    for a in ALTS:
        for m in ('auto', 'exact', 'asymptotic'):
            c.append(call('kendalltau', [x, y], {'alternative': a, 'method': m}, names=SIG))
    c.append(call('kendalltau', [x, y], {'variant': 'c'}, names=SIG))
    xt, yt = ties(30), ties(30, 6)
    for a in ALTS:
        c.append(call('kendalltau', [xt, yt], {'alternative': a}, names=SIG))
    c.append(call('kendalltau', [xt, yt], {'variant': 'c'}, names=SIG))
    c.append(call('kendalltau', [xt, yt], {'method': 'exact'}, names=SIG))            # error: ties
    s = np.arange(12.0)
    c.append(call('kendalltau', [s, s], names=SIG))                                     # c == 0 / perfect
    s2 = s.copy(); s2[[4, 5]] = s2[[5, 4]]
    for a in ALTS:
        c.append(call('kendalltau', [s, s2], {'alternative': a}, names=SIG))           # dis == 1
        c.append(call('kendalltau', [s, s2[::-1]], {'alternative': a}, names=SIG))
    big = np.arange(60.0); big2 = big.copy(); big2[[10, 11]] = big2[[11, 10]]
    c.append(call('kendalltau', [big, big2], names=SIG))                                # exact, n > 33
    c.append(call('kendalltau', [x[:2], y[:2]], {'method': 'asymptotic'}, names=SIG))  # ZeroDivisionError
    xb, yb = corr_pair(175, 0.2)
    for a in ALTS:
        c.append(call('kendalltau', [xb, yb], {'method': 'exact', 'alternative': a}, names=SIG))  # n >= 171
    c.append(call('kendalltau', [np.arange(180.0), np.r_[1.0, 0.0, np.arange(2.0, 180.0)]], {'alternative': 'less'}, names=SIG))
    c.append(call('kendalltau', [np.ones(5), y[:5]], names=SIG))
    c.append(call('kendalltau', [data(15, ints=True), data(15, ints=True)], names=SIG))
    # SciPy's test values (test_stats.TestKendallTau)
    x1 = [12, 2, 1, 12, 2]; x2 = [1, 4, 7, 1, 0]
    c.append(call('kendalltau', [np.array(x1, float), np.array(x2, float)], names=SIG))
    c += paired_axis_cases('kendalltau', SIG)
    return c


def weightedtau_cases():
    c = []
    x = np.array([12, 2, 1, 12, 2], float); y = np.array([1, 4, 7, 1, 0], float)
    for kw in ({}, {'rank': None}, {'rank': False}, {'additive': False}, {'rank': None, 'additive': False}):
        c.append(call('weightedtau', [x, y], kw, names=SIG))
    for n in (2, 3, 10, 40):
        a, b = corr_pair(n, 0.4)
        c.append(call('weightedtau', [a, b], names=SIG))
        c.append(call('weightedtau', [a, b], {'rank': None}, names=SIG))
    a, b = ties(30), ties(30, 6)
    c.append(call('weightedtau', [a, b], names=SIG))
    c.append(call('weightedtau', [a, b], {'rank': False, 'additive': False}, names=SIG))
    a, b = corr_pair(20, 0.2)
    c.append(call('weightedtau', [a, b], {'rank': rng.permutation(20).astype(float)}, names=SIG))
    c.append(call('weightedtau', [a, b], {'rank': ties(20)}, names=SIG))
    rk = ties(20); rk[4] = np.nan
    c.append(call('weightedtau', [a, b], {'rank': rk}, names=SIG))
    c.append(call('weightedtau', [a, b], {'rank': rk, 'nan_policy': 'omit'}, names=SIG))
    an = a.copy(); an[[1, 5]] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('weightedtau', [an, b], {'nan_policy': pol}, names=SIG))
    c.append(call('weightedtau', [np.ones(6), b[:6]], names=SIG))                       # constant
    c.append(call('weightedtau', [data(15, ints=True), data(15, ints=True)], names=SIG))
    X, Y = rng.normal(size=(4, 12)), rng.normal(size=(4, 12))
    for ax in (0, 1, None):
        c.append(call('weightedtau', [X, Y], {'axis': ax}, names=SIG))
    c.append(call('weightedtau', [X, Y], {'axis': 1, 'keepdims': True}, names=SIG))
    c.append(call('weightedtau', [a[:1], b[:1]], names=SIG))
    c.append(call('weightedtau', [a[:5], b[:6]], names=SIG))                            # error
    return c


def spearmanr_cases():
    c = []
    for n in (2, 3, 10, 40, 150):
        x, y = corr_pair(n, 0.4)
        c.append(call('spearmanr', [x, y], names=SIG))
    x, y = corr_pair(30, 0.4)
    for a in ALTS:
        c.append(call('spearmanr', [x, y], {'alternative': a}, names=SIG))
    c.append(call('spearmanr', [ties(30), ties(30, 6)], names=SIG))
    A = rng.normal(size=(25, 4)); A[:, 1] += A[:, 0]
    c.append(call('spearmanr', [A], names=SIG))
    c.append(call('spearmanr', [A[:, :2]], names=SIG))
    c.append(call('spearmanr', [np.ascontiguousarray(A.T)], {'axis': 1}, names=SIG))
    c.append(call('spearmanr', [A[:, :2], A[:, 2:]], names=SIG))
    c.append(call('spearmanr', [A[:, :3], A[:, 3]], {'alternative': 'greater'}, names=SIG))
    c.append(call('spearmanr', [A], {'axis': None}, names=SIG))                        # error
    c.append(call('spearmanr', [A[:, 0], A[:, 1]], {'axis': None}, names=SIG))
    c.append(call('spearmanr', [x[:1], y[:1]], names=SIG))
    c.append(call('spearmanr', [np.ones(8), y[:8]], names=SIG))
    c.append(call('spearmanr', [x], names=SIG))                                         # error
    xn = x.copy(); xn[[3, 9]] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('spearmanr', [xn, y], {'nan_policy': pol}, names=SIG))
    An = A.copy(); An[2, 1] = np.nan; An[7, 3] = np.nan
    for pol in ('propagate', 'omit'):
        c.append(call('spearmanr', [An], {'nan_policy': pol}, names=SIG))
    c.append(call('spearmanr', [An], {'nan_policy': 'omit', 'alternative': 'less'}, names=SIG))
    c.append(call('spearmanr', [data(20, ints=True), data(20, ints=True)], names=SIG))
    xc = np.r_[np.ones(6), np.nan]; yc = rng.normal(size=7); yc[0] = np.nan
    c.append(call('spearmanr', [xc, yc], {'nan_policy': 'omit'}, names=SIG))
    c.append(call('spearmanr', [xn[:4], y[:4]], {'nan_policy': 'omit'}, names=SIG))
    c.append(call('spearmanr', [np.array([np.nan, 1.0, 2.0]), np.array([1.0, np.nan, 3.0])], {'nan_policy': 'omit'}, names=SIG))
    B = rng.normal(size=(60, 5)); B[5, 2] = np.nan
    c.append(call('spearmanr', [B], names=SIG))
    return c


def linregress_cases():
    c = []
    for n in (2, 3, 10, 50, 300):
        x, y = corr_pair(n, 2.0)
        c.append(call('linregress', [x, y], names=LIN))
    x, y = corr_pair(20, -1.5)
    for a in ALTS:
        c.append(call('linregress', [x, y], {'alternative': a}, names=LIN))
    c.append(call('linregress', [x, 3 * x + 1], names=LIN))
    c.append(call('linregress', [x, np.full(20, 2.0)], names=LIN))
    c.append(call('linregress', [np.full(5, 2.0), y[:5]], names=LIN))                  # error
    c.append(call('linregress', [np.array([1.0, 2.0]), np.array([3.0, 3.0])], names=LIN))
    c.append(call('linregress', [data(12, ints=True), data(12, ints=True)], names=LIN))
    c.append(call('linregress', [np.arange(10.0), np.arange(10.0) ** 2], names=LIN))
    X, Y = rng.normal(size=(3, 7)), rng.normal(size=(3, 7))
    c.append(call('linregress', [X, Y], {'axis': None, 'keepdims': True}, names=LIN))
    c.append(call('linregress', [X, Y[0]], {'axis': 1}, names=LIN))
    c += paired_axis_cases('linregress', LIN)
    return c


def theil_cases():
    c = []
    y = np.array([1, 2, 3, 4, 5, 6, 7, 6, 5.5, 8.0])
    c.append(call('theilslopes', [y], names=THEIL))
    c.append(call('theilslopes', [y], {'method': 'joint'}, names=THEIL))
    for al in (0.9, 0.99, 0.05, 0.5, 0.0, 1.0):
        c.append(call('theilslopes', [y], {'alpha': al}, names=THEIL))
    x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3.0])
    c.append(call('theilslopes', [y, x], names=THEIL))
    c.append(call('theilslopes', [y, x], {'method': 'joint'}, names=THEIL))
    for n in (2, 3, 25):
        a, b = corr_pair(n, 1.0)
        c.append(call('theilslopes', [b, a], names=THEIL))
    c.append(call('theilslopes', [ties(20), ties(20, 6)], names=THEIL))
    c.append(call('theilslopes', [y[:3], np.ones(3)], names=THEIL))                    # identical x
    c.append(call('theilslopes', [y], {'method': 'x'}, names=THEIL))                    # error
    Y = rng.normal(size=(4, 9))
    for ax in (0, 1, None):
        c.append(call('theilslopes', [Y], {'axis': ax}, names=THEIL))
    c.append(call('theilslopes', [Y, Y + rng.normal(size=(4, 9))], {'axis': 1, 'keepdims': True}, names=THEIL))
    yn = y.copy(); yn[4] = np.nan
    for pol in ('propagate', 'omit'):
        c.append(call('theilslopes', [yn, x], {'nan_policy': pol}, names=THEIL))
        c.append(call('theilslopes', [yn], {'nan_policy': pol}, names=THEIL))
    c.append(call('theilslopes', [data(12, ints=True)], names=THEIL))
    return c


def siegel_cases():
    c = []
    y = np.array([1, 2, 3, 4, 5, 6, 7, 6, 5.5, 8.0])
    for m in ('hierarchical', 'separate'):
        c.append(call('siegelslopes', [y], {'method': m}, names=SIEGEL))
        x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3.0])
        c.append(call('siegelslopes', [y, x], {'method': m}, names=SIEGEL))
        for n in (2, 3, 8, 25, 40):
            a, b = corr_pair(n, 1.0)
            c.append(call('siegelslopes', [b, a], {'method': m}, names=SIEGEL))
        c.append(call('siegelslopes', [ties(20), ties(20, 6) + 0.25 * np.arange(20)], {'method': m}, names=SIEGEL))
    c.append(call('siegelslopes', [y], {'method': 'x'}, names=SIEGEL))                  # error
    Y = rng.normal(size=(3, 8))
    for ax in (0, 1, None):
        c.append(call('siegelslopes', [Y], {'axis': ax}, names=SIEGEL))
    yn = y.copy(); yn[2] = np.nan
    for pol in ('propagate', 'omit'):
        c.append(call('siegelslopes', [yn], {'nan_policy': pol}, names=SIEGEL))
    c.append(call('siegelslopes', [data(9, ints=True)], names=SIEGEL))
    return c


def chatterjee_cases():
    c = []
    for n in (2, 3, 10, 50, 200):
        x = rng.normal(size=n)
        y = np.sin(3 * x) + 0.3 * rng.normal(size=n)
        c.append(call('chatterjeexi', [x, y], names=SIG))
        c.append(call('chatterjeexi', [x, y], {'y_continuous': True}, names=SIG))
    x = rng.normal(size=40)
    c.append(call('chatterjeexi', [x, ties(40)], names=SIG))                            # ties in y only
    c.append(call('chatterjeexi', [x, np.ones(40)], names=SIG))
    c += paired_axis_cases('chatterjeexi', SIG)
    return c


def spearmanrho_cases():
    c = []
    for n in (2, 3, 10, 40):
        x, y = corr_pair(n, 0.5)
        c.append(call('spearmanrho', [x, y], names=SIG))
    x, y = corr_pair(30, 0.5)
    for a in ALTS:
        c.append(call('spearmanrho', [x, y], {'alternative': a}, names=SIG))
    c.append(call('spearmanrho', [ties(30), ties(30, 6)], names=SIG))
    c.append(call('spearmanrho', [np.ones(5), y[:5]], names=SIG))
    c += paired_axis_cases('spearmanrho', SIG)
    return c


def pointbiserial_cases():
    c = []
    for n in (2, 3, 10, 40):
        b = rng.integers(0, 2, size=n).astype(float)
        yv = b + rng.normal(size=n)
        c.append(call('pointbiserialr', [b, yv], names=SIG))
    # SciPy's test (test_stats.test_pointbiserial)
    x = [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    y = [14.8, 13.8, 12.4, 10.1, 7.1, 6.1, 5.8, 4.6, 4.3, 3.5, 3.3, 3.2, 3.0, 2.8, 2.8, 2.5, 2.4, 2.3, 2.1, 1.7, 1.7,
         1.5, 1.3, 1.3, 1.2, 1.2, 1.1, 0.8, 0.7, 0.6, 0.5, 0.2, 0.2, 0.1]
    c.append(call('pointbiserialr', [np.array(x, float), np.array(y)], names=SIG))
    c += paired_axis_cases('pointbiserialr', SIG)
    return c


def somersd_cases():
    c = []
    x = np.array([1, 2, 3, 4, 5, 6, 7, 2, 3.0]); y = np.array([2, 2, 1, 4, 5, 5, 7, 1, 3.0])
    for a in ALTS:
        c.append(call('somersd', [x, y], {'alternative': a}, names=SOMERS))
        c.append(call('somersd', [y, x], {'alternative': a}, names=SOMERS))
    table = np.array([[8, 7, 4], [5, 12, 6], [2, 9, 21]])       # an integer contingency table
    for a in ALTS:
        c.append(call('somersd', [table], {'alternative': a}, names=SOMERS))
    c.append(call('somersd', [table.astype(float)], names=SOMERS))
    c.append(call('somersd', [np.ascontiguousarray(table.T)], names=SOMERS))
    c.append(call('somersd', [np.array([[3, 0], [0, 4]])], names=SOMERS))
    c.append(call('somersd', [np.array([[1, 2, 3]])], names=SOMERS))
    c.append(call('somersd', [ties(40, 5), ties(40, 3)], names=SOMERS))
    c.append(call('somersd', [data(25, ints=True), data(25, ints=True)], names=SOMERS))
    c.append(call('somersd', [x, y[:5]], names=SOMERS))                                 # errors
    c.append(call('somersd', [np.array([[1, -2], [3, 4]])], names=SOMERS))
    c.append(call('somersd', [np.array([[1.5, 2], [3, 4]])], names=SOMERS))
    c.append(call('somersd', [np.array([[0, 0], [0, 4]])], names=SOMERS))
    return c


def binned_cases():
    c = []
    x = rng.uniform(0, 10, size=60)
    v = np.sin(x) + rng.normal(size=60)
    for s in ('mean', 'median', 'count', 'sum', 'std', 'min', 'max'):
        c.append(call('binned_statistic', [x, v], {'statistic': s, 'bins': 7}, names=B1))
    c.append(call('binned_statistic', [x, v], names=B1))
    c.append(call('binned_statistic', [x, v], {'bins': np.array([0.0, 1, 2.5, 5, 7.5, 10])}, names=B1))
    c.append(call('binned_statistic', [x, v], {'bins': 5, 'range': [2.0, 8.0]}, names=B1))
    c.append(call('binned_statistic', [x, np.vstack([v, 2 * v + 1])], {'statistic': 'mean', 'bins': 4}, names=B1))
    c.append(call('binned_statistic', [x, np.vstack([v, v * v])], {'statistic': 'median', 'bins': 4}, names=B1))
    c.append(call('binned_statistic', [ties(30, 6), ties(30, 5)], {'statistic': 'median', 'bins': 3}, names=B1))
    xe = np.array([0.0, 1, 2, 3, 4, 5]); ve = np.arange(6.0)
    c.append(call('binned_statistic', [xe, ve], {'statistic': 'count', 'bins': 5}, names=B1))  # right edge
    vn = v.copy(); vn[[3, 10, 20]] = np.nan
    for s in ('mean', 'min', 'max', 'sum'):
        c.append(call('binned_statistic', [x, vn], {'statistic': s, 'bins': 6}, names=B1))
    xn = x.copy(); xn[0] = np.nan
    c.append(call('binned_statistic', [xn, v], {'bins': 5}, names=B1))                  # error: non-finite
    c.append(call('binned_statistic', [x, v[:10]], names=B1))                          # error
    c.append(call('binned_statistic', [data(20, ints=True), data(20, ints=True)], {'statistic': 'sum', 'bins': 3}, names=B1))
    c.append(call('binned_statistic', [x, None], {'statistic': 'count', 'bins': 4}, names=B1))
    c.append(call('binned_statistic', [x, v], {'bins': np.array([3])}, names=B1))
    y = rng.uniform(-1, 1, size=60)
    for s in ('mean', 'count', 'std', 'median', 'max'):
        c.append(call('binned_statistic_2d', [x, y, v], {'statistic': s, 'bins': 4}, names=B2))
    c.append(call('binned_statistic_2d', [x, y, v], {'bins': [3, 5]}, names=B2))
    c.append(call('binned_statistic_2d', [x, y, v], {'bins': [3, 5], 'expand_binnumbers': True}, names=B2))
    c.append(call('binned_statistic_2d', [x, y, v], {'bins': 3, 'range': [[0.0, 10.0], [-1.0, 1.0]]}, names=B2))
    c.append(call('binned_statistic_2d', [x, y, v], {'bins': np.array([[0.0, 2, 5, 10], [-1, -0.5, 0, 1]])}, names=B2))
    c.append(call('binned_statistic_2d', [x, y, np.vstack([v, v + 1])], {'statistic': 'sum', 'bins': 3}, names=B2))
    S = rng.uniform(0, 1, size=(50, 3))
    vv = rng.normal(size=50)
    blk = [call_dd([S, vv], {'statistic': s, 'bins': 3}) for s in ('mean', 'count', 'median', 'min')]
    blk.append(call_dd([S, vv], {'bins': np.array([3, 3, 3]), 'expand_binnumbers': True}))
    blk.append(call_dd([S, vv], {'bins': np.array([[0.0, 0.3, 0.6, 1.0]] * 3)}))
    blk.append(call_dd([S, vv], {'bins': 2, 'range': [[0.0, 1.0]] * 3, 'statistic': 'std'}))
    blk.append(call_dd([S[:, 0], vv], {'bins': 4}))
    return c, blk


def quantile_cases():
    c = []
    methods = ['inverted_cdf', 'averaged_inverted_cdf', 'closest_observation', 'hazen', 'interpolated_inverted_cdf',
               'linear', 'median_unbiased', 'normal_unbiased', 'weibull', 'harrell-davis', '_lower', '_midpoint',
               '_higher', '_nearest', 'round_outward', 'round_inward', 'round_nearest']
    x = u(11)
    p = np.array([0.0, 0.1, 0.25, 0.5, 0.73, 0.9, 1.0])
    for m in methods:
        c.append(call('quantile', [x, p], {'method': m}))
        c.append(call('quantile', [x[:4], np.array([0.2, 0.5, 0.95])], {'method': m}))
    c.append(call('quantile', [x, 0.3]))
    c.append(call('quantile', [x, np.array([0.3])]))
    c.append(call('quantile', [x, np.array([0.3])], {'keepdims': False}))
    c.append(call('quantile', [x, p], {'keepdims': False}))                             # error
    c.append(call('quantile', [x, np.array([-0.1, 0.5, 1.2, np.nan])]))
    X = u((5, 9))
    for ax in (0, 1, -1, None):
        c.append(call('quantile', [X, 0.4], {'axis': ax}))
        c.append(call('quantile', [X, np.array([0.1, 0.6])], {'axis': ax}))
        c.append(call('quantile', [X, np.array([[0.1], [0.6]])], {'axis': ax}))
    c.append(call('quantile', [X, 0.4], {'axis': 0, 'keepdims': True}))
    c.append(call('quantile', [X, 0.4], {'axis': None, 'keepdims': True}))
    c.append(call('quantile', [X, np.array([[0.2], [0.5], [0.7], [0.9], [0.1]])], {'axis': 1}))
    c.append(call('quantile', [X, np.array([[0.2, 0.8]])], {'axis': 0, 'method': 'harrell-davis'}))
    c.append(call('quantile', [X, np.array([0.2, 0.8])], {'axis': 1, 'method': 'harrell-davis'}))
    c.append(call('quantile', [X, np.array([[0.2], [0.8]])], {'axis': 0, 'method': 'harrell-davis'}))
    Xh = X.copy(); Xh[0, 1] = np.nan; Xh[2, 5] = np.nan
    c.append(call('quantile', [Xh, np.array([0.3, 0.8])], {'axis': 1, 'nan_policy': 'omit', 'method': 'harrell-davis'}))
    c.append(call('quantile', [Xh, np.array([[0.3], [0.8]])], {'axis': 0, 'nan_policy': 'omit', 'method': 'harrell-davis'}))
    X5 = u((5, 1))
    c.append(call('quantile', [X5, np.array([[0.3], [0.8]])], {'axis': 0, 'method': 'harrell-davis'}))
    X40 = u((40, 3))
    c.append(call('quantile', [X40, np.array([[0.3], [0.8]])], {'axis': 0, 'method': 'harrell-davis'}))
    c.append(call('quantile', [np.ascontiguousarray(X40.T), np.array([0.3, 0.8])], {'axis': 1, 'method': 'harrell-davis'}))
    Xn = X.copy(); Xn[1, 2] = np.nan; Xn[3, :] = np.nan
    for pol in ('propagate', 'omit', 'raise'):
        c.append(call('quantile', [Xn, 0.5], {'axis': 1, 'nan_policy': pol}))
        c.append(call('quantile', [Xn, np.array([0.3, 0.8])], {'axis': 1, 'nan_policy': pol, 'method': 'harrell-davis'}))
    w = rng.integers(0, 4, size=11).astype(float)
    for m in methods[:9]:
        c.append(call('quantile', [x, p], {'method': m, 'weights': w}))
    c.append(call('quantile', [X, np.array([0.25, 0.75])], {'axis': 1, 'weights': rng.integers(1, 3, size=(5, 9)).astype(float)}))
    c.append(call('quantile', [x, p], {'method': 'harrell-davis', 'weights': w}))       # error
    c.append(call('quantile', [x, 0.5], {'method': 'foo'}))                             # error
    c.append(call('quantile', [data(10, ints=True), np.array([0.2, 0.5])]))
    c.append(call('quantile', [np.zeros((3, 0)), 0.5], {'axis': 1}))
    return c


def boxcox_cases():
    c = []
    x = rng.uniform(0.1, 20, size=12)
    for lm in (0.0, 0.5, -1.0, 2.0, 1e-10):
        c.append(call('boxcox', [x, lm]))
    c.append(call('boxcox', [x.reshape(3, 4), np.array([0.5, 1.0, -0.5, 0.0])]))
    c.append(call('boxcox', [3.0, 0.25]))
    c.append(call('boxcox', [data(6, ints=True) + 6, 0.5]))
    c.append(call('boxcox', [np.array([-1.0, 0.0, 2.0]), 0.5]))
    return c


def yeojohnson_cases():
    c = []
    z = rng.uniform(-5, 5, size=12)
    for lm in (0.0, 2.0, 0.5, -1.0, 3.0):
        c.append(call('yeojohnson', [z, lm]))
    c.append(call('yeojohnson', [z.reshape(3, 4), 1.5]))
    c.append(call('yeojohnson', [data(8, ints=True), 0.7]))
    c.append(call('yeojohnson', [-2.5, 0.3]))
    return c


def main():
    binned, dd = binned_cases()
    blocks = {
        'pearsonr': pearsonr_cases(),
        'spearmanr': spearmanr_cases(),
        'spearmanrho': spearmanrho_cases(),
        'kendalltau': kendall_cases(),
        'weightedtau': weightedtau_cases(),
        'pointbiserialr': pointbiserial_cases(),
        'chatterjeexi': chatterjee_cases(),
        'linregress': linregress_cases(),
        'theilslopes': theil_cases(),
        'siegelslopes': siegel_cases(),
        'somersd': somersd_cases(),
    }
    blocks['binned_statistic'] = [cc for cc in binned if len(cc['args']) == 2]
    blocks['binned_statistic_2d'] = [cc for cc in binned if len(cc['args']) == 3]
    blocks['binned_statistic_dd'] = dd
    blocks['quantile'] = quantile_cases()
    blocks['boxcox'] = boxcox_cases()
    blocks['yeojohnson'] = yeojohnson_cases()
    write('corr', blocks)


if __name__ == '__main__':
    main()
