#!/usr/bin/env python3
"""scipy.stats hypothesis tests, group tests2 (shapiro, anderson, anderson_ksamp, cramervonmises,
cramervonmises_2samp, ks_1samp, kstest, ks_2samp, epps_singleton_2samp, alexandergovern, page_trend_test,
poisson_means_test, combine_pvalues, bws_test, quantile_test) -> tests/fixtures/parity/stats_tests2.json.

Run from tools/parity: python3 fixtures_stats_tests2.py (FIXTURE_OUT=path overrides the output file).
"""
import numpy as np
import scipy.stats as st

from fixtures_stats_fn import call, write, data, rng

SR = ['statistic', 'pvalue']
KS = ['statistic', 'pvalue', 'statistic_location', 'statistic_sign']


def call_cdf(fn, args, kwargs=None, names=None):
    """ks_1samp takes a callable cdf; Tessero takes the distribution's name. The fixture records the name and
    the result of SciPy called with that distribution's cdf method."""
    def wrapper(x, name, *a, **k):
        return getattr(st, fn)(x, getattr(st, name).cdf, *a, **k)
    setattr(st, '_tessero_' + fn, wrapper)
    return call('_tessero_' + fn, args, kwargs, names)


def nanify(a, frac=0.2):
    a = np.array(a, dtype=float)
    m = rng.uniform(size=a.shape) < frac
    a[m] = np.nan
    return a


B = {}

# ---------------------------------------------------------------- shapiro
c = []
for n in (3, 4, 5, 6, 7, 11, 12, 20, 50, 200):
    c.append(call('shapiro', [data(n)], names=SR))
c.append(call('shapiro', [np.exp(data(30, -1, 1))], names=SR))
c.append(call('shapiro', [data(9, ties=True)], names=SR))
c.append(call('shapiro', [data(8, ints=True)], names=SR))
c.append(call('shapiro', [np.array([1.0, 2.0, 4.0])], names=SR))
c.append(call('shapiro', [np.array([1.0, 1.0, 1.0, 1.0])], names=SR))
c.append(call('shapiro', [np.array([1.0, 2.0])], names=SR))
c.append(call('shapiro', [np.array([1.0])], names=SR))
x2 = data((6, 5))
c.append(call('shapiro', [x2], names=SR))
c.append(call('shapiro', [x2], {'axis': 0}, names=SR))
c.append(call('shapiro', [x2], {'axis': 1}, names=SR))
c.append(call('shapiro', [x2], {'axis': -1, 'keepdims': True}, names=SR))
c.append(call('shapiro', [x2], {'keepdims': True}, names=SR))
xn = nanify(data((7, 4)))
for pol in ('propagate', 'omit', 'raise'):
    c.append(call('shapiro', [xn], {'axis': 0, 'nan_policy': pol}, names=SR))
    c.append(call('shapiro', [xn[:, 0]], {'nan_policy': pol}, names=SR))
c.append(call('shapiro', [data((3, 2, 6))], {'axis': 2}, names=SR))
c.append(call('shapiro', [data((3, 2, 6))], {'axis': 5}, names=SR))
B['shapiro'] = c

# ---------------------------------------------------------------- cramervonmises
c = []
for n in (2, 3, 5, 10, 40):
    c.append(call('cramervonmises', [data(n, -2, 2), 'norm'], names=SR))
c.append(call('cramervonmises', [data(25, 0, 3), 'expon'], names=SR))
c.append(call('cramervonmises', [data(25, 0, 3) + 2.1, 'norm'], {'args': (2,)}, names=SR))
c.append(call('cramervonmises', [data(20, 0, 1), 'uniform'], names=SR))
c.append(call('cramervonmises', [data(20, 0.1, 5), 'gamma', (2.0,)], names=SR))
c.append(call('cramervonmises', [data(20, 0.05, 0.95), 'beta', (2.0, 3.0)], names=SR))
c.append(call('cramervonmises', [data(20, -3, 3), 't', (4.0, 0.5, 2.0)], names=SR))
c.append(call('cramervonmises', [data(20, 0.1, 6), 'chi2', (3.0,)], names=SR))
c.append(call('cramervonmises', [data(20, 0.1, 4), 'f', (3.0, 7.0)], names=SR))
c.append(call('cramervonmises', [data(30, 5, 9), 'norm'], names=SR))
c.append(call('cramervonmises', [data(1), 'norm'], names=SR))
x2 = data((8, 3), -2, 2)
c.append(call('cramervonmises', [x2, 'norm'], names=SR))
c.append(call('cramervonmises', [x2, 'norm'], {'axis': 1}, names=SR))
c.append(call('cramervonmises', [x2, 'norm'], {'axis': None, 'keepdims': True}, names=SR))
c.append(call('cramervonmises', [data((1, 4)), 'norm'], names=SR))
c.append(call('cramervonmises', [data((12, 2), -2, 2), 'norm'], names=SR))
xn = nanify(data((9, 3), -2, 2))
for pol in ('propagate', 'omit', 'raise'):
    c.append(call('cramervonmises', [xn, 'norm'], {'nan_policy': pol}, names=SR))
c.append(call('cramervonmises', [data(6, ints=True), 'norm'], names=SR))
B['cramervonmises'] = c

# ---------------------------------------------------------------- cramervonmises_2samp
c = []
for nx, ny in ((2, 2), (3, 4), (5, 7), (7, 6), (10, 10), (20, 15), (21, 5), (30, 40)):
    c.append(call('cramervonmises_2samp', [data(nx), data(ny, -8, 12)], names=SR))
x, y = data(8), data(9)
for m in ('auto', 'exact', 'asymptotic', 'bogus'):
    c.append(call('cramervonmises_2samp', [x, y], {'method': m}, names=SR))
c.append(call('cramervonmises_2samp', [data(9, ties=True), data(7, ties=True)], names=SR))
c.append(call('cramervonmises_2samp', [data(6, ints=True), data(8, ints=True)], names=SR))
c.append(call('cramervonmises_2samp', [data(1), data(5)], names=SR))
c.append(call('cramervonmises_2samp', [data((6, 3)), data((5, 3))], names=SR))
c.append(call('cramervonmises_2samp', [data((3, 6)), data((3, 5))], {'axis': 1, 'keepdims': True}, names=SR))
c.append(call('cramervonmises_2samp', [data((3, 6)), data((1, 5))], {'axis': 1}, names=SR))
c.append(call('cramervonmises_2samp', [data((3, 4)), data(7)], {'axis': None}, names=SR))
xn, yn = nanify(data((8, 2))), nanify(data((9, 2)))
for pol in ('propagate', 'omit', 'raise'):
    c.append(call('cramervonmises_2samp', [xn, yn], {'nan_policy': pol}, names=SR))
B['cramervonmises_2samp'] = c

# ---------------------------------------------------------------- ks_1samp / kstest / ks_2samp
c = []
for n in (1, 2, 3, 5, 10, 30, 100, 150, 400):
    c.append(call_cdf('ks_1samp', [data(n, -2.5, 2.5), 'norm'], names=KS))
x = data(20, -2, 2.5)
for alt in ('two-sided', 'less', 'greater', 'g', 'L', 'bogus'):
    c.append(call_cdf('ks_1samp', [x, 'norm'], {'alternative': alt}, names=KS))
for m in ('auto', 'exact', 'approx', 'asymp'):
    c.append(call_cdf('ks_1samp', [x, 'norm'], {'method': m}, names=KS))
c.append(call_cdf('ks_1samp', [data(40, 0, 2), 'expon'], names=KS))
c.append(call_cdf('ks_1samp', [data(40, 0, 2), 'expon', (0.5, 2.0)], names=KS))
c.append(call_cdf('ks_1samp', [data(30, 0, 1), 'uniform'], names=KS))
c.append(call_cdf('ks_1samp', [data(30, 0.1, 5), 'gamma', (2.0,)], {'alternative': 'less'}, names=KS))
c.append(call_cdf('ks_1samp', [data(30, 0.05, 0.95), 'beta', (2.0, 2.0)], names=KS))
c.append(call_cdf('ks_1samp', [data(30, -3, 3), 't', (3.0,)], names=KS))
c.append(call_cdf('ks_1samp', [data(30, 0.1, 9), 'chi2', (4.0,)], names=KS))
c.append(call_cdf('ks_1samp', [data(30, 0.1, 4), 'f', (5.0, 6.0)], names=KS))
c.append(call_cdf('ks_1samp', [data(60, 3, 6), 'norm'], names=KS))
c.append(call_cdf('ks_1samp', [data(12, -1, 1), 'norm'], names=KS))
c.append(call_cdf('ks_1samp', [data(80, -1.5, 1.5), 'norm'], names=KS))
c.append(call_cdf('ks_1samp', [data(141, -2, 2), 'norm'], names=KS))
c.append(call_cdf('ks_1samp', [data(1000, -2.2, 2.2), 'norm'], names=KS))
c.append(call_cdf('ks_1samp', [data(7, ints=True), 'norm'], names=KS))
x2 = data((6, 4), -2, 2)
c.append(call_cdf('ks_1samp', [x2, 'norm'], names=KS))
c.append(call_cdf('ks_1samp', [x2, 'norm'], {'axis': 1, 'keepdims': True}, names=KS))
c.append(call_cdf('ks_1samp', [x2, 'norm'], {'axis': None}, names=KS))
c.append(call_cdf('ks_1samp', [data((12, 3), -2, 2), 'norm'], names=KS))
xn = nanify(data((8, 3), -2, 2))
for pol in ('propagate', 'omit', 'raise'):
    c.append(call_cdf('ks_1samp', [xn, 'norm'], {'nan_policy': pol}, names=KS))
    c.append(call_cdf('ks_1samp', [xn[:, 0], 'norm'], {'nan_policy': pol}, names=KS))
B['ks_1samp'] = c

c = []
c.append(call('kstest', [data(25, -2, 2), 'norm'], names=KS))
c.append(call('kstest', [data(25, 0, 3), 'expon'], {'alternative': 'greater'}, names=KS))
c.append(call('kstest', [data(25, 0, 3), 'gamma'], {'args': (2.0,), 'alternative': 'less'}, names=KS))
c.append(call('kstest', [data(25, -2, 2), 'norm'], {'alternative': 'two_sided', 'method': 'asymp'}, names=KS))
c.append(call('kstest', [data(25, -2, 2), 'norm'], {'alternative': 'g'}, names=KS))
c.append(call('kstest', [data(12), data(15)], names=KS))
c.append(call('kstest', [data(12), data(15)], {'alternative': 'less'}, names=KS))
c.append(call('kstest', [data(12), data(15)], {'method': 'approx'}, names=KS))
c.append(call('kstest', [data((5, 3)), 'norm'], names=KS))
c.append(call('kstest', [data((5, 3)), data((7, 3))], {'axis': 0, 'keepdims': True}, names=KS))
xn = nanify(data((8, 2), -2, 2))
for pol in ('propagate', 'omit'):
    c.append(call('kstest', [xn, 'norm'], {'nan_policy': pol}, names=KS))
B['kstest'] = c

c = []
for n1, n2 in ((1, 1), (1, 3), (3, 3), (5, 8), (10, 10), (12, 18), (30, 45), (60, 25), (100, 100), (251, 99), (1000, 1200)):
    c.append(call('ks_2samp', [data(n1), data(n2, -6, 14)], names=KS))
x, y = data(15), data(21, -7, 13)
for alt in ('two-sided', 'less', 'greater'):
    for m in ('auto', 'exact', 'asymp'):
        c.append(call('ks_2samp', [x, y], {'alternative': alt, 'method': m}, names=KS))
xe, ye = data(20), data(20, -6, 14)
for alt in ('two-sided', 'less', 'greater'):
    c.append(call('ks_2samp', [xe, ye], {'alternative': alt}, names=KS))
c.append(call('ks_2samp', [x, y], {'method': 'bogus'}, names=KS))
c.append(call('ks_2samp', [data(40, ties=True), data(35, ties=True)], names=KS))
c.append(call('ks_2samp', [data(9, ints=True), data(11, ints=True)], names=KS))
c.append(call('ks_2samp', [data(300), data(450) + 3], {'alternative': 'less', 'method': 'exact'}, names=KS))
c.append(call('ks_2samp', [data(300), data(460) - 3], {'alternative': 'greater'}, names=KS))
c.append(call('ks_2samp', [data((6, 3)), data((9, 3))], names=KS))
c.append(call('ks_2samp', [data((3, 6)), data((3, 9))], {'axis': -1, 'keepdims': True}, names=KS))
xn, yn = nanify(data((10, 2))), nanify(data((12, 2)))
for pol in ('propagate', 'omit', 'raise'):
    c.append(call('ks_2samp', [xn, yn], {'nan_policy': pol}, names=KS))
B['ks_2samp'] = c

# ---------------------------------------------------------------- anderson
AR = ['statistic', 'critical_values', 'significance_level']
c = []
for dist in ('norm', 'expon', 'logistic', 'gumbel_r', 'gumbel_l', 'gumbel', 'extreme1', 'Norm'):
    for n in (5, 12, 40):
        x = data(n, 0.2, 6)
        c.append(call('anderson', [x, dist], names=AR))
        c.append(call('anderson', [x, dist], {'method': 'interpolate'}, names=SR))
c.append(call('anderson', [data(30, 0.1, 3), 'bogus'], names=AR))
c.append(call('anderson', [data(9, ties=True)], names=AR))
c.append(call('anderson', [data(9, ints=True)], {'method': 'interpolate'}, names=SR))
c.append(call('anderson', [np.linspace(-3, 3, 50) ** 3], {'method': 'interpolate'}, names=SR))
c.append(call('anderson', [np.exp(data(30, -1, 1)), 'norm'], {'method': 'interpolate'}, names=SR))
c.append(call('anderson', [data(30, 0, 1) ** 4, 'expon'], {'method': 'interpolate'}, names=SR))
c.append(call('anderson', [data(25, -3, 3), 'logistic'], {'method': 'interpolate'}, names=SR))
c.append(call('anderson', [data(100, -3, 3), 'gumbel_r'], {'method': 'interpolate'}, names=SR))
B['anderson'] = c

# ---------------------------------------------------------------- anderson_ksamp
AK = ['statistic', 'critical_values', 'pvalue']
c = []
for k, m in ((2, 5), (2, 10), (3, 8), (4, 6), (5, 20)):
    s_ = data((k, m))
    c.append(call('anderson_ksamp', [s_], names=AK))
    c.append(call('anderson_ksamp', [s_, False], names=AK))
    for v in ('midrank', 'right', 'continuous'):
        c.append(call('anderson_ksamp', [s_], {'variant': v}, names=SR))
s_ = data((3, 10), ties=True)
c.append(call('anderson_ksamp', [s_], names=AK))
c.append(call('anderson_ksamp', [s_, False], names=AK))
for v in ('midrank', 'right', 'continuous', 'bogus'):
    c.append(call('anderson_ksamp', [s_], {'variant': v}, names=SR))
c.append(call('anderson_ksamp', [np.vstack([data(15), data(15) + 6])], names=AK))
c.append(call('anderson_ksamp', [np.vstack([data(12), data(12) + 1.5])], names=AK))
c.append(call('anderson_ksamp', [np.vstack([data(12), data(12) + 3.0])], names=AK))
c.append(call('anderson_ksamp', [data((3, 12), ints=True)], names=AK))
c.append(call('anderson_ksamp', [data((1, 10))], names=AK))
c.append(call('anderson_ksamp', [np.ones((2, 5))], names=AK))
c.append(call('anderson_ksamp', [data((2, 2))], names=AK))
B['anderson_ksamp'] = c

# ---------------------------------------------------------------- epps_singleton_2samp
def epps_cond(x, y, t=(0.4, 0.8), axis=0):
    """Condition number of the covariance matrix SciPy passes to pinv (the largest over the slices)."""
    x = np.moveaxis(np.asarray(x, float), axis, -1)
    y = np.moveaxis(np.asarray(y, float), axis, -1)
    worst = 1.0
    for a, b in zip(x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])):
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        a, b = np.where(np.isfinite(a), a, 1.0), np.where(np.isfinite(b), b, 1.0)
        if len(a) < 5 or len(b) < 5:
            continue
        n = len(a) + len(b)
        ts = np.asarray(t, float)[:, None] / (st.iqr(np.concatenate([a, b])) / 2)
        gx = np.concatenate([np.cos(ts * a), np.sin(ts * a)])
        gy = np.concatenate([np.cos(ts * b), np.sin(ts * b)])
        cv = n / len(a) * np.cov(gx, bias=True) + n / len(b) * np.cov(gy, bias=True)
        sv = np.linalg.svd(np.atleast_2d(cv), compute_uv=False)
        sv = sv[sv > sv[0] * len(sv) * np.finfo(float).eps]
        worst = max(worst, sv[0] / sv[-1])
    return worst


def epps(args, kwargs=None):
    """The statistic is g' pinv(C) g: a last-bit difference in C (Tessero's SVD and LAPACK's gesdd sum in
    different orders) is amplified by cond(C). The expected value is compared with rtol = max(1e-12, cond(C) eps),
    the first-order perturbation bound of the pseudo-inverse, computed from the call's own data."""
    kwargs = kwargs or {}
    rec = call('epps_singleton_2samp', args, kwargs, names=SR)
    t = kwargs.get('t', (0.4, 0.8))
    if 'error' not in rec['expect'] and all(v > 0 for v in t):
        with np.errstate(all='ignore'):
            r = epps_cond(args[0], args[1], t, kwargs.get('axis', 0)) * np.finfo(float).eps
        if r > 1e-12:
            rec['tol'] = {'rtol': float(r), 'why': 'cond(C) eps: pinv of an ill-conditioned covariance'}
    return rec


c = []
for nx, ny in ((5, 5), (6, 9), (12, 15), (24, 30), (40, 25)):
    c.append(epps([data(nx), data(ny, -5, 15)]))
x, y = data(20), data(22, -4, 12)
c.append(epps([x, y], {'t': (0.3, 0.6, 1.2)}))
c.append(epps([x, y], {'t': (0.5,)}))
c.append(epps([x, y], {'t': (0.5, -1.0)}))
c.append(epps([data(4), data(10)]))
c.append(epps([data(15, ties=True), data(16, ties=True)]))
c.append(epps([data(15, ints=True), data(16, ints=True)]))
xi = data(10); xi[3] = np.inf
c.append(epps([xi, data(10)]))
c.append(epps([data((8, 3)), data((9, 3))]))
c.append(epps([data((3, 8)), data((3, 9))], {'axis': 1, 'keepdims': True}))
xn, yn = nanify(data((10, 2))), nanify(data((12, 2)))
for pol in ('propagate', 'omit', 'raise'):
    c.append(epps([xn, yn], {'nan_policy': pol}))
B['epps_singleton_2samp'] = c

# ---------------------------------------------------------------- alexandergovern
c = []
c.append(call('alexandergovern', [data(5), data(6, -5, 15), data(7, 0, 30)], names=SR))
c.append(call('alexandergovern', [data(10), data(12)], names=SR))
c.append(call('alexandergovern', [data(8), data(9), data(10), data(11), data(8, 3, 4)], names=SR))
c.append(call('alexandergovern', [data(2), data(3)], names=SR))
c.append(call('alexandergovern', [data(1), data(3)], names=SR))
c.append(call('alexandergovern', [np.ones(4), data(5)], names=SR))
c.append(call('alexandergovern', [data(6, ints=True), data(7, ints=True), data(5, ints=True)], names=SR))
c.append(call('alexandergovern', [data(40), data(50, -3, 12), data(45, -10, 8)], names=SR))
c.append(call('alexandergovern', [data((6, 3)), data((7, 3)), data((5, 3))], names=SR))
c.append(call('alexandergovern', [data((3, 6)), data((3, 7))], {'axis': 1, 'keepdims': True}, names=SR))
c.append(call('alexandergovern', [data((3, 6)), data((3, 7))], {'axis': None}, names=SR))
xn, yn = nanify(data((8, 2))), nanify(data((9, 2)))
for pol in ('propagate', 'omit', 'raise'):
    c.append(call('alexandergovern', [xn, yn], {'nan_policy': pol}, names=SR))
B['alexandergovern'] = c

# ---------------------------------------------------------------- combine_pvalues
c = []
pv = data(6, 0.001, 0.9)
for m in ('fisher', 'pearson', 'mudholkar_george', 'tippett', 'stouffer', 'bogus'):
    c.append(call('combine_pvalues', [pv], {'method': m}, names=SR))
c.append(call('combine_pvalues', [pv], {'method': 'stouffer', 'weights': data(6, 0.5, 3)}, names=SR))
c.append(call('combine_pvalues', [data(30, 0.01, 0.99)], names=SR))
c.append(call('combine_pvalues', [np.array([0.01, 0.2, 0.3])], {'method': 'stouffer', 'weights': np.array([1.0, 2.0, 3.0])}, names=SR))
p2 = data((5, 4), 0.01, 0.99)
for m in ('fisher', 'pearson', 'mudholkar_george', 'tippett', 'stouffer'):
    c.append(call('combine_pvalues', [p2], {'method': m}, names=SR))
    c.append(call('combine_pvalues', [p2], {'method': m, 'axis': 1, 'keepdims': True}, names=SR))
c.append(call('combine_pvalues', [data((12, 3), 0.01, 0.99)], names=SR))
c.append(call('combine_pvalues', [p2], {'method': 'stouffer', 'weights': data(5, 0.5, 2)}, names=SR))
c.append(call('combine_pvalues', [p2], {'method': 'stouffer', 'weights': data((5, 4), 0.5, 2), 'axis': None}, names=SR))
pn = nanify(data((7, 3), 0.01, 0.99))
for pol in ('propagate', 'omit', 'raise'):
    c.append(call('combine_pvalues', [pn], {'nan_policy': pol}, names=SR))
    c.append(call('combine_pvalues', [pn[:, 0]], {'method': 'stouffer', 'nan_policy': pol}, names=SR))
c.append(call('combine_pvalues', [np.array([], dtype=float)], names=SR))
B['combine_pvalues'] = c

# ---------------------------------------------------------------- page_trend_test
c = []
for m, n in ((2, 3), (3, 4), (4, 5), (6, 3), (12, 4), (13, 4), (10, 8), (21, 3), (5, 9)):
    c.append(call('page_trend_test', [data((m, n))], names=SR))
d_ = data((5, 4))
for meth in ('exact', 'asymptotic', 'bogus'):
    c.append(call('page_trend_test', [d_], {'method': meth}, names=SR))
c.append(call('page_trend_test', [d_], {'predicted_ranks': [3, 1, 2, 4]}, names=SR))
c.append(call('page_trend_test', [d_], {'predicted_ranks': [3, 1, 1, 4]}, names=SR))
c.append(call('page_trend_test', [data((6, 5), ties=True)], names=SR))
r_ = np.argsort(np.argsort(data((7, 4)), axis=1), axis=1) + 1
c.append(call('page_trend_test', [r_], {'ranked': True}, names=SR))
c.append(call('page_trend_test', [r_.astype(float)], {'ranked': True, 'method': 'asymptotic'}, names=SR))
c.append(call('page_trend_test', [data((7, 4))], {'ranked': True}, names=SR))
c.append(call('page_trend_test', [data((1, 4))], names=SR))
c.append(call('page_trend_test', [data((4, 2))], names=SR))
c.append(call('page_trend_test', [nanify(data((4, 4)), 0.3)], names=SR))
c.append(call('page_trend_test', [data(6)], names=SR))
B['page_trend_test'] = c

# ---------------------------------------------------------------- poisson_means_test
c = []
for args in ((0, 100, 3, 100), (10, 20, 5, 25), (3, 10, 8, 10), (21, 5, 13, 4), (100, 1000, 70, 1000), (0, 1, 0, 1),
             (5, 2.5, 7, 3.5)):
    for alt in ('two-sided', 'less', 'greater'):
        c.append(call('poisson_means_test', list(args), {'alternative': alt}, names=SR))
c.append(call('poisson_means_test', [20, 10, 5, 10], {'diff': 0.5}, names=SR))
c.append(call('poisson_means_test', [5, 10, 20, 10], {'diff': 1.0}, names=SR))
c.append(call('poisson_means_test', [2.5, 10, 20, 10], names=SR))
c.append(call('poisson_means_test', [-1, 10, 20, 10], names=SR))
c.append(call('poisson_means_test', [1, 0, 20, 10], names=SR))
c.append(call('poisson_means_test', [1, 5, 20, 10], {'diff': -1}, names=SR))
c.append(call('poisson_means_test', [1, 5, 20, 10], {'alternative': 'bogus'}, names=SR))
c.append(call('poisson_means_test', [12, 5, 20, 10], {'alternative': 'LESS'}, names=SR))
B['poisson_means_test'] = c

# ---------------------------------------------------------------- quantile_test
QT = ['statistic', 'statistic_type', 'pvalue']
c = []
x = data(25)
for q_, p_ in ((0, 0.5), (2.0, 0.5), (-3.5, 0.25), (5.0, 0.9), (0, 0.01)):
    for alt in ('two-sided', 'less', 'greater'):
        c.append(call('quantile_test', [x], {'q': q_, 'p': p_, 'alternative': alt}, names=QT))
c.append(call('quantile_test', [data(15, ties=True)], {'q': 2.0}, names=QT))
c.append(call('quantile_test', [data(15, ints=True)], {'q': 1}, names=QT))
c.append(call('quantile_test', [data(1)], names=QT))
c.append(call('quantile_test', [x], {'p': 1.0}, names=QT))
c.append(call('quantile_test', [x], {'alternative': 'bogus'}, names=QT))
c.append(call('quantile_test', [data((3, 3))], names=QT))
c.append(call('quantile_test', [np.array([1.0, np.nan, 3.0, -2.0])], {'q': 1.5}, names=QT))
c.append(call('quantile_test', [np.array([-1.0, -2.0, 3.0, 4.0])], {'q': 0, 'p': 0.5}, names=QT))
B['quantile_test'] = c

# ---------------------------------------------------------------- bws_test
BW = ['statistic', 'pvalue', 'null_distribution']
c = []
x_ = np.array([1, 2, 3, 4, 6, 7, 8], dtype=float)
y_ = np.array([5, 9, 10, 11, 12, 13, 14], dtype=float)
for alt in ('two-sided', 'less', 'greater', 'Greater', 'bogus'):
    c.append(call('bws_test', [x_, y_], {'alternative': alt}, names=BW))
for nx, ny in ((1, 1), (2, 3), (4, 4), (5, 7), (6, 8), (3, 12)):
    c.append(call('bws_test', [data(nx), data(ny, -5, 15)], names=BW))
    c.append(call('bws_test', [data(nx), data(ny, -5, 15)], {'alternative': 'less'}, names=BW))
c.append(call('bws_test', [data(6, ties=True), data(6, ties=True)], names=BW))
c.append(call('bws_test', [data(5, ints=True), data(6, ints=True)], names=BW))
c.append(call('bws_test', [np.array([1.0, np.nan]), data(4)], names=BW))
c.append(call('bws_test', [data((2, 3)), data(4)], names=BW))
B['bws_test'] = c

# ---------------------------------------------------------------- N-d layouts (axis in the middle, broadcasting)
B['cramervonmises'].append(call('cramervonmises', [data((2, 12, 3), -2, 2), 'norm'], {'axis': 1}, names=SR))
B['cramervonmises'].append(call('cramervonmises', [data((12, 2), -2, 2), 'norm'], {'axis': 0, 'keepdims': True}, names=SR))
B['ks_1samp'].append(call_cdf('ks_1samp', [data((2, 9, 3), -2, 2), 'norm'], {'axis': 1}, names=KS))
B['ks_2samp'].append(call('ks_2samp', [data((2, 9, 3)), data((1, 11, 3))], {'axis': 1}, names=KS))
B['combine_pvalues'].append(call('combine_pvalues', [data((3, 10, 2), 0.01, 0.99)], {'axis': 1}, names=SR))
B['combine_pvalues'].append(call('combine_pvalues', [data((5, 4), 0.01, 0.99)], {'method': 'stouffer', 'weights': data(4, 0.5, 2), 'axis': 1}, names=SR))
B['combine_pvalues'].append(call('combine_pvalues', [data((5, 4), 0.01, 0.99)], {'method': 'stouffer', 'weights': data((5, 1), 0.5, 2), 'axis': 0}, names=SR))
B['alexandergovern'].append(call('alexandergovern', [data((2, 9, 3)), data((2, 10, 3)), data((1, 8, 3))], {'axis': 1}, names=SR))
B['cramervonmises_2samp'].append(call('cramervonmises_2samp', [data((2, 9, 3)), data((2, 7, 3))], {'axis': 1}, names=SR))
B['epps_singleton_2samp'].append(call('epps_singleton_2samp', [data((2, 9, 2)), data((2, 10, 2))], {'axis': 1}, names=SR))
B['shapiro'].append(call('shapiro', [data((2, 9, 3))], {'axis': 1, 'keepdims': True}, names=SR))

write('tests2', B)
