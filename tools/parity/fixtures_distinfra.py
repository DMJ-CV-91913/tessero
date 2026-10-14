#!/usr/bin/env python3
r"""scipy.stats distribution-infrastructure fixtures: the pure-PHP meta-distribution facades
(Tessero\Stats\Truncated, OrderStatistic, Mixture, Transformed) verified against scipy's new
(SciPy 1.15+) distribution transforms truncate / order_statistic / Mixture / exp / log / abs.

Each facade composes over Tessero's kernel-backed frozen Distribution (Stats::<name>(...)), so the
fixtures carry, per case, a *base spec* describing how run-tests.php should build that base via its
legacy (loc, scale[, shape]) factory, plus the transform's own parameters, the method, its argument
and the scipy-produced expected value(s).

Writes one JSON per symbol into tests/fixtures/parity/, all read by run-tests.php's metadist_cases
branch:
  stats_truncate.json         kind:'truncate'
  stats_order_statistic.json  kind:'order_statistic'
  stats_mixture.json          kind:'mixture'
  stats_exp.json              kind:'exp'
  stats_log.json              kind:'log'
  stats_abs.json              kind:'abs'

Base spec schema: {'name': <Stats factory>, 'params': [<loc/scale/shape args>]}. The transform cases
cover pdf/logpdf/cdf/logcdf/ccdf/logccdf/icdf/iccdf/median/support plus mixture mean/variance, each
with scalar args and a representative broadcast (array-arg) row, matched to rtol 1e-10 (stored exact).
"""
import json
import math
import os

import numpy as np
import scipy
import scipy.stats as st
from scipy.stats import (Normal, Uniform, make_distribution, truncate,
                         order_statistic, Mixture, abs as sabs, exp as sexp, log as slog)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIXDIR = os.path.join(ROOT, 'tests', 'fixtures', 'parity')
SCIPY = scipy.__version__


def val(x):
    f = float(np.asarray(x).item())
    if np.isposinf(f):
        return 'inf'
    if np.isneginf(f):
        return '-inf'
    if np.isnan(f):
        return 'nan'
    return f


def arr(x):
    return [val(v) for v in np.asarray(x).ravel()]


# A base is (scipy-modern RV used to build the transform, Tessero base spec, legacy frozen dist).
# Tessero base spec: Stats::<name>(*params) -> the SAME frozen distribution the modern RV wraps.
def base_normal(mu, sigma):
    return Normal(mu=mu, sigma=sigma), {'name': 'norm', 'params': [mu, sigma]}, st.norm(mu, sigma)


def base_uniform(a, b):
    return Uniform(a=a, b=b), {'name': 'uniform', 'params': [a, b - a]}, st.uniform(a, b - a)


def base_expon(scale):
    # make_distribution(expon)() is the STANDARD exponential; a scaled one is scale * that, which is
    # the legacy expon(0, scale). Tessero: Stats::expon(0, scale).
    E = make_distribution(st.expon)()
    if scale != 1.0:
        E = scale * E
    return E, {'name': 'expon', 'params': [0.0, scale]}, st.expon(0, scale)


SCALAR_METHODS = ['pdf', 'logpdf', 'cdf', 'logcdf', 'ccdf', 'logccdf']
INV_METHODS = ['icdf', 'iccdf']
XS_INNER = None  # set per case


def emit(path, kind, cases):
    out = {'module': 'stats', 'scipy': SCIPY, 'metadist_cases': cases}
    with open(os.path.join(FIXDIR, path), 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'{path}: {len(cases)} cases')


# =========================== TRUNCATE ===========================================================
tcases = []


def tadd(base, lb, ub, method, arg, expect):
    tcases.append({'kind': 'truncate', 'base': base, 'lb': lb, 'ub': ub,
                   'method': method, 'arg': arg, 'expect': expect})


TRUNC = [
    (base_normal(0.0, 1.0), -1.0, 2.0),
    (base_normal(1.5, 2.0), -2.0, 5.0),
    (base_normal(-3.0, 0.5), -4.0, -2.0),
    (base_uniform(0.0, 1.0), 0.2, 0.8),
    (base_expon(1.0), 0.2, 3.0),
    (base_expon(2.5), 0.5, 6.0),
]
for (M, spec, _leg), lb, ub in TRUNC:
    T = truncate(M, lb, ub)
    xs = list(np.linspace(lb + (ub - lb) * 0.05, ub - (ub - lb) * 0.05, 5))
    for x in xs:
        for m in SCALAR_METHODS:
            tadd(spec, lb, ub, m, float(x), val(getattr(T, m)(x)))
    for q in (0.05, 0.3, 0.5, 0.7, 0.95):
        for m in INV_METHODS:
            tadd(spec, lb, ub, m, q, val(getattr(T, m)(q)))
    tadd(spec, lb, ub, 'median', None, val(T.median()))
    a, b = T.support()
    tadd(spec, lb, ub, 'support', None, [val(a), val(b)])
    # one broadcast (array-arg) row
    tadd(spec, lb, ub, 'pdf', [float(v) for v in xs], arr(T.pdf(np.asarray(xs))))
    tadd(spec, lb, ub, 'cdf', [float(v) for v in xs], arr(T.cdf(np.asarray(xs))))
emit('stats_truncate.json', 'truncate', tcases)


# =========================== ORDER_STATISTIC ====================================================
ocases = []


def oadd(base, n, r, method, arg, expect):
    ocases.append({'kind': 'order_statistic', 'base': base, 'n': n, 'r': r,
                   'method': method, 'arg': arg, 'expect': expect})


OSTAT = [
    (base_normal(0.0, 1.0), 5, 3),
    (base_normal(2.0, 1.5), 4, 4),
    (base_normal(0.0, 1.0), 6, 1),
    (base_uniform(0.0, 1.0), 7, 2),
    (base_uniform(0.0, 1.0), 10, 5),
    (base_expon(1.0), 5, 2),
]
for (M, spec, leg), n, r in OSTAT:
    O = order_statistic(M, n=n, r=r)
    xs = list(np.linspace(leg.ppf(0.05), leg.ppf(0.95), 5))
    for x in xs:
        for m in SCALAR_METHODS:
            oadd(spec, n, r, m, float(x), val(getattr(O, m)(x)))
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        for m in INV_METHODS:
            oadd(spec, n, r, m, q, val(getattr(O, m)(q)))
    oadd(spec, n, r, 'median', None, val(O.median()))
    a, b = O.support()
    oadd(spec, n, r, 'support', None, [val(a), val(b)])
    oadd(spec, n, r, 'pdf', [float(v) for v in xs], arr(O.pdf(np.asarray(xs))))
    oadd(spec, n, r, 'cdf', [float(v) for v in xs], arr(O.cdf(np.asarray(xs))))
emit('stats_order_statistic.json', 'order_statistic', ocases)


# =========================== MIXTURE ============================================================
mcases = []


def madd(comps, weights, method, arg, expect):
    mcases.append({'kind': 'mixture', 'components': comps, 'weights': weights,
                   'method': method, 'arg': arg, 'expect': expect})


# Each mixture: list of (modern RV, base spec); weights.
MIX = [
    ([base_normal(-1.0, 1.0), base_normal(2.0, 0.5)], [0.3, 0.7]),
    ([base_normal(0.0, 1.0), base_normal(0.0, 3.0), base_normal(5.0, 1.0)], [0.2, 0.5, 0.3]),
    ([base_uniform(0.0, 1.0), base_uniform(2.0, 4.0)], [0.5, 0.5]),
    ([base_normal(0.0, 1.0), base_expon(1.0)], [0.4, 0.6]),
]
for comps, weights in MIX:
    Mix = Mixture([c[0] for c in comps], weights=weights)
    specs = [c[1] for c in comps]
    xs = list(np.linspace(-3.0, 6.0, 6))
    for x in xs:
        for m in SCALAR_METHODS:
            madd(specs, weights, m, float(x), val(getattr(Mix, m)(x)))
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        for m in INV_METHODS:
            madd(specs, weights, m, q, val(getattr(Mix, m)(q)))
    madd(specs, weights, 'mean', None, val(Mix.mean()))
    madd(specs, weights, 'variance', None, val(Mix.variance()))
    madd(specs, weights, 'median', None, val(Mix.median()))
    madd(specs, weights, 'pdf', [float(v) for v in xs], arr(Mix.pdf(np.asarray(xs))))
    madd(specs, weights, 'cdf', [float(v) for v in xs], arr(Mix.cdf(np.asarray(xs))))
emit('stats_mixture.json', 'mixture', mcases)


# =========================== EXP / LOG / ABS ====================================================
def transform_cases(kind, build, bases, xs_fn):
    cases = []
    for (M, spec, leg) in bases:
        Y = build(M)
        xs = xs_fn(leg)
        for x in xs:
            for m in SCALAR_METHODS:
                cases.append({'kind': kind, 'base': spec, 'method': m, 'arg': float(x),
                              'expect': val(getattr(Y, m)(x))})
        for q in (0.1, 0.3, 0.5, 0.7, 0.9):
            for m in INV_METHODS:
                cases.append({'kind': kind, 'base': spec, 'method': m, 'arg': q,
                              'expect': val(getattr(Y, m)(q))})
        cases.append({'kind': kind, 'base': spec, 'method': 'median', 'arg': None,
                      'expect': val(Y.median())})
        a, b = Y.support()
        cases.append({'kind': kind, 'base': spec, 'method': 'support', 'arg': None,
                      'expect': [val(a), val(b)]})
        cases.append({'kind': kind, 'base': spec, 'method': 'pdf', 'arg': [float(v) for v in xs],
                      'expect': arr(Y.pdf(np.asarray(xs)))})
        cases.append({'kind': kind, 'base': spec, 'method': 'cdf', 'arg': [float(v) for v in xs],
                      'expect': arr(Y.cdf(np.asarray(xs)))})
    return cases


EXP_BASES = [base_normal(0.0, 1.0), base_normal(0.5, 0.8), base_normal(-1.0, 2.0)]
emit('stats_exp.json', 'exp', transform_cases(
    'exp', sexp, EXP_BASES,
    lambda leg: list(np.exp(np.linspace(leg.ppf(0.05), leg.ppf(0.95), 5)))))

LOG_BASES = [base_uniform(0.5, 3.0), base_expon(1.0), base_expon(2.0)]
emit('stats_log.json', 'log', transform_cases(
    'log', slog, LOG_BASES,
    lambda leg: list(np.log(np.linspace(leg.ppf(0.05), leg.ppf(0.95), 5)))))

ABS_BASES = [base_normal(0.0, 1.0), base_normal(0.0, 2.5), base_normal(0.5, 1.0)]
emit('stats_abs.json', 'abs', transform_cases(
    'abs', sabs, ABS_BASES,
    lambda leg: list(np.linspace(0.1, abs(leg.ppf(0.95)) * 1.2, 5))))
