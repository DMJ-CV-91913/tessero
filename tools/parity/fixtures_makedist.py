#!/usr/bin/env python3
r"""scipy.stats.make_distribution(dist) fixtures.

make_distribution(stats.X) returns a new (SciPy 1.15+) ContinuousDistribution CLASS; calling it with
the base distribution's shape parameters yields a frozen new-API distribution object whose
pdf/logpdf/cdf/logcdf/ccdf/logccdf/icdf/iccdf/mean/variance/median/support/entropy/moment equal the
wrapped standard (loc 0, scale 1) legacy distribution. Tessero reproduces it with the pure-PHP
Tessero\Stats\MakeDistribution facade over its kernel-backed frozen Distribution (Stats::X), so the
fixtures carry, per case, the base spec {name, shape:{...}}, the method, its argument and scipy's
expected value(s). Matched to rtol 1e-10 by run-tests.php's makedist_cases branch.

Writes tests/fixtures/parity/stats_make_distribution.json.
"""
import json
import os

import numpy as np
import scipy
import scipy.stats as st
from scipy.stats import make_distribution

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


SCALAR_METHODS = ['pdf', 'logpdf', 'cdf', 'logcdf', 'ccdf', 'logccdf']
INV_METHODS = ['icdf', 'iccdf']

# (scipy name, shape kwargs, legacy frozen dist for ppf-based x grid, x-sample points, inv-qs)
BASES = [
    ('norm',    {},               st.norm(),          [-1.2, -0.3, 0.5, 1.4],  [0.05, 0.3, 0.5, 0.7, 0.95]),
    ('expon',   {},               st.expon(),         [0.2, 0.8, 1.5, 3.0],    [0.05, 0.3, 0.5, 0.7, 0.95]),
    ('uniform', {},               st.uniform(),       [0.1, 0.4, 0.6, 0.9],    [0.05, 0.3, 0.5, 0.7, 0.95]),
    ('cauchy',  {},               st.cauchy(),        [-2.0, -0.5, 0.5, 2.0],  [0.1, 0.3, 0.5, 0.7, 0.9]),
    ('laplace', {},               st.laplace(),       [-1.5, -0.4, 0.4, 1.5],  [0.05, 0.3, 0.5, 0.7, 0.95]),
    ('gamma',   {'a': 2.5},       st.gamma(2.5),      [0.5, 1.5, 3.0, 5.0],    [0.05, 0.3, 0.5, 0.7, 0.95]),
    ('chi2',    {'df': 4},        st.chi2(4),         [1.0, 3.0, 6.0, 9.0],    [0.05, 0.3, 0.5, 0.7, 0.95]),
    ('t',       {'df': 5},        st.t(5),            [-1.5, -0.3, 0.3, 1.5],  [0.05, 0.3, 0.5, 0.7, 0.95]),
    ('lognorm', {'s': 0.9},       st.lognorm(0.9),    [0.3, 0.8, 1.5, 3.0],    [0.05, 0.3, 0.5, 0.7, 0.95]),
    ('beta',    {'a': 2.0, 'b': 3.0}, st.beta(2.0, 3.0), [0.15, 0.4, 0.6, 0.85], [0.05, 0.3, 0.5, 0.7, 0.95]),
]

cases = []


def add(name, shape, method, arg, expect):
    cases.append({'name': name, 'shape': shape, 'method': method, 'arg': arg, 'expect': expect})


for name, shape, leg, xs, qs in BASES:
    MD = make_distribution(getattr(st, name))
    Y = MD(**shape)
    for x in xs:
        for m in SCALAR_METHODS:
            add(name, shape, m, float(x), val(getattr(Y, m)(x)))
    for q in qs:
        for m in INV_METHODS:
            add(name, shape, m, float(q), val(getattr(Y, m)(q)))
    add(name, shape, 'mean', None, val(Y.mean()))
    add(name, shape, 'median', None, val(Y.median()))
    add(name, shape, 'variance', None, val(Y.variance()))
    add(name, shape, 'entropy', None, val(Y.entropy()))
    for order in (1, 2, 3):
        # raw (non-central) moment — new-API moment(order) default kind is 'raw'; == legacy .moment(order).
        add(name, shape, 'moment', order, val(Y.moment(order)))
    a, b = Y.support()
    add(name, shape, 'support', None, [val(a), val(b)])
    # broadcast (array-arg) rows
    add(name, shape, 'pdf', [float(v) for v in xs], arr(Y.pdf(np.asarray(xs))))
    add(name, shape, 'cdf', [float(v) for v in xs], arr(Y.cdf(np.asarray(xs))))
    add(name, shape, 'logpdf', [float(v) for v in xs], arr(Y.logpdf(np.asarray(xs))))

out = {'module': 'stats', 'scipy': SCIPY, 'makedist_cases': cases}
path = os.path.join(FIXDIR, 'stats_make_distribution.json')
with open(path, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'stats_make_distribution.json: {len(cases)} cases')
