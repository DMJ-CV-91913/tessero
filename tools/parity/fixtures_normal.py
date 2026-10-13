#!/usr/bin/env python3
"""scipy.stats.Normal(mu, sigma) fixtures: the pure-PHP Tessero\\Stats\\Normal facade (over the
kernel-backed Stats::norm distribution) verified against the new (SciPy 1.15+) random-variable class
API. Deterministic embedded (mu, sigma) parameter sets and evaluation points; host-independent.

Each row is {dist:'normal', params:[mu,sigma], method, arg, expect}:
  * scalar methods carry a scalar arg (or null for the no-arg summaries), scalar expect;
  * the array rows carry a list arg and a list expect so run-tests.php exercises the broadcast path.
moment rows carry arg=[order, kind]; support rows expect the [a, b] pair; ilogcdf/ilogccdf args are
log-probabilities. kurtosis is the NON-excess (Pearson) kurtosis the RV API returns."""
import json
import os

import numpy as np
from scipy.stats import Normal

OUT = os.environ.get('NORMAL_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_normal.json')

# (mu, sigma) parameter sets
PARAMS = [(0.0, 1.0), (1.5, 2.0), (-2.0, 0.5), (10.0, 3.0)]
# points at which to evaluate the density / cdf family
XS = [-1.5, -0.25, 0.0, 0.7, 2.0]
# probabilities for the inverse family
PS = [0.05, 0.3, 0.5, 0.7, 0.95]


def val(x):
    # encode non-finite as the strings run-tests.php's dec() understands ('inf', '-inf', 'nan'),
    # so the JSON stays strictly standard (PHP json_decode rejects bare Infinity/NaN).
    f = float(np.asarray(x).item())
    if np.isposinf(f):
        return 'inf'
    if np.isneginf(f):
        return '-inf'
    if np.isnan(f):
        return 'nan'
    return f


cases = []


def add(params, method, arg, expect):
    cases.append({'dist': 'normal', 'params': list(params), 'method': method,
                  'arg': arg, 'expect': expect})


for (mu, sigma) in PARAMS:
    n = Normal(mu=mu, sigma=sigma)

    # density / cumulative / survival family, scalar args
    for x in XS:
        add((mu, sigma), 'pdf', x, val(n.pdf(x)))
        add((mu, sigma), 'logpdf', x, val(n.logpdf(x)))
        add((mu, sigma), 'cdf', x, val(n.cdf(x)))
        add((mu, sigma), 'logcdf', x, val(n.logcdf(x)))
        add((mu, sigma), 'ccdf', x, val(n.ccdf(x)))
        add((mu, sigma), 'logccdf', x, val(n.logccdf(x)))

    # inverse family (quantiles), scalar args
    for p in PS:
        add((mu, sigma), 'icdf', p, val(n.icdf(p)))
        add((mu, sigma), 'iccdf', p, val(n.iccdf(p)))
        # ilogcdf / ilogccdf take a log-probability
        y = float(np.log(p))
        add((mu, sigma), 'ilogcdf', y, val(n.ilogcdf(y)))
        add((mu, sigma), 'ilogccdf', y, val(n.ilogccdf(y)))

    # no-arg summaries
    add((mu, sigma), 'mean', None, val(n.mean()))
    add((mu, sigma), 'median', None, val(n.median()))
    add((mu, sigma), 'variance', None, val(n.variance()))
    add((mu, sigma), 'standard_deviation', None, val(n.standard_deviation()))
    add((mu, sigma), 'skewness', None, val(n.skewness()))
    add((mu, sigma), 'kurtosis', None, val(n.kurtosis()))      # NON-excess (Pearson): 3.0
    add((mu, sigma), 'entropy', None, val(n.entropy()))
    add((mu, sigma), 'mode', None, val(n.mode()))

    # support: expect the [a, b] end-point pair
    a, b = n.support()
    add((mu, sigma), 'support', None, [val(a), val(b)])

    # moments: arg = [order, kind]
    for order in (1, 2, 3, 4):
        for kind in ('raw', 'central', 'standardized'):
            add((mu, sigma), 'moment', [order, kind], val(n.moment(order, kind)))

# array (broadcast) rows: vector arg against a single (mu, sigma)
amu, asigma = 1.5, 2.0
an = Normal(mu=amu, sigma=asigma)
add((amu, asigma), 'pdf', XS, [val(v) for v in np.asarray(an.pdf(np.asarray(XS)))])
add((amu, asigma), 'cdf', XS, [val(v) for v in np.asarray(an.cdf(np.asarray(XS)))])
add((amu, asigma), 'ccdf', XS, [val(v) for v in np.asarray(an.ccdf(np.asarray(XS)))])
add((amu, asigma), 'logpdf', XS, [val(v) for v in np.asarray(an.logpdf(np.asarray(XS)))])
add((amu, asigma), 'icdf', PS, [val(v) for v in np.asarray(an.icdf(np.asarray(PS)))])
add((amu, asigma), 'iccdf', PS, [val(v) for v in np.asarray(an.iccdf(np.asarray(PS)))])
ys = [float(np.log(p)) for p in PS]
add((amu, asigma), 'ilogcdf', ys, [val(v) for v in np.asarray(an.ilogcdf(np.asarray(ys)))])

out = {'module': 'stats', 'scipy': __import__('scipy').__version__, 'rvdist_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'stats_normal: {len(cases)} cases')
