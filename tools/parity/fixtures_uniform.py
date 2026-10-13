#!/usr/bin/env python3
"""scipy.stats.Uniform(a, b) fixtures: the pure-PHP Tessero\\Stats\\Uniform facade (over the
kernel-backed Stats::uniform distribution) verified against the new (SciPy 1.15+) random-variable
class API. Deterministic embedded (a, b) parameter sets and evaluation points; host-independent.

Each row is {dist:'uniform', params:[a, b], method, arg, expect} — the SAME schema the
rvdist_cases branch in run-tests.php already consumes for scipy.stats.Normal:
  * scalar methods carry a scalar arg (or null for the no-arg summaries), scalar expect;
  * the array rows carry a list arg and a list expect so run-tests.php exercises the broadcast path.
moment rows carry arg=[order, kind]; support rows expect the [a, b] pair; ilogcdf/ilogccdf args are
log-probabilities. kurtosis is the NON-excess (Pearson) kurtosis the RV API returns (1.8).

PRECISION NOTE: scipy's MODERN Uniform computes kurtosis / the standardized 4th moment by numerical
quadrature, so those drift by ~1e-12 across parameters and are NOT reproducible by a closed-form
facade within the 1e-12 parity tolerance. The expected values here are therefore taken from the
LEGACY frozen distribution scipy.stats.uniform(loc=a, scale=b-a) — which is byte-for-byte the
distribution the facade wraps (Stats::uniform) — mapping each modern method onto its legacy name.
This matches the Normal fixture's approach (whose values happen to be exact)."""
import json
import math
import os

import numpy as np
from scipy.stats import uniform as legacy

OUT = os.environ.get('UNIFORM_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_uniform.json')

# (a, b) = (lower, upper) parameter sets; width = b - a
PARAMS = [(0.0, 1.0), (-2.0, 4.0), (1.5, 2.5), (-5.0, -1.0),
          (0.0, 10.0), (3.0, 3.5), (-0.5, 0.5), (2.0, 8.0)]
# points at which to evaluate the density / cdf family (span inside and outside the supports)
XS = [-1.5, -0.25, 0.0, 0.7, 2.0]
# probabilities for the inverse family
PS = [0.05, 0.3, 0.5, 0.7, 0.95]


def val(x):
    # encode non-finite as the strings run-tests.php's dec() understands ('inf', '-inf', 'nan').
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
    cases.append({'dist': 'uniform', 'params': list(params), 'method': method,
                  'arg': arg, 'expect': expect})


# reusable standardized uniform (unit variance, zero mean): [-sqrt(3), sqrt(3)]
SA = math.sqrt(3.0)


def frozens(a, b):
    w = b - a
    fz = legacy(loc=a, scale=w)                     # the distribution the facade wraps
    central = legacy(loc=-w / 2.0, scale=w)         # zero-mean, same width
    std = legacy(loc=-SA, scale=2.0 * SA)           # unit variance
    return fz, central, std


for (a, b) in PARAMS:
    fz, central, std = frozens(a, b)

    # density / cumulative / survival family, scalar args
    for x in XS:
        add((a, b), 'pdf', x, val(fz.pdf(x)))
        add((a, b), 'logpdf', x, val(fz.logpdf(x)))
        add((a, b), 'cdf', x, val(fz.cdf(x)))
        add((a, b), 'logcdf', x, val(fz.logcdf(x)))
        add((a, b), 'ccdf', x, val(fz.sf(x)))
        add((a, b), 'logccdf', x, val(fz.logsf(x)))

    # inverse family (quantiles), scalar args
    for p in PS:
        add((a, b), 'icdf', p, val(fz.ppf(p)))
        add((a, b), 'iccdf', p, val(fz.isf(p)))
        # ilogcdf / ilogccdf take a log-probability y; facade does ppf(exp(y)) / isf(exp(y))
        y = float(np.log(p))
        add((a, b), 'ilogcdf', y, val(fz.ppf(math.exp(y))))
        add((a, b), 'ilogccdf', y, val(fz.isf(math.exp(y))))

    # no-arg summaries
    add((a, b), 'mean', None, val(fz.mean()))
    add((a, b), 'median', None, val(fz.median()))
    add((a, b), 'variance', None, val(fz.var()))
    add((a, b), 'standard_deviation', None, val(fz.std()))
    add((a, b), 'skewness', None, val(fz.stats(moments='s')))          # 0.0
    add((a, b), 'kurtosis', None, val(float(fz.stats(moments='k')) + 3.0))  # NON-excess: 1.8
    add((a, b), 'entropy', None, val(fz.entropy()))
    add((a, b), 'mode', None, val((a + b) / 2.0))                      # interval midpoint

    # support: expect the [a, b] end-point pair
    lo, hi = fz.support()
    add((a, b), 'support', None, [val(lo), val(hi)])

    # moments: arg = [order, kind]
    for order in (1, 2, 3, 4):
        add((a, b), 'moment', [order, 'raw'], val(fz.moment(order)))
        add((a, b), 'moment', [order, 'central'], val(central.moment(order)))
        add((a, b), 'moment', [order, 'standardized'], val(std.moment(order)))

# array (broadcast) rows: vector arg against a single (a, b)
aa, ab = -2.0, 4.0
afz, _, _ = frozens(aa, ab)
add((aa, ab), 'pdf', XS, [val(v) for v in np.asarray(afz.pdf(np.asarray(XS)))])
add((aa, ab), 'cdf', XS, [val(v) for v in np.asarray(afz.cdf(np.asarray(XS)))])
add((aa, ab), 'ccdf', XS, [val(v) for v in np.asarray(afz.sf(np.asarray(XS)))])
add((aa, ab), 'logpdf', XS, [val(v) for v in np.asarray(afz.logpdf(np.asarray(XS)))])
add((aa, ab), 'icdf', PS, [val(v) for v in np.asarray(afz.ppf(np.asarray(PS)))])
add((aa, ab), 'iccdf', PS, [val(v) for v in np.asarray(afz.isf(np.asarray(PS)))])
ys = [float(np.log(p)) for p in PS]
add((aa, ab), 'ilogcdf', ys, [val(afz.ppf(math.exp(y))) for y in ys])

out = {'module': 'stats', 'scipy': __import__('scipy').__version__, 'rvdist_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'stats_uniform: {len(cases)} cases')
