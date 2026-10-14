#!/usr/bin/env python3
"""scipy.stats.Logistic() / scipy.stats.Binomial(n, p) fixtures: the pure-PHP Tessero\\Stats\\Logistic
and Tessero\\Stats\\Binomial facades (over the kernel-backed Stats::logistic / Stats::binom
distributions) verified against the new (SciPy 1.15+) random-variable class API. Deterministic
embedded parameter sets and evaluation points; host-independent.

Writes TWO sibling fixtures, both read by run-tests.php's rvdist_cases branch:
  stats_logistic.json  (dist:'logistic', params:[loc, scale])
  stats_binomial.json  (dist:'binomial', params:[n, p])

Each row is {dist, params, method, arg, expect}:
  * scalar methods carry a scalar arg (or null for the no-arg summaries), scalar expect;
  * the array rows carry a list arg and a list expect so run-tests.php exercises the broadcast path.
moment rows carry arg=[order, kind]; support rows expect the [a, b] pair; ilogcdf/ilogccdf args are
log-probabilities. kurtosis is the NON-excess (Pearson) kurtosis the RV API returns.

NOTE on precision: scipy's MODERN Logistic computes kurtosis by numerical quadrature (4.19999999...),
so this stores the ANALYTIC frozen-distribution value (4.2), exactly as the Uniform facade does with
its quadrature-drifting kurtosis. Binomial central / standardized moments are stored from the modern
class but reproduced by the facade via stable direct summation over the support (both ~5e-14)."""
import json
import math
import os

import numpy as np
from scipy.stats import Logistic, Binomial, logistic, binom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIXDIR = os.path.join(ROOT, 'tests', 'fixtures', 'parity')
import scipy
SCIPY = scipy.__version__


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


# =========================== LOGISTIC =============================================================
# scipy's modern Logistic() is the STANDARD logistic; a located / scaled one is loc + scale*Logistic(),
# which equals the legacy logistic(loc, scale) that the Tessero facade wraps. The facade's analytic
# kurtosis (4.2) is used in place of the modern class's quadrature value, so we store leg.stats('k')+3.
LPARAMS = [(0.0, 1.0), (1.5, 2.0), (-2.0, 0.5), (10.0, 3.0),
           (0.0, 0.25), (-5.0, 4.0), (3.3, 1.1), (0.0, 10.0)]
XS = [-1.5, -0.25, 0.0, 0.7, 2.0]
PS = [0.05, 0.3, 0.5, 0.7, 0.95]

lcases = []


def ladd(params, method, arg, expect):
    lcases.append({'dist': 'logistic', 'params': list(params), 'method': method,
                   'arg': arg, 'expect': expect})


for (loc, scale) in LPARAMS:
    M = loc + scale * Logistic()        # modern random-variable class
    leg = logistic(loc, scale)          # legacy == what the facade wraps

    for x in XS:
        ladd((loc, scale), 'pdf', x, val(M.pdf(x)))
        ladd((loc, scale), 'logpdf', x, val(M.logpdf(x)))
        ladd((loc, scale), 'cdf', x, val(M.cdf(x)))
        ladd((loc, scale), 'logcdf', x, val(M.logcdf(x)))
        ladd((loc, scale), 'ccdf', x, val(M.ccdf(x)))
        ladd((loc, scale), 'logccdf', x, val(M.logccdf(x)))

    for p in PS:
        ladd((loc, scale), 'icdf', p, val(M.icdf(p)))
        ladd((loc, scale), 'iccdf', p, val(M.iccdf(p)))
        y = float(np.log(p))
        ladd((loc, scale), 'ilogcdf', y, val(M.ilogcdf(y)))
        ladd((loc, scale), 'ilogccdf', y, val(M.ilogccdf(y)))

    ladd((loc, scale), 'mean', None, val(M.mean()))
    ladd((loc, scale), 'median', None, val(M.median()))
    ladd((loc, scale), 'variance', None, val(M.variance()))
    ladd((loc, scale), 'standard_deviation', None, val(M.standard_deviation()))
    ladd((loc, scale), 'skewness', None, val(M.skewness()))
    # ANALYTIC non-excess kurtosis (4.2) -- the facade value; the modern class drifts by ~1e-12.
    ladd((loc, scale), 'kurtosis', None, val(leg.stats('k') + 3.0))
    ladd((loc, scale), 'entropy', None, val(M.entropy()))
    ladd((loc, scale), 'mode', None, val(M.mode()))

    a, b = M.support()
    ladd((loc, scale), 'support', None, [val(a), val(b)])

    # ANALYTIC moments from the legacy frozen distributions (exactly what the facade computes):
    #   raw = logistic(loc, scale).moment ; central = logistic(0, scale).moment ;
    #   standardized = logistic(0, sqrt(3)/pi).moment (the unit-variance logistic).
    # The MODERN class's 4th moments drift by ~1e-12 (quadrature), so -- like kurtosis -- we store
    # the analytic (frozen) value instead, which the facade reproduces bit-for-bit.
    UNIT = math.sqrt(3.0) / math.pi
    cen_dist = logistic(0.0, scale)
    std_dist = logistic(0.0, UNIT)
    for order in (1, 2, 3, 4):
        ladd((loc, scale), 'moment', [order, 'raw'], val(leg.moment(order)))
        ladd((loc, scale), 'moment', [order, 'central'], val(cen_dist.moment(order)))
        ladd((loc, scale), 'moment', [order, 'standardized'], val(std_dist.moment(order)))

# array (broadcast) rows against a single (loc, scale)
aloc, ascale = 1.5, 2.0
aM = aloc + ascale * Logistic()
ladd((aloc, ascale), 'pdf', XS, [val(v) for v in np.asarray(aM.pdf(np.asarray(XS)))])
ladd((aloc, ascale), 'cdf', XS, [val(v) for v in np.asarray(aM.cdf(np.asarray(XS)))])
ladd((aloc, ascale), 'ccdf', XS, [val(v) for v in np.asarray(aM.ccdf(np.asarray(XS)))])
ladd((aloc, ascale), 'logpdf', XS, [val(v) for v in np.asarray(aM.logpdf(np.asarray(XS)))])
ladd((aloc, ascale), 'icdf', PS, [val(v) for v in np.asarray(aM.icdf(np.asarray(PS)))])
ladd((aloc, ascale), 'iccdf', PS, [val(v) for v in np.asarray(aM.iccdf(np.asarray(PS)))])

lout = {'module': 'stats', 'scipy': SCIPY, 'rvdist_cases': lcases}
with open(os.path.join(FIXDIR, 'stats_logistic.json'), 'w') as f:
    json.dump(lout, f, separators=(',', ':'))
print(f'stats_logistic: {len(lcases)} cases')


# =========================== BINOMIAL =============================================================
# scipy.stats.Binomial(n=n, p=p) == legacy binom(n, p), which the facade wraps. cdf is the kernel's
# regularised incomplete beta; central / standardized moments are reproduced by the facade via stable
# direct summation over the support (matching the modern class to ~5e-14).
BPARAMS = [(10, 0.3), (1, 0.5), (20, 0.7), (5, 0.1),
           (100, 0.5), (50, 0.95), (8, 0.25), (15, 0.6)]
KS = [0, 1, 2, 3, 5]
PS2 = [0.05, 0.3, 0.5, 0.7, 0.95]

bcases = []


def badd(params, method, arg, expect):
    bcases.append({'dist': 'binomial', 'params': list(params), 'method': method,
                   'arg': arg, 'expect': expect})


for (n, p) in BPARAMS:
    M = Binomial(n=n, p=p)

    for k in KS:
        badd((n, p), 'pmf', k, val(M.pmf(k)))
        badd((n, p), 'logpmf', k, val(M.logpmf(k)))
        badd((n, p), 'cdf', k, val(M.cdf(k)))
        badd((n, p), 'logcdf', k, val(M.logcdf(k)))
        badd((n, p), 'ccdf', k, val(M.ccdf(k)))
        badd((n, p), 'logccdf', k, val(M.logccdf(k)))

    for pr in PS2:
        badd((n, p), 'icdf', pr, val(M.icdf(pr)))
        badd((n, p), 'iccdf', pr, val(M.iccdf(pr)))
        y = float(np.log(pr))
        badd((n, p), 'ilogcdf', y, val(M.ilogcdf(y)))
        badd((n, p), 'ilogccdf', y, val(M.ilogccdf(y)))

    badd((n, p), 'mean', None, val(M.mean()))
    badd((n, p), 'median', None, val(M.median()))
    badd((n, p), 'variance', None, val(M.variance()))
    badd((n, p), 'standard_deviation', None, val(M.standard_deviation()))
    badd((n, p), 'skewness', None, val(M.skewness()))
    badd((n, p), 'kurtosis', None, val(M.kurtosis()))      # NON-excess (Pearson)
    badd((n, p), 'entropy', None, val(M.entropy()))
    badd((n, p), 'mode', None, val(M.mode()))

    a, b = M.support()
    badd((n, p), 'support', None, [val(a), val(b)])

    for order in (1, 2, 3, 4):
        for kind in ('raw', 'central', 'standardized'):
            badd((n, p), 'moment', [order, kind], val(M.moment(order, kind)))

# array (broadcast / vectorised arg) rows against a single (n, p)
an, ap = 10, 0.3
aB = Binomial(n=an, p=ap)
badd((an, ap), 'pmf', KS, [val(v) for v in np.asarray(aB.pmf(np.asarray(KS)))])
badd((an, ap), 'logpmf', KS, [val(v) for v in np.asarray(aB.logpmf(np.asarray(KS)))])
badd((an, ap), 'cdf', KS, [val(v) for v in np.asarray(aB.cdf(np.asarray(KS)))])
badd((an, ap), 'ccdf', KS, [val(v) for v in np.asarray(aB.ccdf(np.asarray(KS)))])
badd((an, ap), 'icdf', PS2, [val(v) for v in np.asarray(aB.icdf(np.asarray(PS2)))])
badd((an, ap), 'iccdf', PS2, [val(v) for v in np.asarray(aB.iccdf(np.asarray(PS2)))])

bout = {'module': 'stats', 'scipy': SCIPY, 'rvdist_cases': bcases}
with open(os.path.join(FIXDIR, 'stats_binomial.json'), 'w') as f:
    json.dump(bout, f, separators=(',', ':'))
print(f'stats_binomial: {len(bcases)} cases')
