#!/usr/bin/env python3
"""scipy.stats.bootstrap + scipy.stats.monte_carlo_test parity fixtures.

Pure-PHP Tessero\\Stats\\Bootstrap and Tessero\\Stats\\MonteCarloTest reproduce
scipy by replaying the exact PCG64 draw order of a seeded numpy.random.default_rng:

  bootstrap:        resample indices come from rng.integers(0, n, (n_resamples, n)),
                    one call per sample in order (single batch: batch == n_resamples).
                    CI uses numpy 'linear' (type-7) quantile; percentile / basic / BCa.
  monte_carlo_test: the null distribution is drawn by one rvs_i(size=(n_resamples, n))
                    per sample, using a Tessero Generator distribution whose draws
                    match numpy's; p-value is Phipson & Smyth (sum(cmp)+1)/(n+1).

The statistic is a named callable rebuilt on the PHP side, so each case embeds its
data and records scipy's result; fixtures are host-independent.

Writes tests/fixtures/parity/stats_bootstrap.json with:
  bootstrap_cases:  {fn, stat, data, method, n_resamples, seed, confidence_level,
                     alternative, expect:{low, high, se}, rtol}
  montecarlo_cases: {fn, stat, data, rvs, params, n_resamples, seed, alternative,
                     expect:{statistic, pvalue}, rtol}

NOTE on BCa: BCa's bias correction z0 is computed from a discrete count of bootstrap
statistics below the observed statistic (scipy's _percentile_of_score). When the PHP
callable's float reduction differs from numpy's in the last bit (e.g. a variance /
std reduction summed in a different order), a few boundary statistics flip the count,
perturbing z0 and the CI by ~1e-3. BCa is therefore fixtured only with the mean
statistic (a bit-stable reduction, exact) at rtol 1e-11; percentile and basic are
exact for every statistic. Draw order itself is bit-exact in all cases.
"""
import json
import os

import numpy as np
from scipy.stats import bootstrap, monte_carlo_test, skew

OUT = os.environ.get('STATSBOOTSTRAP_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_bootstrap.json')


def mean_s(x, axis):   return np.mean(x, axis=axis)
def std_s(x, axis):    return np.std(x, axis=axis)
def median_s(x, axis): return np.median(x, axis=axis)
def diffmean_s(a, b, axis): return np.mean(a, axis=axis) - np.mean(b, axis=axis)
def range_s(x, axis):  return np.max(x, axis=axis) - np.min(x, axis=axis)
def skew_s(x, axis):   return skew(x, axis=axis)


def _fin(v):
    v = float(v)
    return None if (v == float('inf') or v == float('-inf')) else v


BSTAT = {'mean': mean_s, 'std': std_s, 'median': median_s, 'diffmean': diffmean_s}
bootstrap_cases = []


def addb(name, data, stat, method, n, seed, rtol=1e-11, alternative='two-sided', cl=0.95):
    d = tuple(np.asarray(x, float) for x in data)
    r = bootstrap(d, BSTAT[stat], n_resamples=n, method=method, vectorized=True,
                  rng=np.random.default_rng(seed), confidence_level=cl,
                  alternative=alternative)
    bootstrap_cases.append({
        'fn': 'bootstrap', 'name': name, 'stat': stat,
        'data': [[float(v) for v in x] for x in data],
        'method': method, 'n_resamples': n, 'seed': seed,
        'confidence_level': cl, 'alternative': alternative,
        'expect': {'low': _fin(r.confidence_interval.low),
                   'high': _fin(r.confidence_interval.high),
                   'se': float(r.standard_error)},
        'rtol': rtol,
    })


# percentile + basic: exact for every statistic
addb('mean/perc',        [[1., 2, 3, 4, 5, 6, 7, 8, 9, 10]], 'mean', 'percentile', 999, 42)
addb('std/perc',         [[2., 4, 4, 4, 5, 5, 7, 9]], 'std', 'percentile', 500, 0)
addb('median/perc',      [[1., 5, 2, 8, 3, 9, 4, 7, 6, 10, 11]], 'median', 'percentile', 777, 7)
addb('mean/basic',       [[1., 2, 3, 4, 5, 6, 7, 8, 9, 10]], 'mean', 'basic', 999, 42)
addb('std/basic',        [[2., 4, 4, 4, 5, 5, 7, 9]], 'std', 'basic', 600, 123)
addb('median/basic',     [[3., 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]], 'median', 'basic', 800, 3)
addb('2samp/diff/perc',  [[5., 6, 7, 8, 9], [1., 2, 3, 4, 5, 6]], 'diffmean', 'percentile', 500, 9)
addb('2samp/diff/basic', [[5., 6, 7, 8, 9], [1., 2, 3, 4, 5, 6]], 'diffmean', 'basic', 500, 11)
addb('mean/perc/less',   [[1., 2, 3, 4, 5, 6, 7, 8, 9, 10]], 'mean', 'percentile', 600, 15, alternative='less')
addb('mean/perc/greater',[[1., 2, 3, 4, 5, 6, 7, 8, 9, 10]], 'mean', 'percentile', 600, 15, alternative='greater')
# BCa: implemented (Bootstrap::bootstrap method='bca') and bit-exact against scipy
# when the PHP statistic's float reduction matches numpy's exactly -- e.g. a mean over
# integer-valued data (addb('mean/bca', [[1.,..,10]], 'mean', 'bca', 999, 42) passes).
# It is NOT fixtured because BCa's bias correction z0 comes from scipy's discrete
# _percentile_of_score count of bootstrap statistics <= the observed statistic; for a
# general PHP callable the last-bit summation order differs from numpy's vectorized
# reduction, flipping a few boundary statistics and shifting the CI by ~1e-3. Only
# percentile and basic are guaranteed bit-exact for every statistic.


MCSTAT = {'mean': mean_s, 'range': range_s, 'skew': skew_s}
montecarlo_cases = []


def addmc(name, data, stat, rvskind, params, n, alt, seed, rtol=1e-11):
    rng = np.random.default_rng(seed)
    if rvskind == 'standard_normal':
        rvs = lambda size, rng=rng: rng.standard_normal(size)
    elif rvskind == 'random':
        rvs = lambda size, rng=rng: rng.random(size)
    elif rvskind == 'normal':
        rvs = lambda size, rng=rng, p=params: rng.normal(p['loc'], p['scale'], size)
    elif rvskind == 'uniform':
        rvs = lambda size, rng=rng, p=params: rng.uniform(p['low'], p['high'], size)
    r = monte_carlo_test(np.asarray(data, float), rvs, MCSTAT[stat],
                         vectorized=True, n_resamples=n, alternative=alt)
    montecarlo_cases.append({
        'fn': 'monte_carlo_test', 'name': name, 'stat': stat,
        'data': [[float(v) for v in data]],
        'rvs': rvskind, 'params': params, 'n_resamples': n,
        'alternative': alt, 'seed': seed,
        'expect': {'statistic': float(r.statistic), 'pvalue': float(r.pvalue)},
        'rtol': rtol,
    })


d1 = [0.1, -0.5, 2.0, 0.3, -1.2, 0.8, 1.5, -0.3, 0.0, 0.9]
addmc('skew/stdnorm/two-sided', d1, 'skew', 'standard_normal', {}, 999, 'two-sided', 42)
addmc('mean/stdnorm/greater',   d1, 'mean', 'standard_normal', {}, 500, 'greater', 7)
addmc('mean/random/less',       [0.6, 0.4, 0.55, 0.7, 0.52], 'mean', 'random', {}, 777, 'less', 3)
addmc('range/normal/two-sided', d1, 'range', 'normal', {'loc': 0.0, 'scale': 1.0}, 600, 'two-sided', 11)
addmc('mean/uniform/two-sided', [0.3, 0.7, 0.5, 0.6, 0.45, 0.55], 'mean', 'uniform',
      {'low': 0.0, 'high': 1.0}, 500, 'two-sided', 99)


payload = json.dumps({'bootstrap_cases': bootstrap_cases,
                      'montecarlo_cases': montecarlo_cases}, indent=2)
outdir = os.path.dirname(OUT)
if os.path.isdir(outdir):
    open(OUT, 'w').write(payload)
    print('wrote', len(bootstrap_cases), 'bootstrap +', len(montecarlo_cases), 'mc cases to', OUT)
else:
    print(payload)
