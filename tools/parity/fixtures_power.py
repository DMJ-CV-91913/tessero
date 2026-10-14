#!/usr/bin/env python3
"""scipy.stats.power parity fixtures.

Pure-PHP Tessero\\Stats\\Power reproduces scipy bit-for-bit by replaying the exact
PCG64 draw order of a seeded numpy.random.default_rng:

  - scipy.stats.power draws each sample with ONE rvs_i(size=(n_resamples, n_obs_i))
    call per rvs callable, in order (single batch: batch defaults to n_resamples).
    The rng is NOT passed into rvs; the rvs closes over a seeded default_rng, so a
    single Generator(seed) drives every draw in sample order. Tessero's Generator
    reproduces standard_normal/normal exactly.
  - the test is a deterministic hypothesis test applied along axis=-1, returning a
    p-value per resample; power = mean(pvalues < significance).

Supported tests: ttest_1samp (two-sided, vs popmean) and ttest_ind (two-sided,
equal-variance / pooled). Both p-values are closed-form via the Student-t CDF
(scipy.special.stdtr), which Tessero exposes.

Writes tests/fixtures/parity/stats_power.json with:
  power_cases: {name, test, rvs:[{kind,params}], n_observations, significance,
                n_resamples, seed, test_kwargs, expect:{power}, rtol}
"""
import json
import os

import numpy as np
from scipy import stats


OUT = os.environ.get('STATSPOWER_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_power.json')

power_cases = []


def _rvs_callable(kind, params, rng):
    if kind == 'standard_normal':
        return lambda size: rng.standard_normal(size)
    if kind == 'normal':
        return lambda size: rng.normal(params['loc'], params['scale'], size)
    raise ValueError(kind)


def add(name, test, rvs_spec, n_obs, significance, n_resamples, seed,
        test_kwargs=None, rtol=1e-12):
    rng = np.random.default_rng(seed)
    rvs = tuple(_rvs_callable(k, p, rng) for (k, p) in rvs_spec)
    rvs_arg = rvs if len(rvs) > 1 else rvs[0]
    n_arg = tuple(n_obs) if len(n_obs) > 1 else n_obs[0]
    tkw = test_kwargs or {}
    if test == 'ttest_1samp':
        testfun = (lambda x, axis, pm=tkw.get('popmean', 0.0):
                   stats.ttest_1samp(x, pm, axis=axis))
    elif test == 'ttest_ind':
        testfun = stats.ttest_ind
    else:
        raise ValueError(test)
    res = stats.power(testfun, rvs_arg, n_arg, significance=significance,
                      n_resamples=n_resamples)
    power_cases.append({
        'name': name, 'test': test,
        'rvs': [{'kind': k, 'params': p} for (k, p) in rvs_spec],
        'n_observations': list(n_obs), 'significance': significance,
        'n_resamples': n_resamples, 'seed': seed, 'test_kwargs': tkw,
        'expect': {'power': float(res.power)}, 'rtol': rtol,
    })


# One-sample t-test power (standard_normal shifted via loc in a 'normal' rvs).
add('ttest_1samp/loc0.5', 'ttest_1samp',
    [('normal', {'loc': 0.5, 'scale': 1.0})], [15], 0.05, 1000, 42,
    {'popmean': 0.0})
add('ttest_1samp/loc0.8/sig0.01', 'ttest_1samp',
    [('normal', {'loc': 0.8, 'scale': 1.0})], [20], 0.01, 500, 7,
    {'popmean': 0.0})
add('ttest_1samp/null', 'ttest_1samp',
    [('standard_normal', {})], [10], 0.05, 777, 3, {'popmean': 0.0})
add('ttest_1samp/loc0.3/big', 'ttest_1samp',
    [('normal', {'loc': 0.3, 'scale': 1.0})], [30], 0.025, 1234, 99,
    {'popmean': 0.0})

# Two-sample independent t-test power.
add('ttest_ind/shift1.0', 'ttest_ind',
    [('standard_normal', {}), ('normal', {'loc': 1.0, 'scale': 1.0})],
    [15, 20], 0.05, 1000, 42)
add('ttest_ind/shift0.5', 'ttest_ind',
    [('standard_normal', {}), ('normal', {'loc': 0.5, 'scale': 1.0})],
    [12, 12], 0.01, 800, 5)
add('ttest_ind/null', 'ttest_ind',
    [('standard_normal', {}), ('standard_normal', {})],
    [25, 25], 0.05, 1000, 17)

payload = json.dumps({'power_cases': power_cases}, indent=2, allow_nan=False)
outdir = os.path.dirname(OUT)
if os.path.isdir(outdir):
    open(OUT, 'w').write(payload)
    print('wrote', len(power_cases), 'power cases to', OUT)
else:
    print(payload)
