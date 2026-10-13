#!/usr/bin/env python3
"""scipy.stats.permutation_test parity fixtures (exact full-enumeration path).

Pure-PHP Tessero.PermutationTest reproduces scipy.stats.permutation_test with
n_resamples=inf (exact_test=True, adjustment=0) for the three permutation_type
values. The test statistic is supplied as a named callable (rebuilt on the PHP
side), so every case embeds its data and records scipy statistic/pvalue; the
fixture is host-independent.

Writes tests/fixtures/parity/stats_permtest.json with a permtest_cases block:
  {fn, stat, data, permutation_type, alternative, expect:{statistic, pvalue}, rtol}
"""
import json
import os

import numpy as np
from scipy import stats

OUT = os.environ.get('STATSPERMTEST_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_permtest.json')


def sp_mean_diff(a, b, axis):
    return np.mean(a, axis=axis) - np.mean(b, axis=axis)


def sp_sum_first(*samples, axis):
    return np.sum(samples[0], axis=axis)


def sp_pearson(a, b, axis):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    am = a - a.mean(axis=axis, keepdims=True)
    bm = b - b.mean(axis=axis, keepdims=True)
    num = np.sum(am * bm, axis=axis)
    den = np.sqrt(np.sum(am * am, axis=axis) * np.sum(bm * bm, axis=axis))
    return num / den


def sp_mean_paired(d, axis):
    return np.mean(d, axis=axis)


STATS = {
    'mean_diff': sp_mean_diff,
    'sum_first': sp_sum_first,
    'pearson': sp_pearson,
    'mean_paired': sp_mean_paired,
}

cases = []


def add(stat, data, permutation_type, alternative, rtol=1e-12):
    res = stats.permutation_test(
        [np.asarray(x, float) for x in data], STATS[stat],
        permutation_type=permutation_type, alternative=alternative,
        n_resamples=np.inf)
    cases.append({
        'fn': 'permutation_test',
        'stat': stat,
        'data': [[float(v) for v in x] for x in data],
        'permutation_type': permutation_type,
        'alternative': alternative,
        'expect': {'statistic': float(res.statistic), 'pvalue': float(res.pvalue)},
        'rtol': rtol,
    })


add('mean_diff', [[1., 2., 3.], [4., 5.]], 'independent', 'two-sided')
add('mean_diff', [[1., 2., 3.], [4., 5.]], 'independent', 'less')
add('mean_diff', [[1., 2., 3.], [4., 5.]], 'independent', 'greater')
add('sum_first', [[1., 2.], [3., 4.], [5., 6.]], 'independent', 'two-sided')
add('pearson', [[1., 2., 3., 4.], [2., 1., 4., 3.]], 'pairings', 'two-sided')
add('pearson', [[1., 2., 3., 4.], [2., 1., 4., 3.]], 'pairings', 'greater')
add('mean_paired', [[0.5, -1.2, 2.3, -0.7, 1.1]], 'samples', 'two-sided')
add('mean_paired', [[0.5, -1.2, 2.3, -0.7, 1.1]], 'samples', 'less')
add('mean_paired', [[0.5, -1.2, 2.3, -0.7, 1.1]], 'samples', 'greater')

payload = json.dumps({'permtest_cases': cases}, indent=2)
outdir = os.path.dirname(OUT)
if os.path.isdir(outdir):
    _w = open(OUT, 'w').write(payload)
    print('wrote', len(cases), 'cases to', OUT)
else:
    print(payload)
