#!/usr/bin/env python3
"""scipy.stats.logrank two-sample log-rank survival test.

The pure-PHP Tessero\\LogRank reproduces scipy.stats.logrank: it merges each sample's event
(uncensored) and right-censored times, forms the per-event-time observed/expected/variance table, and
returns the signed standard-normal statistic Z with its p-value (normal SF/CDF for the three
alternatives). All data is embedded in the fixture, so the recorded scipy expectation is
host-independent (no pinned environment required).

Writes tests/fixtures/parity/stats_logrank.json with a `logrank_cases` block:
  {fn:'logrank', x_uncensored, x_right, y_uncensored, y_right, alternative,
   expect:{statistic, pvalue}, rtol}
"""
import json
import os

import numpy as np
import scipy
import scipy.stats as st
from scipy.stats import CensoredData

OUT = os.environ.get('STATSLOGRANK_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_logrank.json')

cases = []


def add(xu, xr, yu, yr, alternative='two-sided', rtol=1e-9):
    X = CensoredData(uncensored=xu, right=xr)
    Y = CensoredData(uncensored=yu, right=yr)
    r = st.logrank(X, Y, alternative=alternative)
    cases.append({
        'fn': 'logrank',
        'x_uncensored': [float(v) for v in xu],
        'x_right': [float(v) for v in xr],
        'y_uncensored': [float(v) for v in yu],
        'y_right': [float(v) for v in yr],
        'alternative': alternative,
        'expect': {'statistic': float(r.statistic), 'pvalue': float(r.pvalue)},
        'rtol': rtol,
    })


# Case 1: the scipy docstring example (Bland & Altman gliomas); all three alternatives.
gx_u = [6, 13, 21, 30, 37, 38, 49, 50, 63, 79, 86, 98, 202, 219]
gx_r = [31, 47, 80, 82, 82, 149]
gy_u = [10, 10, 12, 13, 14, 15, 16, 17, 18, 20, 24, 24, 25, 28, 30, 33, 35, 37,
        40, 40, 46, 48, 76, 81, 82, 91, 112, 181]
gy_r = [34, 40, 70]
for alt in ('two-sided', 'less', 'greater'):
    add(gx_u, gx_r, gy_u, gy_r, alt)

# Case 2: small samples, both censored, ties across groups.
add([2, 3, 5, 7, 11], [4, 9], [1, 2, 6, 8, 10, 12], [5])

# Case 3: one sample with no censoring at all.
add([1, 1, 2, 3, 3, 4, 5], [], [2, 2, 3, 4, 6, 7], [1, 5])

# Case 4: tied event times spanning both groups, mixed censoring.
add([5.0, 8.0, 10.0], [12.0], [6.0, 7.0, 9.0, 11.0], [])

# Cases 5-9: random right-censored survival samples (deterministic seeds).
for seed in range(5):
    rng = np.random.default_rng(1000 + seed)
    nx = int(rng.integers(8, 20))
    ny = int(rng.integers(8, 20))
    tx = rng.exponential(10.0, size=nx)
    ty = rng.exponential(12.0, size=ny)
    # mark ~25% of each as right-censored
    cx = rng.random(nx) < 0.25
    cy = rng.random(ny) < 0.25
    xu = list(np.round(tx[~cx], 3)); xr = list(np.round(tx[cx], 3))
    yu = list(np.round(ty[~cy], 3)); yr = list(np.round(ty[cy], 3))
    if not xu or not yu:
        continue
    add(xu, xr, yu, yr)

out = {'module': 'stats', 'scipy': scipy.__version__, 'logrank_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'stats_logrank: {len(cases)} cases -> {OUT}')
