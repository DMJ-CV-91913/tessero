#!/usr/bin/env python3
"""scipy.stats.goodness_of_fit parity fixtures.

Pure-PHP Tessero\\Stats\\GoodnessOfFit reproduces scipy bit-for-bit for the subset
of goodness_of_fit configurations that avoid irreproducible distribution fitting:

  - dist='norm': the only distribution goodness_of_fit fits in closed form
    (loc = mean, scale = std(ddof=1)), so there is no numerical optimizer in the
    Monte Carlo loop. Any combination of known/unknown loc, scale is reproducible.
  - any all-fixed distribution (both params in known_params): no fit step; samples
    are drawn from the fully specified distribution. Fixtured here for 'expon'.

The Monte Carlo null distribution is drawn by one rvs(size=(n_mc, n)) call on the
null-hypothesized distribution, matching scipy.stats.monte_carlo_test's single batch:
  norm  -> loc + scale * default_rng(seed).standard_normal((n_mc, n))
  expon -> loc + scale * default_rng(seed).standard_exponential((n_mc, n))
both of which Tessero's Generator reproduces exactly. Statistics ad/ks/cvm/filliben
are deterministic closed-form; the p-value is scipy's (b+1)/(n_mc+1).

Writes tests/fixtures/parity/stats_goodness_of_fit.json with:
  gof_cases: {name, dist, data, statistic, known_params, n_mc_samples, seed,
              expect:{statistic, pvalue}, rtol}
"""
import json
import os

import numpy as np
from scipy import stats

OUT = os.environ.get('STATSGOF_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_goodness_of_fit.json')

DISTS = {'norm': stats.norm, 'expon': stats.expon}
gof_cases = []


def add(name, dist, data, statistic, known_params, n_mc, seed, rtol=1e-11):
    data = [float(v) for v in data]
    res = stats.goodness_of_fit(
        DISTS[dist], np.asarray(data, float), statistic=statistic,
        known_params=(known_params or None), n_mc_samples=n_mc,
        rng=np.random.default_rng(seed))
    gof_cases.append({
        'name': name, 'dist': dist, 'data': data, 'statistic': statistic,
        'known_params': known_params, 'n_mc_samples': n_mc, 'seed': seed,
        'expect': {'statistic': float(res.statistic), 'pvalue': float(res.pvalue)},
        'rtol': rtol,
    })


# Reproducible sample (its own rng is unrelated to the MC seed).
d_norm = list(stats.skewnorm(4).rvs(40, random_state=np.random.default_rng(99)))
d_norm2 = list(stats.norm(1.0, 2.0).rvs(55, random_state=np.random.default_rng(123)))
d_expon = list(stats.expon(0, 2).rvs(50, random_state=np.random.default_rng(3)))

# norm, fitted (loc & scale free) -- all four statistics
add('norm/ks/fit', 'norm', d_norm, 'ks', {}, 99, 7)
add('norm/cvm/fit', 'norm', d_norm, 'cvm', {}, 99, 7)
add('norm/ad/fit', 'norm', d_norm, 'ad', {}, 99, 7)
add('norm/filliben/fit', 'norm', d_norm, 'filliben', {}, 99, 7)

# norm, one parameter known
add('norm/ks/known-loc', 'norm', d_norm, 'ks', {'loc': 0.5}, 99, 7)
add('norm/cvm/known-scale', 'norm', d_norm, 'cvm', {'scale': 1.2}, 99, 7)
add('norm/ad/known-loc', 'norm', d_norm, 'ad', {'loc': 0.5}, 99, 7)

# norm, both known (all-fixed, no fit)
add('norm/ks/known-both', 'norm', d_norm, 'ks', {'loc': 0.5, 'scale': 1.2}, 99, 7)
add('norm/ad/known-both', 'norm', d_norm2, 'ad', {'loc': 1.0, 'scale': 2.0}, 199, 11)

# larger null, different seed
add('norm/cvm/fit/big', 'norm', d_norm2, 'cvm', {}, 199, 21)
add('norm/filliben/fit/big', 'norm', d_norm2, 'filliben', {}, 199, 33)

# expon, all-fixed (both params known) -- standard_exponential draw
add('expon/cvm/known-both', 'expon', d_expon, 'cvm', {'loc': 0.0, 'scale': 2.0}, 99, 11)
add('expon/ks/known-both', 'expon', d_expon, 'ks', {'loc': 0.0, 'scale': 2.0}, 149, 4)
add('expon/ad/known-both', 'expon', d_expon, 'ad', {'loc': 0.0, 'scale': 2.0}, 149, 8)

payload = json.dumps({'gof_cases': gof_cases}, indent=2, allow_nan=False)
outdir = os.path.dirname(OUT)
if os.path.isdir(outdir):
    open(OUT, 'w').write(payload)
    print('wrote', len(gof_cases), 'goodness_of_fit cases to', OUT)
else:
    print(payload)
