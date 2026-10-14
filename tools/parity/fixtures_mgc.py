#!/usr/bin/env python3
"""scipy.stats.multiscale_graphcorr parity fixtures.

Pure-PHP Tessero\\Stats\\MultiscaleGraphcorr reproduces:
  - the MGC statistic: scipy/stats/_stats.pyx::_local_correlations(global_corr='mgc')
    (column-dense-rank + col-mean centering, cumulative local covariance) then the
    beta-threshold connected-component smoothing from scipy/stats/_mgc.py, and
  - the permutation p-value: for each of `reps` replications scipy seeds one legacy
    numpy.random.RandomState from four uint32 drawn by a top-level MT19937, permutes y,
    and recomputes the statistic. Tessero\\Random\\MT19937 reproduces that MT19937 stream
    bit-for-bit (init_genrand for int seeds, init_by_array for array seeds, legacy
    randint/permutation), so the integer p-value matches exactly.

random_state is passed to scipy as an int (RandomState(seed)); the RandomState(seed)
object form seeds identically. The statistic is deterministic and matches to machine
precision; the p-value (a rational k/(reps+1)) matches exactly.

Writes tests/fixtures/parity/stats_mgc.json with:
  mgc_cases: {name, x, y, compute_distance, reps, seed, is_twosamp,
              expect:{statistic, pvalue}, rtol}
"""
import json
import os
import warnings

import numpy as np
from scipy.stats import multiscale_graphcorr

warnings.filterwarnings("ignore")

OUT = os.environ.get('STATSMGC_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_mgc.json')

cases = []


def add(name, x, y, reps, seed, compute_distance=True, is_twosamp=False, rtol=1e-11):
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    res = multiscale_graphcorr(xa, ya, reps=reps, workers=1,
                               random_state=seed, is_twosamp=is_twosamp)
    cases.append({
        'name': name,
        'x': xa.tolist(),
        'y': ya.tolist(),
        'compute_distance': compute_distance,
        'reps': int(reps),
        'seed': int(seed),
        'is_twosamp': bool(is_twosamp),
        'expect': {'statistic': float(res.statistic), 'pvalue': float(res.pvalue)},
        'rtol': rtol,
    })


# 1-D linear (perfect dependence)
x = np.arange(20.0)
add('linear-1d', x, x.copy(), 100, 1)

# 2-D strong dependence with noise
rng = np.random.RandomState(99)
x2 = rng.normal(size=(15, 2))
y2 = x2 * 2 + rng.normal(size=(15, 2)) * 0.1
add('noisy-2d', x2, y2, 100, 2)

# independent (null)
x3 = rng.normal(size=15)
y3 = rng.normal(size=15)
add('independent', x3, y3, 100, 42)

# quadratic dependence
x4 = np.linspace(-1, 1, 18)
add('quadratic', x4, x4 ** 2, 200, 7)

# sine with trend, larger seed
x5 = np.linspace(0, 3, 14)
y5 = np.sin(x5) + np.linspace(0, 0.5, 14)
add('sine-trend', x5, y5, 150, 1000)

# large int seed (> small ints, still < 2**32)
x6 = np.linspace(0, 2, 16)
add('cubic-bigseed', x6, x6 ** 3, 120, 2147483648 % (1 << 32) or 123456789, )

# distance-matrix input (compute_distance=None in scipy) -> precomputed euclidean
x7 = np.linspace(0, 1, 12).reshape(-1, 1)
y7 = (x7 ** 2)
from scipy.spatial.distance import cdist
dx7 = cdist(x7, x7)
dy7 = cdist(y7, y7)
res7 = multiscale_graphcorr(dx7, dy7, compute_distance=None, reps=100,
                            workers=1, random_state=5)
cases.append({
    'name': 'distmat-input',
    'x': dx7.tolist(),
    'y': dy7.tolist(),
    'compute_distance': False,
    'reps': 100,
    'seed': 5,
    'is_twosamp': False,
    'expect': {'statistic': float(res7.statistic), 'pvalue': float(res7.pvalue)},
    'rtol': 1e-11,
})

payload = json.dumps({'mgc_cases': cases}, indent=2, allow_nan=False)
outdir = os.path.dirname(OUT)
if os.path.isdir(outdir):
    open(OUT, 'w').write(payload)
    print('wrote', len(cases), 'mgc cases to', OUT)
else:
    print(payload)
