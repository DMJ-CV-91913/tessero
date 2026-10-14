#!/usr/bin/env python3
"""scipy.stats.Covariance factory/container and scipy.stats.CensoredData container.

Both are deterministic data/linear-algebra containers (no RNG in the quantities we test):

  * Tessero\\Stats\\Covariance reproduces scipy.stats.Covariance across all four factory methods
    (from_precision / from_cholesky / from_diagonal / from_eigendecomposition). We record
    whiten(x), colorize(x), log_pdet, rank and the explicit covariance for 1-D and 2-D inputs and
    compare to rtol 1e-10 (observed agreement is ~1e-16).

  * Tessero\\Stats\\CensoredData reproduces scipy.stats.CensoredData's public partitioning: the raw
    uncensored / left / right / interval arrays after canonicalisation, plus len() and num_censored().
    These are exact (integer counts / embedded float values), compared bit-for-bit.

All inputs are embedded so the recorded expectation is host-independent.

Writes tests/fixtures/parity/stats_classes.json with:
  covariance_cases: {kind, arg{...}, x1, xm, whiten1, whitenm, colorize1, colorizem,
                     log_pdet, rank, cov, rtol}
  censored_cases:   {ctor, arg{...}, uncensored, left, right, interval, len, num_censored}
"""
import json
import os

import numpy as np
from scipy.stats import Covariance, CensoredData

OUT = os.environ.get('STATSCLASSES_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_classes.json')

rng = np.random.default_rng(31415)

# ---- Covariance ---------------------------------------------------------------------------------

covariance_cases = []


def add_cov(kind, cov_obj, arg, x1, xm, rtol=1e-10):
    covariance_cases.append({
        'kind': kind,
        'arg': arg,
        'x1': [float(v) for v in x1],
        'xm': [[float(v) for v in row] for row in xm],
        'whiten1': [float(v) for v in cov_obj.whiten(x1)],
        'whitenm': [[float(v) for v in row] for row in cov_obj.whiten(xm)],
        'colorize1': [float(v) for v in cov_obj.colorize(x1)],
        'colorizem': [[float(v) for v in row] for row in cov_obj.colorize(xm)],
        'log_pdet': float(cov_obj.log_pdet),
        'rank': int(cov_obj.rank),
        'cov': [[float(v) for v in row] for row in cov_obj.covariance],
        'rtol': rtol,
    })


for seed_n in (3, 4, 5):
    A = rng.standard_normal((seed_n, seed_n))
    A = A @ A.T + seed_n * np.eye(seed_n)
    x1 = rng.standard_normal(seed_n)
    xm = rng.standard_normal((4, seed_n))

    P = np.linalg.inv(A)
    add_cov('precision', Covariance.from_precision(P), {'P': P.tolist()}, x1, xm)

    L = np.linalg.cholesky(A)
    add_cov('cholesky', Covariance.from_cholesky(L), {'L': L.tolist()}, x1, xm)

    d = rng.random(seed_n) + 0.5
    add_cov('diagonal', Covariance.from_diagonal(d), {'d': d.tolist()}, x1, xm)

    w, v = np.linalg.eigh(A)
    add_cov('eigh', Covariance.from_eigendecomposition((w, v)),
            {'w': w.tolist(), 'v': v.tolist()}, x1, xm)

# singular diagonal (one zero eigenvalue) -> supported by from_diagonal / from_eigendecomposition
d0 = np.array([2.0, 0.0, 3.0])
x1 = rng.standard_normal(3)
xm = rng.standard_normal((2, 3))
add_cov('diagonal', Covariance.from_diagonal(d0), {'d': d0.tolist()}, x1, xm)

# ---- CensoredData -------------------------------------------------------------------------------

censored_cases = []


def add_censored(ctor, arg, obj):
    censored_cases.append({
        'ctor': ctor,
        'arg': arg,
        'uncensored': [float(v) for v in obj._uncensored],
        'left': [float(v) for v in obj._left],
        'right': [float(v) for v in obj._right],
        'interval': [[float(v) for v in row] for row in obj._interval],
        'len': len(obj),
        'num_censored': int(obj.num_censored()),
    })


# direct constructor with all four kinds
add_censored('ctor',
             {'uncensored': [1, 1.5], 'left': [0], 'right': [10], 'interval': [[2, 3]]},
             CensoredData(uncensored=[1, 1.5], left=[0], right=[10], interval=[[2, 3]]))

# right_censored factory
add_censored('right_censored',
             {'x': [4, 10, 24, 25], 'censored': [False, False, True, True]},
             CensoredData.right_censored([4, 10, 24, 25], [False, False, True, True]))

# left_censored factory
add_censored('left_censored',
             {'x': [0.12, 0.033, 1e-3, 1e-3], 'censored': [False, False, True, True]},
             CensoredData.left_censored([0.12, 0.033, 1e-3, 1e-3], [False, False, True, True]))

# interval_censored factory
add_censored('interval_censored',
             {'low': [0.5, 2.0, 3.0, 5.5], 'high': [1.0, 2.5, 3.5, 7.0]},
             CensoredData.interval_censored([0.5, 2.0, 3.0, 5.5], [1.0, 2.5, 3.5, 7.0]))

# interval input with +/- inf rows and a degenerate row -> canonicalised to left/right/uncensored.
# Infinities are JSON-encoded as the sentinel strings "-inf"/"inf" in the `arg` (the run-tests branch
# decodes them); every recorded *expectation* array is finite.
inf = float('inf')
add_censored('ctor',
             {'interval': [['-inf', 0], [10, 'inf'], [1, 1], [2, 3], [1.5, 1.5]]},
             CensoredData(interval=[[-inf, 0], [10, inf], [1, 1], [2, 3], [1.5, 1.5]]))

# uncensored-only
add_censored('ctor',
             {'uncensored': [3.0, 4.0, 5.0]},
             CensoredData(uncensored=[3.0, 4.0, 5.0]))

# mix: direct uncensored + interval-derived (ordering check: direct first)
add_censored('ctor',
             {'uncensored': [7.0], 'interval': [[1, 1], [2, 3]]},
             CensoredData(uncensored=[7.0], interval=[[1, 1], [2, 3]]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump({'covariance_cases': covariance_cases, 'censored_cases': censored_cases}, f,
              allow_nan=False, indent=1)
print(f'wrote {len(covariance_cases)} covariance + {len(censored_cases)} censored cases to {OUT}')
