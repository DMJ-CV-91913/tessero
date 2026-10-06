#!/usr/bin/env python3
"""scipy.spatial.distance fixtures: pdist, cdist, squareform. The values are plain float64 arithmetic (no BLAS /
AVX512 float64 dispatch), so they are host-independent and this generator does not require the pinned environment."""
import json, os
import numpy as np
import scipy.spatial.distance as dist
import fixtures_np as F  # enc / enc_result

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'distance.json')
rng = np.random.default_rng(20260930)


def pos(shape):
    return rng.uniform(0.5, 10.0, shape)


def call(fn, args, kwargs=None):
    kwargs = kwargs or {}
    f = getattr(dist, fn)
    r = f(*args, **kwargs)
    return {'args': [F.enc(a) for a in args], 'kwargs': {k: F.enc(v) for k, v in kwargs.items()},
            'expect': F.enc_result(r, []), 'compare': 'tol'}


METRICS = ['euclidean', 'sqeuclidean', 'cityblock', 'chebyshev', 'cosine', 'correlation', 'braycurtis', 'canberra', 'hamming']
X = pos((5, 4))
XA, XB = pos((4, 3)), pos((6, 3))

pdist_cases = [call('pdist', [X])]                                    # default euclidean
pdist_cases += [call('pdist', [X], {'metric': m}) for m in METRICS]
pdist_cases += [call('pdist', [X], {'metric': 'minkowski', 'p': 3.0}),
                call('pdist', [rng.uniform(-5, 5, (6, 4))])]          # mixed-sign euclidean

cdist_cases = [call('cdist', [XA, XB])]
cdist_cases += [call('cdist', [XA, XB], {'metric': m}) for m in METRICS]
cdist_cases += [call('cdist', [XA, XB], {'metric': 'minkowski', 'p': 4.0})]

cond = dist.pdist(X)                                                  # a valid condensed vector
sq = dist.squareform(cond)
square_cases = [call('squareform', [cond]), call('squareform', [sq]),
                call('squareform', [dist.pdist(pos((7, 3)))])]

# individual pairwise metric functions distance.<metric>(u, v)
vu, vv = pos(5), pos(5)
vu2, vv2 = rng.uniform(-5, 5, 6), rng.uniform(-5, 5, 6)
metric_blocks = []
for m in METRICS:
    cs = [call(m, [vu, vv]), call(m, [vu2, vv2])]
    if m == 'hamming':
        cs.append(call('hamming', [np.array([1., 2., 3., 4., 5.]), np.array([1., 2., 0., 4., 0.])]))
    metric_blocks.append({'fn': m, 'cases': cs})
metric_blocks.append({'fn': 'minkowski', 'cases': [call('minkowski', [vu, vv, 3.0]), call('minkowski', [vu2, vv2, 4.0]), call('minkowski', [vu, vv, 1.0])]})

# boolean (set) dissimilarity functions
bu = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
bv = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=bool)
bu2 = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
bv2 = np.array([1, 0, 0, 1, 1, 1, 0, 1], dtype=bool)
for m in ['dice', 'jaccard', 'rogerstanimoto', 'russellrao', 'sokalsneath', 'yule']:
    metric_blocks.append({'fn': m, 'cases': [call(m, [bu, bv]), call(m, [bu2, bv2])]})

# metrics with extra args: mahalanobis (inverse covariance VI), seuclidean (variance V), jensenshannon (base)
mu, mv = pos(4), pos(4)
A4 = rng.uniform(-1, 1, (4, 4)); VI4 = A4 @ A4.T + 4 * np.eye(4)        # symmetric positive-definite
A3 = rng.uniform(-1, 1, (3, 3)); VI3 = A3 @ A3.T + 3 * np.eye(3)
metric_blocks.append({'fn': 'mahalanobis', 'cases': [call('mahalanobis', [mu, mv, VI4]), call('mahalanobis', [pos(3), pos(3), VI3])]})
metric_blocks.append({'fn': 'seuclidean', 'cases': [call('seuclidean', [mu, mv, pos(4)]), call('seuclidean', [vu, vv, pos(5)])]})
metric_blocks.append({'fn': 'jensenshannon', 'cases': [call('jensenshannon', [pos(5), pos(5)]), call('jensenshannon', [pos(5), pos(5), 2.0]), call('jensenshannon', [pos(4), pos(4)])]})

# condensed/redundant validation + observation-count helpers
_cvec = dist.pdist(pos((6, 3)))        # a valid condensed vector (len 15 -> 6 obs)
_dm = dist.squareform(_cvec)           # a valid 6x6 distance matrix
metric_blocks.append({'fn': 'num_obs_dm', 'cases': [call('num_obs_dm', [_dm]), call('num_obs_dm', [dist.squareform(dist.pdist(pos((4, 2))))])]})
metric_blocks.append({'fn': 'num_obs_y', 'cases': [call('num_obs_y', [_cvec]), call('num_obs_y', [dist.pdist(pos((5, 2)))])]})
metric_blocks.append({'fn': 'is_valid_dm', 'cases': [call('is_valid_dm', [_dm]), call('is_valid_dm', [np.ones((3, 3))])]})
metric_blocks.append({'fn': 'is_valid_y', 'cases': [call('is_valid_y', [_cvec]), call('is_valid_y', [pos(4)])]})

out = {'module': 'distance', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(),
       'calls': [{'fn': 'pdist', 'cases': pdist_cases}, {'fn': 'cdist', 'cases': cdist_cases},
                 {'fn': 'squareform', 'cases': square_cases}] + metric_blocks}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'distance: pdist {len(pdist_cases)}, cdist {len(cdist_cases)}, squareform {len(square_cases)}, + {len(metric_blocks)} metric fns')
