#!/usr/bin/env python3
"""scipy.cluster.hierarchy fixtures: agglomerative linkage matrix Z for every method
(single, complete, average, weighted, ward, centroid, median), from both condensed pairwise
distances (pdist) and raw observation matrices (euclidean). Generated in the pinned environment.

Z = (n-1, 4): [cluster_i, cluster_j, distance, sample_count]. Random continuous observations avoid
distance ties, so the merge order is unambiguous and matches SciPy exactly."""
import json, os
import numpy as np
from scipy.cluster.hierarchy import linkage, single, complete, average, weighted, ward, centroid, median
from scipy.spatial.distance import pdist
import fixtures_np as F

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'hierarchy.json')
rng = np.random.default_rng(20261010)

METHODS = ['single', 'complete', 'average', 'weighted', 'ward', 'centroid', 'median']
ALIAS = {'single': single, 'complete': complete, 'average': average, 'weighted': weighted,
         'ward': ward, 'centroid': centroid, 'median': median}
TOL = {'atol': 1e-9, 'rtol': 1e-9}

def obs(n, f):
    return rng.normal(size=(n, f)) * rng.uniform(0.7, 3.0, f) + rng.uniform(-2, 2, f)

link_cases = []
alias_cases = {m: [] for m in METHODS}

# linkage(y, method): condensed-distance input for every method, several sizes
for (n, f) in ((6, 3), (10, 2), (8, 4), (12, 3)):
    X = obs(n, f)
    y = pdist(X)
    for m in METHODS:
        Z = linkage(y, method=m)
        link_cases.append({'args': [F.enc(y), F.enc(m)], 'kwargs': {},
                           'expect': F.enc_result(np.asarray(Z, float), []), 'compare': 'tol', 'tol': TOL})

# linkage(X, method): observation-matrix input (euclidean), a few methods/sizes
for (n, f) in ((7, 2), (9, 3)):
    X = obs(n, f)
    for m in ('single', 'complete', 'average', 'ward'):
        Z = linkage(X, method=m)
        link_cases.append({'args': [F.enc(X), F.enc(m)], 'kwargs': {},
                           'expect': F.enc_result(np.asarray(Z, float), []), 'compare': 'tol', 'tol': TOL})

# method aliases: y = condensed distances
for (n, f) in ((6, 3), (9, 2), (11, 4)):
    X = obs(n, f)
    y = pdist(X)
    for m in METHODS:
        Z = ALIAS[m](y)
        alias_cases[m].append({'args': [F.enc(y)], 'kwargs': {},
                               'expect': F.enc_result(np.asarray(Z, float), []), 'compare': 'tol', 'tol': TOL})

calls = [{'fn': 'linkage', 'cases': link_cases}]
for m in METHODS:
    calls.append({'fn': m, 'cases': alias_cases[m]})

out = {'module': 'hierarchy', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'))
print('hierarchy: linkage %d, aliases %s' % (len(link_cases), {m: len(alias_cases[m]) for m in METHODS}))
