#!/usr/bin/env python3
"""scipy.cluster.hierarchy fixtures: agglomerative linkage matrix Z for every method
(single, complete, average, weighted, ward, centroid, median), from both condensed pairwise
distances (pdist) and raw observation matrices (euclidean). Generated in the pinned environment.

Z = (n-1, 4): [cluster_i, cluster_j, distance, sample_count]. Random continuous observations avoid
distance ties, so the merge order is unambiguous and matches SciPy exactly."""
import json, os
import numpy as np
from scipy.cluster.hierarchy import (linkage, single, complete, average, weighted, ward, centroid, median,
    num_obs_linkage, is_valid_linkage, is_monotonic, is_valid_im, correspond, maxdists, inconsistent,
    maxinconsts, maxRstat, leaves_list, cophenet, to_mlab_linkage, from_mlab_linkage)
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

# ---- routines over an existing linkage matrix Z ----
def arr_case(args, expect):
    return {'args': args, 'kwargs': {}, 'expect': F.enc_result(expect, []), 'compare': 'tol', 'tol': TOL}

def bool_case(args, expect):
    return {'args': args, 'kwargs': {}, 'expect': F.enc_result(bool(expect), []), 'compare': 'tol'}

def int_case(args, expect):
    return {'args': args, 'kwargs': {}, 'expect': F.enc_result(int(expect), []), 'compare': 'tol'}

ZSET = []                                                         # (Z, y) pairs over several sizes/methods
for (n, f, m) in ((6, 3, 'average'), (10, 2, 'single'), (8, 4, 'complete'), (12, 3, 'ward'), (9, 2, 'weighted')):
    X = obs(n, f); y = pdist(X); ZSET.append((linkage(y, method=m), y))

num_obs_c, valid_c, mono_c, validim_c, corr_c = [], [], [], [], []
maxd_c, incon_c, maxinc_c, maxr_c, leaves_c, coph_c, tomlab_c, frommlab_c = [], [], [], [], [], [], [], []

for (Z, y) in ZSET:
    Zf = np.asarray(Z, float)
    num_obs_c.append(int_case([F.enc(Zf)], num_obs_linkage(Z)))
    valid_c.append(bool_case([F.enc(Zf)], is_valid_linkage(Z)))
    mono_c.append(bool_case([F.enc(Zf)], is_monotonic(Z)))
    corr_c.append(bool_case([F.enc(Zf), F.enc(np.asarray(y, float))], correspond(Z, y)))
    maxd_c.append(arr_case([F.enc(Zf)], np.asarray(maxdists(Z), float)))
    leaves_c.append(arr_case([F.enc(Zf)], np.asarray(leaves_list(Z), dtype=np.int64)))
    coph_c.append(arr_case([F.enc(Zf)], np.asarray(cophenet(Z), float)))
    mZ = to_mlab_linkage(Z)
    tomlab_c.append(arr_case([F.enc(Zf)], np.asarray(mZ, float)))
    frommlab_c.append(arr_case([F.enc(np.asarray(mZ, float))], np.asarray(from_mlab_linkage(mZ), float)))
    for d in (1, 2, 3):
        R = inconsistent(Z, d)
        incon_c.append(arr_case([F.enc(Zf), F.enc(d)], np.asarray(R, float)))
    R2 = inconsistent(Z, 2)
    R2f = np.asarray(R2, float)
    validim_c.append(bool_case([F.enc(R2f)], is_valid_im(R2)))
    maxinc_c.append(arr_case([F.enc(Zf), F.enc(R2f)], np.asarray(maxinconsts(Z, R2), float)))
    for i in (0, 1, 2, 3):
        maxr_c.append(arr_case([F.enc(Zf), F.enc(R2f), F.enc(i)], np.asarray(maxRstat(Z, R2, i), float)))

# a few predicate false-cases (SciPy returns False, no exception)
Zbad = np.asarray(ZSET[0][0], float).copy(); Zbad[0, 2] = -1.0            # negative distance
valid_c.append(bool_case([F.enc(Zbad)], is_valid_linkage(Zbad)))
Rbad = np.asarray(inconsistent(ZSET[0][0], 2), float).copy(); Rbad[0, 0] = -1.0
validim_c.append(bool_case([F.enc(Rbad)], is_valid_im(Rbad)))
Zc, yc = ZSET[0]
ymis = np.asarray(pdist(obs(5, 2)), float)                               # wrong number of observations
corr_c.append(bool_case([F.enc(np.asarray(Zc, float)), F.enc(ymis)], correspond(Zc, ymis)))

calls += [
    {'fn': 'num_obs_linkage', 'cases': num_obs_c}, {'fn': 'is_valid_linkage', 'cases': valid_c},
    {'fn': 'is_monotonic', 'cases': mono_c}, {'fn': 'is_valid_im', 'cases': validim_c},
    {'fn': 'correspond', 'cases': corr_c}, {'fn': 'maxdists', 'cases': maxd_c},
    {'fn': 'inconsistent', 'cases': incon_c}, {'fn': 'maxinconsts', 'cases': maxinc_c},
    {'fn': 'maxRstat', 'cases': maxr_c}, {'fn': 'leaves_list', 'cases': leaves_c},
    {'fn': 'cophenet', 'cases': coph_c}, {'fn': 'to_mlab_linkage', 'cases': tomlab_c},
    {'fn': 'from_mlab_linkage', 'cases': frommlab_c},
]

out = {'module': 'hierarchy', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'))
print('hierarchy: linkage %d, aliases %s, +13 Z-routines (%d cases)'
      % (len(link_cases), {m: len(alias_cases[m]) for m in METHODS},
         sum(len(c['cases']) for c in calls[8:])))
