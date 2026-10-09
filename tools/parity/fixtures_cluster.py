#!/usr/bin/env python3
"""scipy.cluster.vq fixtures: whiten (per-feature std normalization) and vq (nearest code-book vector).
Deterministic arithmetic over the observation matrix; generated in the pinned environment for consistency with
the other scipy groups."""
import json, os
import numpy as np
from scipy.cluster.vq import whiten, vq, kmeans2, kmeans
import fixtures_np as F  # enc / enc_result / fixture_env

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'cluster.json')
rng = np.random.default_rng(20261008)

wht, vqc, km2, kmc = [], [], [], []

# whiten: several shapes, including a column with zero variance (left unchanged)
for (n, f) in ((6, 3), (10, 2), (8, 4)):
    obs = rng.normal(size=(n, f)) * rng.uniform(0.5, 4.0, f) + rng.uniform(-2, 2, f)
    wht.append({'args': [F.enc(obs)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(whiten(obs)), []), 'compare': 'tol', 'tol': {'atol': 1e-12}})
obsz = rng.normal(size=(7, 3)); obsz[:, 1] = 2.5          # a constant (zero-std) feature
wht.append({'args': [F.enc(obsz)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(whiten(obsz)), []), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# vq: assign observations to the nearest code-book vector (indices + distances)
for (n, f, k) in ((12, 2, 3), (15, 3, 4), (9, 4, 2)):
    obs = rng.normal(size=(n, f))
    cb = rng.normal(size=(k, f))
    code, dist = vq(obs, cb)
    vqc.append({'args': [F.enc(obs), F.enc(cb)], 'kwargs': {},
                'expect': F.enc_result((np.asarray(code, dtype=np.int64), np.asarray(dist, float)), ['code', 'dist']), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# kmeans2 (minit='matrix', deterministic): Lloyd iterations from explicit initial centroids on well-separated
# blobs (all clusters stay non-empty, assignments stable). Returns centroids and labels.
for (nper, f, k) in ((20, 2, 3), (25, 3, 4), (15, 4, 2)):
    centers = rng.uniform(-10.0, 10.0, (k, f))
    data = np.vstack([centers[i] + 0.4 * rng.normal(size=(nper, f)) for i in range(k)])
    init = centers + 0.5 * rng.normal(size=(k, f))
    cent, lbl = kmeans2(data, init, minit='matrix')
    km2.append({'args': [F.enc(data), F.enc(init), F.enc(10), F.enc(1e-5), F.enc('matrix')], 'kwargs': {},
                'expect': F.enc_result((np.asarray(cent, float), np.asarray(lbl, dtype=np.int64)), ['centroid', 'label']), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# kmeans (explicit guess code book, deterministic): iterate to convergence on well-separated blobs
for (nper, f, k) in ((20, 2, 3), (25, 3, 4), (15, 2, 2)):
    centers = rng.uniform(-10.0, 10.0, (k, f))
    obs = np.vstack([centers[i] + 0.5 * rng.normal(size=(nper, f)) for i in range(k)])
    guess = centers + 0.6 * rng.normal(size=(k, f))
    cbk, dist = kmeans(obs, guess, thresh=1e-5)
    kmc.append({'args': [F.enc(obs), F.enc(guess), F.enc(20), F.enc(1e-5)], 'kwargs': {},
                'expect': F.enc_result((np.asarray(cbk, float), float(dist)), ['codebook', 'distortion']), 'compare': 'tol', 'tol': {'atol': 1e-9}})

out = {'module': 'cluster', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(),
       'calls': [{'fn': 'whiten', 'cases': wht}, {'fn': 'vq', 'cases': vqc}, {'fn': 'kmeans2', 'cases': km2}, {'fn': 'kmeans', 'cases': kmc}]}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'cluster: whiten {len(wht)}, vq {len(vqc)}, kmeans2 {len(km2)}, kmeans {len(kmc)} cases')
