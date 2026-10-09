#!/usr/bin/env python3
"""scipy.cluster.vq fixtures: whiten (per-feature std normalization) and vq (nearest code-book vector).
Deterministic arithmetic over the observation matrix; generated in the pinned environment for consistency with
the other scipy groups."""
import json, os
import numpy as np
from scipy.cluster.vq import whiten, vq
import fixtures_np as F  # enc / enc_result / fixture_env

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'cluster.json')
rng = np.random.default_rng(20261008)

wht, vqc = [], []

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

out = {'module': 'cluster', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(),
       'calls': [{'fn': 'whiten', 'cases': wht}, {'fn': 'vq', 'cases': vqc}]}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'cluster: whiten {len(wht)}, vq {len(vqc)} cases')
