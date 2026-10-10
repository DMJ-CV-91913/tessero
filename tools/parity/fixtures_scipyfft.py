#!/usr/bin/env python3
"""scipy.fft fixtures: dct/idct/dst/idst for 1-D real input, types 2 and 3, norm='backward' (the default).
Plain trigonometric summations (no BLAS), host-independent; the pinned environment is not required."""
import json, os
import numpy as np
import scipy.fft as sf
import fixtures_np as F  # enc / enc_result

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'scipyfft.json')
rng = np.random.default_rng(20261010)

FN = {'dct': sf.dct, 'idct': sf.idct, 'dst': sf.dst, 'idst': sf.idst}
calls = []
for name, fn in FN.items():
    cases = []
    for n in (4, 5, 8, 9):
        x = rng.normal(size=n)
        for t in (2, 3):
            cases.append({'args': [F.enc(x), F.enc(t)], 'kwargs': {},
                          'expect': F.enc_result(np.asarray(fn(x, type=t), float), []), 'compare': 'tol',
                          'tol': {'atol': 1e-10, 'rtol': 1e-10}})
    calls.append({'fn': name, 'cases': cases})

out = {'module': 'scipyfft', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print('scipyfft: %d transforms, %d cases' % (len(calls), sum(len(c['cases']) for c in calls)))
