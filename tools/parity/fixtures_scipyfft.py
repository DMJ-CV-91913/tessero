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

# n-D variants over 2-D arrays
FN2 = {'dctn': sf.dctn, 'idctn': sf.idctn, 'dstn': sf.dstn, 'idstn': sf.idstn}
for name, fn in FN2.items():
    cases = []
    for (m, n) in ((3, 4), (5, 5), (4, 6)):
        x = rng.normal(size=(m, n))
        for t in (2, 3):
            cases.append({'args': [F.enc(x), F.enc(t)], 'kwargs': {},
                          'expect': F.enc_result(np.asarray(fn(x, type=t), float), []), 'compare': 'tol',
                          'tol': {'atol': 1e-10, 'rtol': 1e-10}})
    calls.append({'fn': name, 'cases': cases})

# FFTLog: fht / ifht / fhtoffset
_fc, _ic, _oc = [], [], []
for _n in (8, 16, 7, 9):
    _x = rng.random(_n) + 0.1
    for _mu in (0.0, 0.5, 1.5):
        _A = sf.fht(_x, 0.1, _mu)
        _fc.append({'args': [F.enc(_x), F.enc(0.1), F.enc(_mu)], 'kwargs': {},
                    'expect': F.enc_result(np.asarray(_A, float), []), 'compare': 'tol', 'tol': {'atol': 1e-10, 'rtol': 1e-9}})
        _a2 = sf.ifht(_A, 0.1, _mu)
        _ic.append({'args': [F.enc(_A), F.enc(0.1), F.enc(_mu)], 'kwargs': {},
                    'expect': F.enc_result(np.asarray(_a2, float), []), 'compare': 'tol', 'tol': {'atol': 1e-10, 'rtol': 1e-9}})
    _Ab = sf.fht(_x, 0.1, 1.0, bias=0.2)
    _fc.append({'args': [F.enc(_x), F.enc(0.1), F.enc(1.0)], 'kwargs': {'bias': F.enc(0.2)},
                'expect': F.enc_result(np.asarray(_Ab, float), []), 'compare': 'tol', 'tol': {'atol': 1e-10, 'rtol': 1e-9}})
for _dln in (0.1, 0.05):
    for _mu in (0.0, 0.5, 2.0):
        for _bias in (0.0, 0.3):
            _oc.append({'args': [F.enc(_dln), F.enc(_mu)], 'kwargs': {'bias': F.enc(_bias)},
                        'expect': F.enc_result(float(sf.fhtoffset(_dln, _mu, 0.0, _bias)), []), 'compare': 'tol', 'tol': {'atol': 1e-12, 'rtol': 1e-10}})
calls.append({'fn': 'fht', 'cases': _fc})
calls.append({'fn': 'ifht', 'cases': _ic})
calls.append({'fn': 'fhtoffset', 'cases': _oc})

out = {'module': 'scipyfft', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print('scipyfft: %d transforms, %d cases' % (len(calls), sum(len(c['cases']) for c in calls)))
