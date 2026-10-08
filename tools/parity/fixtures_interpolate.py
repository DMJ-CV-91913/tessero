#!/usr/bin/env python3
"""scipy.interpolate.pchip_interpolate fixtures (monotone piecewise-cubic interpolation) evaluated at query
points. Deterministic arithmetic over the sample arrays; generated in the pinned environment for consistency
with the other scipy groups."""
import json, os
import numpy as np
from scipy.interpolate import pchip_interpolate
import fixtures_np as F  # enc / enc_result / fixture_env

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'interpolate.json')
rng = np.random.default_rng(20261008)

pc = []
for n in (6, 9):
    xi = np.sort(rng.uniform(-3.0, 5.0, n))
    xi = np.unique(xi)                      # strictly increasing
    yi = np.sin(xi) + 0.3 * xi
    x = np.linspace(xi[0], xi[-1], 23)
    yv = pchip_interpolate(xi, yi, x)
    pc.append({'args': [F.enc(xi), F.enc(yi), F.enc(x)], 'kwargs': {},
               'expect': F.enc_result(yv, []), 'compare': 'tol', 'tol': {'atol': 1e-9}})
# a monotone-data case (exercises the shape-preserving zero-slope branch)
xi = np.linspace(0.0, 10.0, 8)
yi = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0])
x = np.linspace(0.0, 10.0, 25)
pc.append({'args': [F.enc(xi), F.enc(yi), F.enc(x)], 'kwargs': {},
           'expect': F.enc_result(pchip_interpolate(xi, yi, x), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})

calls = [{'fn': 'pchip_interpolate', 'cases': pc}]
out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate: pchip_interpolate {len(pc)} cases')
