#!/usr/bin/env python3
"""scipy.interpolate fixtures: 1-D piecewise-cubic interpolators evaluated at query points -- pchip_interpolate,
PchipInterpolator, Akima1DInterpolator, CubicSpline (not-a-knot) and CubicHermiteSpline. Deterministic arithmetic
over the sample arrays; generated in the pinned environment for consistency with the other scipy groups."""
import json, os
import numpy as np
from scipy.interpolate import (pchip_interpolate, PchipInterpolator, Akima1DInterpolator,
                               CubicSpline, CubicHermiteSpline)
import fixtures_np as F  # enc / enc_result / fixture_env

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'interpolate.json')
rng = np.random.default_rng(20261008)

pc, pci, akc, csc, chc = [], [], [], [], []


def xyc(xi, yi, xnew, vals):
    return {'args': [F.enc(xi), F.enc(yi), F.enc(xnew)], 'kwargs': {}, 'expect': F.enc_result(vals, []), 'compare': 'tol', 'tol': {'atol': 1e-9}}


for n in (4, 6, 9):
    xi = np.concatenate([[0.0], np.cumsum(rng.uniform(0.5, 2.0, n - 1))]) - 1.0   # strictly increasing, non-uniform
    yi = np.sin(xi) + 0.3 * xi
    dydx = np.cos(xi) + 0.3
    xnew = np.linspace(xi[0], xi[-1], 23)
    pc.append(xyc(xi, yi, xnew, pchip_interpolate(xi, yi, xnew)))
    pci.append(xyc(xi, yi, xnew, PchipInterpolator(xi, yi)(xnew)))
    akc.append(xyc(xi, yi, xnew, Akima1DInterpolator(xi, yi)(xnew)))
    csc.append(xyc(xi, yi, xnew, CubicSpline(xi, yi)(xnew)))
    chc.append({'args': [F.enc(xi), F.enc(yi), F.enc(dydx), F.enc(xnew)], 'kwargs': {},
                'expect': F.enc_result(CubicHermiteSpline(xi, yi, dydx)(xnew), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# monotone / flat data exercises PCHIP's shape-preserving zero-slope branch
xi = np.linspace(0.0, 10.0, 8); yi = np.array([0., 0., 0., 1., 2., 3., 3., 3.]); xnew = np.linspace(0.0, 10.0, 25)
pc.append(xyc(xi, yi, xnew, pchip_interpolate(xi, yi, xnew)))

# n == 3 exercises CubicSpline's not-a-knot parabola branch
xi = np.array([0.0, 1.3, 2.5]); yi = np.sin(xi) + 0.3 * xi; xnew = np.linspace(0.0, 2.5, 15)
csc.append(xyc(xi, yi, xnew, CubicSpline(xi, yi)(xnew)))

calls = [{'fn': 'pchip_interpolate', 'cases': pc}, {'fn': 'PchipInterpolator', 'cases': pci},
         {'fn': 'Akima1DInterpolator', 'cases': akc}, {'fn': 'CubicSpline', 'cases': csc},
         {'fn': 'CubicHermiteSpline', 'cases': chc}]
out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate: pchip_interpolate {len(pc)}, PchipInterpolator {len(pci)}, Akima1DInterpolator {len(akc)}, '
      f'CubicSpline {len(csc)}, CubicHermiteSpline {len(chc)} cases')
