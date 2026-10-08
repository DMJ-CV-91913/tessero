#!/usr/bin/env python3
"""scipy.interpolate fixtures: 1-D piecewise-cubic interpolators evaluated at query points -- pchip_interpolate,
PchipInterpolator, Akima1DInterpolator, CubicSpline (not-a-knot) and CubicHermiteSpline. Deterministic arithmetic
over the sample arrays; generated in the pinned environment for consistency with the other scipy groups."""
import json, os
import numpy as np
from scipy.interpolate import (pchip_interpolate, PchipInterpolator, Akima1DInterpolator,
                               CubicSpline, CubicHermiteSpline,
                               barycentric_interpolate, krogh_interpolate,
                               make_interp_spline, BSpline, pade, splev,
                               splrep, splder, splantider, splint, sproot)
import math
import fixtures_np as F  # enc / enc_result / fixture_env

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'interpolate.json')
rng = np.random.default_rng(20261008)

pc, pci, akc, csc, chc, bcc, kgc, bsp, pdc, mis, spv, sdr, sai, spi, spr = ([] for _ in range(15))


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

# global polynomial interpolation (barycentric / Krogh) -- small, well-conditioned node sets
for n in (5, 6):
    xi = np.concatenate([[0.0], np.cumsum(rng.uniform(0.6, 1.6, n - 1))])
    yi = np.sin(xi) + 0.3 * xi
    xnew = np.linspace(xi[0], xi[-1], 19)
    bcc.append(xyc(xi, yi, xnew, barycentric_interpolate(xi, yi, xnew)))
    kgc.append(xyc(xi, yi, xnew, krogh_interpolate(xi, yi, xnew)))

# BSpline evaluation (de Boor): build (t, c, k) with make_interp_spline, then evaluate
for kk in (2, 3):
    for n in (6, 9):
        xi = np.concatenate([[0.0], np.cumsum(rng.uniform(0.6, 1.6, n - 1))])
        yi = np.sin(xi) + 0.3 * xi
        spl = make_interp_spline(xi, yi, k=kk)
        t, c = np.asarray(spl.t, float), np.asarray(spl.c, float)
        xnew = np.linspace(xi[0], xi[-1], 21)
        bsp.append({'args': [F.enc(t), F.enc(c), F.enc(kk), F.enc(xnew)], 'kwargs': {},
                    'expect': F.enc_result(BSpline(t, c, kk)(xnew), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})
        spv.append({'args': [F.enc(t), F.enc(c), F.enc(kk), F.enc(xnew)], 'kwargs': {},
                    'expect': F.enc_result(np.asarray(splev(xnew, (t, c, kk))), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# pade: rational approximant from Taylor coefficients (use exp's series -> all coefficients nonzero)
for (m, nn) in ((2, 2), (3, 2), (2, 3), (3, 3)):
    an = np.array([1.0 / math.factorial(i) for i in range(m + nn + 1)])
    p, q = pade(an, m, nn)
    pdc.append({'args': [F.enc(an), F.enc(m), F.enc(nn)], 'kwargs': {},
                'expect': F.enc_result((p.coeffs, q.coeffs), ['p', 'q']), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# make_interp_spline: k=1 (linear) and k=3 (not-a-knot cubic) evaluated form
for k in (1, 3):
    for n in (5, 8):
        xi = np.concatenate([[0.0], np.cumsum(rng.uniform(0.6, 1.6, n - 1))])
        yi = np.sin(xi) + 0.3 * xi
        xnew = np.linspace(xi[0], xi[-1], 21)
        mis.append({'args': [F.enc(xi), F.enc(yi), F.enc(k), F.enc(xnew)], 'kwargs': {},
                    'expect': F.enc_result(make_interp_spline(xi, yi, k=k)(xnew), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# splder / splantider: B-spline calculus on a FITPACK (t, c, k) built with splrep
for n in (6, 9):
    xi = np.sort(np.unique(np.concatenate([[0.0], np.cumsum(rng.uniform(0.6, 1.6, n - 1))])))
    yi = np.sin(xi) + 0.3 * xi
    t, c, k = splrep(xi, yi, k=3, s=0)
    t, c = np.asarray(t, float), np.asarray(c, float)
    for nu in (1, 2):
        td, cd, kd = splder((t, c, k), nu)
        sdr.append({'args': [F.enc(t), F.enc(c), F.enc(int(k)), F.enc(nu)], 'kwargs': {},
                    'expect': F.enc_result((np.asarray(td), np.asarray(cd), kd), ['t', 'c', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-9}})
    ta, ca, ka = splantider((t, c, k), 1)
    sai.append({'args': [F.enc(t), F.enc(c), F.enc(int(k)), F.enc(1)], 'kwargs': {},
                'expect': F.enc_result((np.asarray(ta), np.asarray(ca), ka), ['t', 'c', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-9}})
    # splint: definite integral over [a, b] -- interior, full span, limits beyond the base interval (clamped),
    # a fully-outside interval (exactly zero) and a reversed interval (negative sign)
    for (a, b) in [(xi[1], xi[-2]), (xi[0], xi[-1]), (xi[0] - 1.0, xi[-1] + 1.0),
                   (xi[-1] + 0.5, xi[-1] + 1.0), (xi[-2], xi[1])]:
        spi.append({'args': [F.enc(float(a)), F.enc(float(b)), F.enc(t), F.enc(c), F.enc(int(k))], 'kwargs': {},
                    'expect': F.enc_result(float(splint(a, b, (t, c, k))), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# sproot: roots of a cubic (k=3) B-spline. Smooth sinusoidal data with a handful of interior roots, each well
# clear of a knot and comfortably below the default mest=10 so FITPACK does not truncate (its truncation order
# differs from plain ascending). splrep builds the (t, c, k); sproot returns the roots sorted ascending.
for (freq, npts, off) in [(0.9, 12, 0.15), (1.2, 14, 0.1), (0.7, 16, -0.2), (1.5, 15, 0.0)]:
    xs = np.linspace(0.3, 9.7, npts)
    ys = np.sin(freq * xs) + off
    t, c, k = splrep(xs, ys, k=3, s=0)
    t, c = np.asarray(t, float), np.asarray(c, float)
    spr.append({'args': [F.enc(t), F.enc(c), F.enc(int(k))], 'kwargs': {},
                'expect': F.enc_result(np.asarray(sproot((t, c, k))), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})

calls = [{'fn': 'pchip_interpolate', 'cases': pc}, {'fn': 'PchipInterpolator', 'cases': pci},
         {'fn': 'Akima1DInterpolator', 'cases': akc}, {'fn': 'CubicSpline', 'cases': csc},
         {'fn': 'CubicHermiteSpline', 'cases': chc},
         {'fn': 'barycentric_interpolate', 'cases': bcc}, {'fn': 'krogh_interpolate', 'cases': kgc},
         {'fn': 'BSpline', 'cases': bsp}, {'fn': 'pade', 'cases': pdc},
         {'fn': 'make_interp_spline', 'cases': mis}, {'fn': 'splev', 'cases': spv},
         {'fn': 'splder', 'cases': sdr}, {'fn': 'splantider', 'cases': sai},
         {'fn': 'splint', 'cases': spi}, {'fn': 'sproot', 'cases': spr}]
out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate: pchip_interpolate {len(pc)}, PchipInterpolator {len(pci)}, Akima1DInterpolator {len(akc)}, '
      f'CubicSpline {len(csc)}, CubicHermiteSpline {len(chc)} cases')
