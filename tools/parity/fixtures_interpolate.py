#!/usr/bin/env python3
"""scipy.interpolate fixtures: 1-D piecewise-cubic interpolators evaluated at query points -- pchip_interpolate,
PchipInterpolator, Akima1DInterpolator, CubicSpline (not-a-knot) and CubicHermiteSpline. Deterministic arithmetic
over the sample arrays; generated in the pinned environment for consistency with the other scipy groups."""
import json, os
import numpy as np
from scipy.interpolate import insert, spalde, FloaterHormannInterpolator  # noqa: E402 (grouped below)
from scipy.interpolate import (pchip_interpolate, PchipInterpolator, Akima1DInterpolator,
                               CubicSpline, CubicHermiteSpline,
                               barycentric_interpolate, krogh_interpolate,
                               make_interp_spline, BSpline, pade, splev,
                               splrep, splder, splantider, splint, sproot, lagrange,
                               RegularGridInterpolator, interpn, RBFInterpolator, make_lsq_spline,
                               BarycentricInterpolator, KroghInterpolator, PPoly, BPoly)
import math
import fixtures_np as F  # enc / enc_result / fixture_env

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'interpolate.json')
rng = np.random.default_rng(20261008)

pc, pci, akc, csc, chc, bcc, kgc, bsp, pdc, mis, spv, sdr, sai, spi, spr, lag, rgi, ipn, rbf, lsq = ([] for _ in range(20))


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

# lagrange: interpolating polynomial through (x, w) as poly1d coefficients (highest-degree first). Generic
# well-separated nodes over a moderate range so the polynomial is genuinely degree M-1 (poly1d keeps all M
# coefficients -- no exact-zero leading trim) and the coefficients stay modestly sized.
for M in (3, 4, 5, 6):
    xl = np.sort(rng.uniform(-2.0, 2.0, M))
    xl = xl + np.linspace(0, 1e-3, M)          # nudge apart to keep nodes well separated
    wl = np.sin(1.1 * xl) + 0.4 * rng.normal(size=M)
    lag.append({'args': [F.enc(xl), F.enc(wl)], 'kwargs': {},
                'expect': F.enc_result(np.asarray(lagrange(xl, wl).coef), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# RegularGridInterpolator / interpn: multilinear interpolation on a regular n-D grid (n = 1, 2, 3), queried at
# in-bounds points. points is a list of per-axis coordinate arrays (strictly increasing, non-uniform); values is
# the n-D sample array; xi is (m, n). Both functions share the same multilinear core, so the expected values are
# identical.
for n in (1, 2, 3):
    grids = [np.sort(rng.uniform(0.0, 10.0, 5 + n - d)) for d in range(n)]
    grids = [g + np.linspace(0, 1e-3, len(g)) for g in grids]       # ensure strictly increasing
    shp = tuple(len(g) for g in grids)
    vals = np.sin(rng.uniform(size=shp)) + 0.3 * rng.normal(size=shp)
    mq = 7
    xq = np.stack([rng.uniform(grids[d][0] + 1e-2, grids[d][-1] - 1e-2, mq) for d in range(n)], axis=1)
    pts = [np.asarray(g, float) for g in grids]
    rgi.append({'args': [F.enc(pts), F.enc(vals), F.enc(xq)], 'kwargs': {},
                'expect': F.enc_result(np.asarray(RegularGridInterpolator(tuple(pts), vals)(xq)), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})
    ipn.append({'args': [F.enc(pts), F.enc(vals), F.enc(xq)], 'kwargs': {},
                'expect': F.enc_result(np.asarray(interpn(tuple(pts), vals, xq)), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# RBFInterpolator: scattered-data RBF interpolation, default thin_plate_spline (degree-1 tail) and linear
# (degree-0) kernels. Small, well-separated node sets over a moderate range so the augmented system is
# well-conditioned (the kernels are scale-invariant; higher-degree / epsilon-dependent kernels are out of scope).
for kern in ('thin_plate_spline', 'linear'):
    for ndim in (1, 2):
        pnum = 7 if ndim == 1 else 9
        yk = rng.uniform(-2.0, 2.0, (pnum, ndim))
        dk = np.sin(yk.sum(axis=1)) + 0.3 * rng.normal(size=pnum)
        xk = rng.uniform(-1.5, 1.5, (6, ndim))
        rbf.append({'args': [F.enc(yk), F.enc(dk), F.enc(xk), F.enc(kern)], 'kwargs': {},
                    'expect': F.enc_result(np.asarray(RBFInterpolator(yk, dk, kernel=kern)(xk)), []), 'compare': 'tol', 'tol': {'atol': 1e-8}})

# make_lsq_spline: least-squares cubic B-spline fit with an explicit clamped knot vector (a few interior knots at
# data quantiles), evaluated at xnew. Well-conditioned (plenty of data per coefficient) so the normal-equations
# solve matches scipy's banded least-squares.
for npts in (24, 32):
    xs = np.sort(np.unique(rng.uniform(0.0, 10.0, npts)))
    ys = np.sin(0.8 * xs) + 0.1 * rng.normal(size=len(xs))
    kk = 3
    nint = 3
    interior = np.quantile(xs, np.linspace(0, 1, nint + 2)[1:-1])
    tk = np.r_[[xs[0]] * (kk + 1), interior, [xs[-1]] * (kk + 1)]
    xn = np.linspace(xs[0], xs[-1], 17)
    lsq.append({'args': [F.enc(xs), F.enc(ys), F.enc(np.asarray(tk, float)), F.enc(kk), F.enc(xn)], 'kwargs': {},
                'expect': F.enc_result(np.asarray(make_lsq_spline(xs, ys, tk, kk)(xn)), []), 'compare': 'tol', 'tol': {'atol': 1e-8}})

# FloaterHormann rational interpolation, B-spline knot insert (Boehm), spalde (all derivatives)
fhc, insc, spa = [], [], []
for _ in range(6):
    # well-separated nodes keep the barycentric form well-conditioned
    xi = np.linspace(-1.2, 1.2, 6) + rng.uniform(-0.08, 0.08, 6); xi = np.sort(xi)
    yi = rng.normal(size=len(xi)); xnew = rng.uniform(-1, 1, 6)
    fhc.append({'args': [F.enc(xi), F.enc(yi), F.enc(xnew), F.enc(3)], 'kwargs': {},
                'expect': F.enc_result(np.asarray(FloaterHormannInterpolator(xi, yi, d=3)(xnew)), []), 'compare': 'tol', 'tol': {'atol': 1e-8}})
    xs = np.linspace(0, 1, 10); ys = np.sin(5 * xs); t, c, k = splrep(xs, ys, s=0)
    xv = float(rng.uniform(0.12, 0.88))
    T, C, K = insert(xv, (t, c, k))
    Cm = np.asarray(C)[:len(T) - int(k) - 1]                 # meaningful coefficients only (SciPy pads with FITPACK scratch)
    insc.append({'args': [F.enc(float(xv)), F.enc(t), F.enc(c), F.enc(int(k))], 'kwargs': {},
                 'expect': F.enc_result((np.asarray(T), Cm, int(K)), ['t', 'c', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-9}})
    xe = rng.uniform(0.05, 0.95, 5)
    spa.append({'args': [F.enc(t), F.enc(c), F.enc(int(k)), F.enc(xe)], 'kwargs': {},
                'expect': F.enc_result(np.asarray(spalde(xe, (t, c, k))), []), 'compare': 'tol', 'tol': {'atol': 1e-7}})

# construct-and-evaluate interpolator classes: Barycentric/Krogh (global poly), PPoly/BPoly (piecewise)
bci, kgi, ppc, bpc = [], [], [], []
for _ in range(8):
    xi = np.sort(np.linspace(-2, 2, 5) + rng.uniform(-0.2, 0.2, 5)); yi = rng.normal(size=len(xi)); xnew = rng.uniform(-2, 2, 7)
    bci.append(xyc(xi, yi, xnew, BarycentricInterpolator(xi, yi)(xnew)))
    kgi.append(xyc(xi, yi, xnew, KroghInterpolator(xi, yi)(xnew)))
    xb = np.sort(np.unique(rng.uniform(-2, 2, 4))); mm = len(xb) - 1
    cp = rng.normal(size=(3, mm)); cb = rng.normal(size=(3, mm)); xe = rng.uniform(xb[0], xb[-1], 6)
    ppc.append({'args': [F.enc(cp), F.enc(xb), F.enc(xe)], 'kwargs': {},
                'expect': F.enc_result(np.asarray(PPoly(cp, xb)(xe)), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})
    bpc.append({'args': [F.enc(cb), F.enc(xb), F.enc(xe)], 'kwargs': {},
                'expect': F.enc_result(np.asarray(BPoly(cb, xb)(xe)), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})

calls = [{'fn': 'pchip_interpolate', 'cases': pc}, {'fn': 'PchipInterpolator', 'cases': pci},
         {'fn': 'Akima1DInterpolator', 'cases': akc}, {'fn': 'CubicSpline', 'cases': csc},
         {'fn': 'CubicHermiteSpline', 'cases': chc},
         {'fn': 'barycentric_interpolate', 'cases': bcc}, {'fn': 'krogh_interpolate', 'cases': kgc},
         {'fn': 'BSpline', 'cases': bsp}, {'fn': 'pade', 'cases': pdc},
         {'fn': 'make_interp_spline', 'cases': mis}, {'fn': 'splev', 'cases': spv},
         {'fn': 'splder', 'cases': sdr}, {'fn': 'splantider', 'cases': sai},
         {'fn': 'splint', 'cases': spi}, {'fn': 'sproot', 'cases': spr},
         {'fn': 'lagrange', 'cases': lag},
         {'fn': 'RegularGridInterpolator', 'cases': rgi}, {'fn': 'interpn', 'cases': ipn},
         {'fn': 'RBFInterpolator', 'cases': rbf}, {'fn': 'make_lsq_spline', 'cases': lsq},
         {'fn': 'BarycentricInterpolator', 'cases': bci}, {'fn': 'KroghInterpolator', 'cases': kgi},
         {'fn': 'PPoly', 'cases': ppc}, {'fn': 'BPoly', 'cases': bpc},
         {'fn': 'FloaterHormannInterpolator', 'cases': fhc}, {'fn': 'insert', 'cases': insc},
         {'fn': 'spalde', 'cases': spa}]
out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate: pchip_interpolate {len(pc)}, PchipInterpolator {len(pci)}, Akima1DInterpolator {len(akc)}, '
      f'CubicSpline {len(csc)}, CubicHermiteSpline {len(chc)} cases')
