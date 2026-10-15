#!/usr/bin/env python3
"""scipy.interpolate FITPACK scattered/sphere bivariate smoothing-spline fixtures, verified against scipy.

These cover the auto-knot FITPACK surfit/sphere family, implemented in the Tessero kernel by the f2c-vendored
Dierckx surfit_/sphere_ routines (interpolate.bisplrep / interpolate.spherfit), so the knots and coefficients
match SciPy's _dfitpack bit-for-bit (0 ULP):

  bisplrep_cases     -> scipy.interpolate.bisplrep            (tck = [tx, ty, c])
  smoothbiv_cases    -> scipy.interpolate.SmoothBivariateSpline (get_knots / get_coeffs)
  bivariate_cases    -> scipy.interpolate.BivariateSpline      (base-class .ev evaluation via the NdBSpline kernel)
  smoothsphere_cases -> scipy.interpolate.SmoothSphereBivariateSpline (get_knots / get_coeffs)

Host-independent: surfit/sphere are integer-knot iterative solves with no BLAS, deterministic across hosts."""
import json
import os

import numpy as np
from scipy.interpolate import bisplrep, SmoothBivariateSpline, SmoothSphereBivariateSpline

OUT = os.environ.get('FITPACK2D_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_fitpack2d.json')

rng = np.random.default_rng(20261014)
bisplrep_cases, smoothbiv_cases, bivariate_cases, smoothsphere_cases = [], [], [], []

# ---- scattered bivariate (surfit): bisplrep / SmoothBivariateSpline / BivariateSpline ----
for (m, kx, ky, s) in [(50, 3, 3, 0.5), (60, 3, 3, 1.0), (40, 2, 2, 0.8), (35, 1, 1, 0.3), (70, 3, 3, 2.0)]:
    x = rng.uniform(0.0, 1.0, m)
    y = rng.uniform(0.0, 1.0, m)
    z = np.sin(3.0 * x) * np.cos(2.0 * y) + 0.01 * rng.standard_normal(m)
    w = np.ones(m)
    tck = bisplrep(x, y, z, w=w, kx=kx, ky=ky, s=s, eps=1e-16)
    sp = SmoothBivariateSpline(x, y, z, w=w, kx=kx, ky=ky, s=s, eps=1e-16)
    tx, ty = sp.get_knots()
    base = dict(x=x.tolist(), y=y.tolist(), z=z.tolist(), w=w.tolist(), kx=int(kx), ky=int(ky), s=float(s))
    bisplrep_cases.append({**base, 'exp_tx': np.asarray(tck[0]).tolist(),
                           'exp_ty': np.asarray(tck[1]).tolist(), 'exp_c': np.asarray(tck[2]).tolist()})
    smoothbiv_cases.append({**base, 'exp_tx': np.asarray(tx).tolist(),
                            'exp_ty': np.asarray(ty).tolist(), 'exp_c': np.asarray(sp.get_coeffs()).tolist()})
    xi = rng.uniform(0.1, 0.9, 7)
    yi = rng.uniform(0.1, 0.9, 7)
    bivariate_cases.append({**base, 'xi': xi.tolist(), 'yi': yi.tolist(),
                            'exp_ev': np.asarray(sp.ev(xi, yi)).tolist()})

# ---- sphere (sphere_): SmoothSphereBivariateSpline ----
for (m, s) in [(60, 2.0), (80, 3.5), (50, 1.5)]:
    theta = rng.uniform(0.1, np.pi - 0.1, m)
    phi = rng.uniform(0.0, 2.0 * np.pi, m)
    r = np.sin(theta) * np.cos(phi) + 2.0
    w = np.ones(m)
    sp = SmoothSphereBivariateSpline(theta, phi, r, w=w, s=s, eps=1e-16)
    tt, tp = sp.get_knots()
    smoothsphere_cases.append(dict(theta=theta.tolist(), phi=phi.tolist(), r=r.tolist(), w=w.tolist(),
                                   s=float(s), exp_tt=np.asarray(tt).tolist(),
                                   exp_tp=np.asarray(tp).tolist(), exp_c=np.asarray(sp.get_coeffs()).tolist()))

data = {'bisplrep_cases': bisplrep_cases, 'smoothbiv_cases': smoothbiv_cases,
        'bivariate_cases': bivariate_cases, 'smoothsphere_cases': smoothsphere_cases}
with open(OUT, 'w') as f:
    json.dump(data, f)
print(f"wrote {OUT}: {len(bisplrep_cases)} bisplrep, {len(smoothbiv_cases)} smoothbiv, "
      f"{len(bivariate_cases)} bivariate, {len(smoothsphere_cases)} smoothsphere")
