#!/usr/bin/env python3
"""scipy.interpolate sphere least-squares / rectangular-grid bivariate smoothing-spline fixtures.

Cover the FITPACK sphere(iopt=-1)/spgrid family, implemented in the Tessero kernel by the f2c-vendored
Dierckx sphere_/spgrid_ routines (interpolate.spherfit_lsq / interpolate.spgrid), bit-exact vs scipy:

  lsqsphere_cases  -> scipy.interpolate.LSQSphereBivariateSpline (get_knots / get_coeffs)
  rectsphere_cases -> scipy.interpolate.RectSphereBivariateSpline (get_knots / get_coeffs)
"""
import json
import os

import numpy as np
from scipy.interpolate import LSQSphereBivariateSpline, RectSphereBivariateSpline

OUT = os.environ.get('FITPACKSPH2_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_fitpack_sphere2.json')

rng = np.random.default_rng(20261014)
lsqsphere_cases, rectsphere_cases = [], []

# ---- LSQSphereBivariateSpline (sphere iopt=-1, given interior knots) ----
for (ntheta, nphi, nkt, nkp) in [(9, 11, 3, 3), (11, 13, 4, 4), (7, 9, 2, 2)]:
    theta_g = np.linspace(0.15, np.pi - 0.15, ntheta)
    phi_g = np.linspace(0.1, 2.0 * np.pi - 0.1, nphi)
    T, P = np.meshgrid(theta_g, phi_g, indexing='ij')
    theta = T.ravel()
    phi = P.ravel()
    r = np.sin(theta) * np.cos(phi) + 2.0 + 0.01 * rng.standard_normal(theta.size)
    w = np.ones_like(r)
    tt = np.linspace(0.3, np.pi - 0.3, nkt + 2)[1:-1]
    tp = np.linspace(0.3, 2.0 * np.pi - 0.3, nkp + 2)[1:-1]
    sp = LSQSphereBivariateSpline(theta, phi, r, tt, tp, w=w, eps=1e-16)
    ttk, tpk = sp.get_knots()
    lsqsphere_cases.append(dict(
        theta=theta.tolist(), phi=phi.tolist(), r=r.tolist(), w=w.tolist(),
        tt=tt.tolist(), tp=tp.tolist(),
        exp_tt=np.asarray(ttk).tolist(), exp_tp=np.asarray(tpk).tolist(),
        exp_c=np.asarray(sp.get_coeffs()).tolist()))

# ---- RectSphereBivariateSpline (spgrid, rectangular grid) ----
for (nu, nv, s) in [(7, 9, 0.0), (9, 11, 1.0), (6, 8, 0.5)]:
    u = np.linspace(0.12, np.pi - 0.12, nu)
    v = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
    data = np.sin(u)[:, None] * np.cos(v)[None, :] + 2.0
    sp = RectSphereBivariateSpline(u, v, data, s=s)
    tuk, tvk = sp.get_knots()
    rectsphere_cases.append(dict(
        u=u.tolist(), v=v.tolist(), r=data.ravel().tolist(), s=float(s),
        exp_tu=np.asarray(tuk).tolist(), exp_tv=np.asarray(tvk).tolist(),
        exp_c=np.asarray(sp.get_coeffs()).tolist()))

data = {'lsqsphere_cases': lsqsphere_cases, 'rectsphere_cases': rectsphere_cases}
with open(OUT, 'w') as f:
    json.dump(data, f)
print(f"wrote {OUT}: {len(lsqsphere_cases)} lsqsphere, {len(rectsphere_cases)} rectsphere")
