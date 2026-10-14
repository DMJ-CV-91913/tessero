#!/usr/bin/env python3
"""scipy.interpolate.LSQBivariateSpline fixtures, verified against scipy.

A least-squares TENSOR-PRODUCT bivariate B-spline fit with user-supplied INTERIOR knots tx, ty and
degrees kx, ky, evaluated on a tensor grid. Unlike the auto-knot smoothing FITPACK family this is a
plain deterministic linear least-squares problem: SciPy augments the interior knots to the full
clamped knot vectors [x.min()]*(kx+1)+tx+[x.max()]*(kx+1) (and similarly for ty), builds the
tensor-product B-spline design matrix B[i, p*ncy+q] = Nx_p(x_i)*Ny_q(y_i), and solves min ||B c - z||.
Tessero\\Interpolate\\LSQBivariateSpline reproduces exactly that (lstsq over the same design matrix,
then tensor de Boor eval via the NdBSpline kernel). Verified max abs err ~3e-13 vs scipy for
kx, ky in {1, 2, 3}, symmetric and asymmetric degrees.

Pinned env (AVX2 reference host): NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1."""
import json
import os

import numpy as np
from scipy.interpolate import LSQBivariateSpline

OUT = os.environ.get('LSQBIV_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_lsqbiv.json')

cases = []


def add(nd, kx, ky, nintx, ninty, seed, f):
    r = np.random.default_rng(seed)
    x = r.uniform(0.0, 10.0, nd)
    y = r.uniform(0.0, 10.0, nd)
    z = f(x, y) + 0.05 * r.normal(size=nd)
    tx = np.linspace(x.min(), x.max(), nintx + 2)[1:-1]
    ty = np.linspace(y.min(), y.max(), ninty + 2)[1:-1]
    sp = LSQBivariateSpline(x, y, z, tx, ty, kx=kx, ky=ky)
    xq = np.linspace(x.min(), x.max(), 7)
    yq = np.linspace(y.min(), y.max(), 6)
    expect = np.asarray(sp(xq, yq), float)   # (7, 6)
    cases.append({'fn': 'LSQBivariateSpline',
                  'x': x.tolist(), 'y': y.tolist(), 'z': z.tolist(),
                  'tx': tx.tolist(), 'ty': ty.tolist(),
                  'kx': int(kx), 'ky': int(ky),
                  'xq': xq.tolist(), 'yq': yq.tolist(),
                  'expect': expect.tolist()})


add(80, 3, 3, 3, 3, 1, lambda x, y: np.sin(0.4 * x) * np.cos(0.3 * y))
add(100, 3, 3, 4, 2, 2, lambda x, y: 0.3 * x - 0.2 * y + 0.05 * x * y)
add(60, 1, 1, 4, 4, 3, lambda x, y: np.exp(-0.1 * ((x - 5) ** 2 + (y - 5) ** 2)))
add(90, 2, 2, 3, 3, 4, lambda x, y: (x - 5) ** 2 + (y - 4) ** 2)
add(120, 3, 1, 5, 3, 5, lambda x, y: np.sin(0.5 * x) + 0.1 * y)
add(70, 1, 3, 2, 4, 6, lambda x, y: np.cos(0.4 * y) + 0.05 * x)

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'lsqbiv_cases': cases}
with open(OUT, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'), allow_nan=False)
print(f'interpolate_lsqbiv: {len(cases)} cases')
