#!/usr/bin/env python3
"""scipy.interpolate.make_smoothing_spline fixtures, verified against scipy. Penalised (natural cubic)
smoothing B-spline == Tessero\\Interpolate\\MakeSmoothingSpline, a pure-PHP facade over the existing banded
LAPACK prims (slinalg.solve_banded for the solve, slinalg.cholesky_banded for the GCV trace) plus
Optimize\\Minimize::fminbound (lam=None/GCV) and Interpolate::BSpline (de Boor) for evaluation. Each case
evaluates the fitted spline at query points; both lam-given and lam=None (GCV) are covered. GCV uses
fminbound and the de Hoog banded-inverse trace, so values are compared at rtol ~1e-6 (lam-given is exact).

Pinned env for scipy value-gen (banded LAPACK is deterministic under it):
  NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" OPENBLAS_CORETYPE=Haswell
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"""
import json
import os

import numpy as np
from scipy.interpolate import make_smoothing_spline

OUT = os.environ.get('MSS_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_mss.json')

rng = np.random.default_rng(20261012)
cases = []


def add(tag, x, y, w=None, lam=None, query=None):
    x = np.ascontiguousarray(x, float)
    y = np.ascontiguousarray(y, float)
    spl = make_smoothing_spline(x, y, w=(None if w is None else np.asarray(w, float)), lam=lam)
    if query is None:
        query = np.linspace(float(x[0]), float(x[-1]), 60)
    query = np.asarray(query, float)
    expect = spl(query)
    # lam-given is a pure banded solve (bit-exact); lam=None/GCV goes through fminbound + the de Hoog
    # trace, so lam lands in a flat region of the GCV objective and the fit matches scipy only to ~1e-5.
    rtol = 1e-9 if lam is not None else 1e-5
    cases.append({
        'tag': tag,
        'fn': 'make_smoothing_spline',
        'x': x.tolist(),
        'y': y.tolist(),
        'w': (None if w is None else np.asarray(w, float).tolist()),
        'lam': (None if lam is None else float(lam)),
        'query': query.tolist(),
        'rtol': rtol,
        'expect': np.asarray(expect, float).tolist(),
    })


# --- lam-given cases (exact) ---
n = 50
x = np.sort(rng.uniform(-2, 2, n))
yv = x**3 + x**2 * np.sin(4 * x) + rng.normal(0, 1.0, n)
add('lam_given_1.0', x, yv, lam=1.0)
add('lam_given_0.01', x, yv, lam=0.01)
w = rng.uniform(0.5, 2.0, n)
add('lam_given_w', x, yv, w=w, lam=0.5)

# --- lam=None (GCV) cases ---
add('gcv_cubic', x, yv)
add('gcv_cubic_w', x, yv, w=w)

n2 = 120
x2 = np.sort(rng.uniform(0, 10, n2))
y2 = np.sin(x2) + rng.normal(0, 0.3, n2)
add('gcv_sin', x2, y2)

# a small, well-separated dataset (min size path)
x3 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y3 = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0, 0.0])
add('gcv_small', x3, y3)
add('lam_given_small', x3, y3, lam=0.3)

# non-uniform abscissae, GCV
x4 = np.sort(rng.uniform(-5, 5, 80)); x4 = np.unique(x4)
y4 = np.exp(-x4**2 / 4) + rng.normal(0, 0.05, x4.size)
add('gcv_gauss_nonunif', x4, y4)

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'mss_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_mss: {len(cases)} cases -> {OUT}')
