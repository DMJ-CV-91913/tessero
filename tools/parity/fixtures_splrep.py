#!/usr/bin/env python3
"""scipy.interpolate.make_splrep (and the legacy splrep), s=0 branch, verified against scipy. With s=0 both
reduce to make_interp_spline(x, y, k): make_splrep returns it directly; FITPACK splrep(x,y,k,s=0) produces the
identical curve (not-a-knot knots, coefficients agree to machine precision). Tessero\Interpolate\MakeSplrep is
a pure-PHP facade over the existing interpolate.make_interp_spline kernel. Only k=1 (linear) and k=3 (not-a-knot
cubic) are covered (the kernel caps there). Each case builds the s=0 spline and evaluates at query points.
Host-independent, no pinned env (an rtol tolerance is used in run-tests)."""
import json
import os

import numpy as np
from scipy.interpolate import make_splrep, splrep, splev

OUT = os.environ.get('SPLREP_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_splrep.json')

rng = np.random.default_rng(20261013)
cases = []


def add(x, y, xq, k):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xq = np.asarray(xq, float)
    spl = make_splrep(x, y, k=k, s=0)
    expect = spl(xq)
    # cross-check: FITPACK splrep(s=0) agrees (defends the shared fixture key).
    t, c, kk = splrep(x, y, k=k, s=0)
    assert np.allclose(expect, splev(xq, (t, c, kk)), rtol=1e-10, atol=1e-11)
    cases.append({'fn': 'make_splrep', 'k': int(k),
                  'x': x.tolist(), 'y': y.tolist(), 'query': xq.tolist(),
                  'expect': np.asarray(expect, float).tolist()})


# --- k = 1 (piecewise linear) ---
x1 = np.linspace(0.0, 10.0, 21)
add(x1, np.sin(x1), np.linspace(0.4, 9.6, 11), k=1)
x2 = np.array([0.0, 0.3, 1.1, 2.0, 3.7, 4.2, 6.0, 7.9, 9.0, 10.0])
add(x2, np.cos(x2) + 0.5 * x2, np.array([0.1, 1.5, 3.0, 5.5, 8.4, 9.9]), k=1)

# --- k = 3 (not-a-knot cubic) ---
x3 = np.linspace(0.0, 10.0, 21)
add(x3, np.sin(x3), np.linspace(0.4, 9.6, 11), k=3)
x4 = np.array([0.0, 0.5, 1.2, 2.3, 3.0, 4.4, 5.1, 6.8, 7.0, 8.5, 9.2, 10.0])
add(x4, np.exp(-0.2 * x4) * np.sin(x4), np.array([0.2, 1.0, 2.7, 4.0, 5.9, 7.3, 8.8, 9.7]), k=3)
x5 = np.sort(rng.uniform(0.0, 5.0, 15))
x5[0], x5[-1] = 0.0, 5.0
add(x5, rng.standard_normal(15), np.linspace(0.1, 4.9, 9), k=3)
x6 = np.sort(rng.uniform(-3.0, 3.0, 12))
x6 = np.unique(x6)
add(x6, np.tanh(x6), np.linspace(x6[0] + 0.05, x6[-1] - 0.05, 10), k=3)

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'splrep_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_splrep: {len(cases)} cases')
