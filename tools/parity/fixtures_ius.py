#!/usr/bin/env python3
"""scipy.interpolate.InterpolatedUnivariateSpline fixtures, verified against scipy. An interpolating spline with
s=0 is the same curve as make_interp_spline(x, y, k) with the default (interpolating) FITPACK knot pattern and
boundary conditions, so Tessero\\Interpolate\\InterpolatedUnivariateSpline is a pure-PHP facade over the existing
interpolate.make_interp_spline kernel routine. Only k=1 (linear) and k=3 (not-a-knot cubic) are covered: the
Tessero make_interp_spline kernel caps there, and k in {2,4,5} need the general B-spline collocation solve
(out of scope). Each case constructs IUS(x, y, k) and evaluates at query points xq. Host-independent, no pinned
env (bit-level agreement is not asserted; run-tests uses an rtol tolerance)."""
import json
import os

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

OUT = os.environ.get('IUS_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_ius.json')

rng = np.random.default_rng(20261012)
cases = []


def add(x, y, xq, k):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xq = np.asarray(xq, float)
    expect = InterpolatedUnivariateSpline(x, y, k=k)(xq)
    cases.append({'fn': 'InterpolatedUnivariateSpline', 'k': int(k),
                  'x': x.tolist(), 'y': y.tolist(), 'query': xq.tolist(),
                  'expect': np.asarray(expect, float).tolist()})


# --- k = 1 (piecewise linear) ---
x1 = np.linspace(0.0, 10.0, 21)
add(x1, np.sin(x1), np.linspace(0.4, 9.6, 11), k=1)
# non-uniform abscissae, k=1
x2 = np.array([0.0, 0.3, 1.1, 2.0, 3.7, 4.2, 6.0, 7.9, 9.0, 10.0])
add(x2, np.cos(x2) + 0.5 * x2, np.array([0.1, 1.5, 3.0, 5.5, 8.4, 9.9]), k=1)

# --- k = 3 (not-a-knot cubic) ---
x3 = np.linspace(0.0, 10.0, 21)
add(x3, np.sin(x3), np.linspace(0.4, 9.6, 11), k=3)
# non-uniform abscissae, smooth data, k=3
x4 = np.array([0.0, 0.5, 1.2, 2.3, 3.0, 4.4, 5.1, 6.8, 7.0, 8.5, 9.2, 10.0])
add(x4, np.exp(-0.2 * x4) * np.sin(x4), np.array([0.2, 1.0, 2.7, 4.0, 5.9, 7.3, 8.8, 9.7]), k=3)
# random smooth-ish data on a fine grid, k=3
x5 = np.sort(rng.uniform(0.0, 5.0, 15))
x5[0], x5[-1] = 0.0, 5.0
add(x5, rng.standard_normal(15), np.linspace(0.1, 4.9, 9), k=3)

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'ius_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_ius: {len(cases)} cases')
