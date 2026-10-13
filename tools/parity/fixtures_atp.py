#!/usr/bin/env python3
"""scipy.interpolate.approximate_taylor_polynomial fixtures, verified against scipy. Faithful reimplementation
(Newton divided differences re-expanded about x). f is a named kind the PHP harness can rebuild. Deterministic,
no RNG; host-independent (rtol tolerance in run-tests)."""
import json
import os

import numpy as np
from scipy.interpolate import approximate_taylor_polynomial

OUT = os.environ.get('ATP_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_atp.json')

FUNCS = {
    'sinx': np.sin,
    'expx': np.exp,
    'cos2x': lambda z: np.cos(2.0 * z),
    'lorentz': lambda z: 1.0 / (1.0 + z ** 2),
    'gauss': lambda z: np.exp(-z ** 2),
}
cases = []


def add(kind, x, degree, scale, order=None):
    f = FUNCS[kind]
    p = approximate_taylor_polynomial(f, x, degree, scale, order=order)
    cases.append({'fn': 'approximate_taylor_polynomial', 'kind': kind,
                  'x': float(x), 'degree': int(degree), 'scale': float(scale),
                  'order': (None if order is None else int(order)),
                  'expect': np.asarray(p.coeffs, float).tolist()})


add('sinx', 0.0, 3, 1.0)
add('sinx', 0.5, 4, 0.8, order=6)
add('expx', 0.0, 3, 1.0)
add('expx', 1.0, 5, 1.0, order=7)
add('cos2x', 0.0, 4, 1.0, order=6)
add('cos2x', 0.5, 3, 0.9)
add('lorentz', 0.0, 4, 0.8, order=6)
add('lorentz', 0.3, 2, 0.5)
add('gauss', 0.0, 4, 1.0, order=6)
add('gauss', -0.4, 3, 0.7)

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'atp_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_atp: {len(cases)} cases')
