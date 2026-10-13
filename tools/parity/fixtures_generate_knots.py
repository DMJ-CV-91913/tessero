#!/usr/bin/env python3
"""scipy.interpolate.generate_knots (s=0 branch) fixtures, verified against scipy. For s=0 generate_knots yields
one knot vector = _not_a_knot(x, k), a closed-form numeric construction. Tessero\Interpolate\GenerateKnots is a
self-contained pure-PHP routine (no kernel). Covers k=1..5. Host-independent."""
import json
import os

import numpy as np
from scipy.interpolate import generate_knots

OUT = os.environ.get('GENKNOTS_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_generate_knots.json')

rng = np.random.default_rng(20261013)
cases = []


def add(x, k):
    x = np.asarray(x, float)
    y = np.sin(x)  # y unused for s=0 knots but required by the API
    t = list(generate_knots(x, y, k=k, s=0))
    assert len(t) == 1
    cases.append({'fn': 'generate_knots', 'k': int(k),
                  'x': x.tolist(), 'expect': np.asarray(t[0], float).tolist()})


xs = [
    np.linspace(0.0, 10.0, 14),
    np.array([0.0, 0.3, 1.1, 2.0, 3.7, 4.2, 6.0, 7.9, 9.0, 10.0, 11.4, 12.0]),
    np.sort(np.unique(rng.uniform(-5.0, 5.0, 15))),
    np.sort(np.unique(rng.uniform(0.0, 1.0, 11))),
]
for x in xs:
    for k in (1, 2, 3, 4, 5):
        if len(x) > k:
            add(x, k)

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'genknots_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_generate_knots: {len(cases)} cases')
