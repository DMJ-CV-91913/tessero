#!/usr/bin/env python3
"""scipy.interpolate.make_splprep (and legacy splprep), s=0, verified against scipy. For s=0 both build the
chord-length parametrization u and one make_interp_spline per coordinate component. Tessero\Interpolate\
MakeSplprep is a pure-PHP facade over interpolate.make_interp_spline (k=1,3). Each case stores the components,
query parameters uq, u, and the per-component expected values. Host-independent."""
import json
import os

import numpy as np
from scipy.interpolate import make_splprep, splprep, splev

OUT = os.environ.get('SPLPREP_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'interpolate_splprep.json')

rng = np.random.default_rng(20261013)
cases = []


def add(comps, k, uq):
    comps = [np.asarray(c, float) for c in comps]
    uq = np.asarray(uq, float)
    spl, u = make_splprep(comps, k=k, s=0)
    ev = np.atleast_2d(spl(uq))
    if ev.shape[0] != len(comps):
        ev = ev.T
    # cross-check legacy splprep(s=0)
    tck, u2 = splprep(comps, k=k, s=0)
    ev2 = splev(uq, tck)
    assert np.allclose(u, u2, atol=1e-12)
    for j in range(len(comps)):
        assert np.allclose(ev[j], ev2[j], rtol=1e-9, atol=1e-10)
    cases.append({'fn': 'make_splprep', 'k': int(k),
                  'x': [c.tolist() for c in comps],
                  'uq': uq.tolist(),
                  'u': np.asarray(u, float).tolist(),
                  'expect': [np.asarray(ev[j], float).tolist() for j in range(len(comps))]})


m = 13
t = np.linspace(0, 1, m)
add([np.cos(4 * t), np.sin(4 * t)], 3, np.linspace(0, 1, 15))
add([np.cos(4 * t), np.sin(4 * t)], 1, np.linspace(0, 1, 15))
add([t, t ** 2, t ** 3], 3, np.linspace(0, 1, 11))
m2 = 10
t2 = np.sort(rng.uniform(0, 1, m2))
comps = [np.cos(3 * t2) + rng.normal(0, 0.01, m2), np.sin(3 * t2) + rng.normal(0, 0.01, m2)]
add(comps, 3, np.linspace(0, 1, 12))
add([np.exp(-t), np.log(t + 1)], 3, np.linspace(0, 1, 9))
add([np.sin(2 * t), 2 - t, t ** 2], 1, np.linspace(0, 1, 10))

out = {'module': 'interpolate', 'scipy': __import__('scipy').__version__, 'splprep_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'interpolate_splprep: {len(cases)} cases')
