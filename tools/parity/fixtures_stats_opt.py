#!/usr/bin/env python3
"""scipy.stats optimiser-based transform selectors (boxcox_normmax, yeojohnson_normmax).

These minimise a 1-D objective over the data (via Brent / bounded minimisation), so the pure-PHP
Tessero\\StatsFns reproduces them with Tessero's scipy-matched optimisers. The data is embedded in the
fixture, so the recorded expectation from scipy is host-independent (no pinned environment required).
"""
import json
import os

import numpy as np
import scipy.stats as st

OUT = os.environ.get('STATSOPT_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_opt.json')

cases = []


def add(fn, x, expect, method=None, rtol=1e-5):
    c = {'fn': fn, 'x': [float(v) for v in x], 'expect': float(expect), 'rtol': rtol}
    if method:
        c['method'] = method
    cases.append(c)


for seed in range(5):
    x = np.random.default_rng(seed).gamma(2.0, 1.5, size=35) + 0.3   # positive
    add('boxcox_normmax', x, st.boxcox_normmax(x, method='pearsonr'))
    add('boxcox_normmax', x, st.boxcox_normmax(x, method='mle'), method='mle')

for seed in range(5):
    rng = np.random.default_rng(100 + seed)
    xm = rng.normal(0.0, 2.0, size=40)                               # mixed sign
    add('yeojohnson_normmax', xm, st.yeojohnson_normmax(xm))
    xp = rng.gamma(2.0, 1.0, size=30) + 0.2                          # all positive
    add('yeojohnson_normmax', xp, st.yeojohnson_normmax(xp))

out = {'module': 'stats', 'scipy': __import__('scipy').__version__, 'statsfn_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'stats_opt: {len(cases)} cases')
