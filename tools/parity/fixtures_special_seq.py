#!/usr/bin/env python3
"""scipy.special sequence/derivative fixtures (legendre_p, legendre_p_all, lqn, riccati_jn, riccati_yn).

Each function returns a whole family of values (the polynomial/function together with its derivatives,
or a sequence over degree). We flatten scipy's output and compare it against the pure-PHP
``Tessero\\SpecialSeq`` recurrences. Host-independent -- no pinned numpy/scipy environment required.
"""
import json
import os

import numpy as np
import scipy.special as sp

OUT = os.environ.get('SEQ_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'special_seq.json')

RTOL = 1e-8
cases = []


def flat(val):
    if isinstance(val, tuple):
        return [float(v) for part in val for v in np.ravel(np.asarray(part, float))]
    return [float(v) for v in np.ravel(np.asarray(val, float))]


def add(fn, args, val):
    cases.append({'fn': fn, 'args': list(args), 'expect': flat(val), 'rtol': RTOL})


for x in (-0.6, 0.3, 0.8):
    for n in range(0, 7):
        for dn in (0, 1, 2):
            add('legendre_p', [n, x, dn], sp.legendre_p(n, x, diff_n=dn))
    for n in range(0, 6):
        for dn in (0, 1, 2):
            add('legendre_p_all', [n, x, dn], sp.legendre_p_all(n, x, diff_n=dn))
for x in (0.2, 0.5, -0.4, 0.9):
    for n in range(0, 7):
        add('lqn', [n, x], sp.lqn(n, x))
for x in (0.3, 0.7, 1.4, 3.0, 6.0):
    for n in range(0, 7):
        add('riccati_jn', [n, x], sp.riccati_jn(n, x))
        add('riccati_yn', [n, x], sp.riccati_yn(n, x))

out = {'module': 'special', 'scipy': __import__('scipy').__version__, 'specseq_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'special_seq: {len(cases)} sequence cases')
