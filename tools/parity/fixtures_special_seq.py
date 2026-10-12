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
    a = np.asarray(val)
    if np.iscomplexobj(a):
        out = []
        for x in a.ravel():
            out += [float(x.real), float(x.imag)]
        return out
    return [float(v) for v in np.ravel(a.astype(float))]


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

# associated / spherical Legendre (new 1.15 API): values + derivatives, all orders -n..n
for x in (-0.6, 0.3, 0.8):
    for n in range(0, 6):
        for m in range(-n, n + 1):
            for dn in (0, 1, 2):
                for nrm in (0, 1):
                    add('assoc_legendre_p', [n, m, x, dn, nrm],
                        sp.assoc_legendre_p(n, m, x, diff_n=dn, norm=bool(nrm)))
    for N in range(0, 5):
        for M in range(0, max(1, N + 1)):
            for dn in (0, 1, 2):
                for nrm in (0, 1):
                    add('assoc_legendre_p_all', [N, M, x, dn, nrm],
                        sp.assoc_legendre_p_all(N, M, x, diff_n=dn, norm=bool(nrm)))
for th in (0.3, 0.6, 1.2, 2.5):
    for n in range(0, 6):
        for m in range(-n, n + 1):
            for dn in (0, 1, 2):
                add('sph_legendre_p', [n, m, th, dn], sp.sph_legendre_p(n, m, th, diff_n=dn))
    for N in range(0, 5):
        for M in range(0, max(1, N + 1)):
            for dn in (0, 1, 2):
                add('sph_legendre_p_all', [N, M, th, dn], sp.sph_legendre_p_all(N, M, th, diff_n=dn))

# sph_harm_y_all: complex spherical harmonics for all degrees/orders (= sph_legendre_p * exp(i m phi))
for th, ph in ((0.6, 0.4), (1.2, 2.0), (0.3, -1.0), (2.5, 3.0)):
    for n in range(0, 5):
        for m in range(0, max(1, n + 1)):
            add('sph_harm_y_all', [n, m, th, ph], sp.sph_harm_y_all(n, m, th, ph))

# lqmn: Legendre Q_n^m + derivative (2-D), |x| < 1
for x in (0.2, 0.5, -0.4, 0.8):
    for M in range(0, 4):
        for N in range(0, 5):
            add('lqmn', [M, N, x], sp.lqmn(M, N, x))
# stirling2: scalar pairs, n <= 50 (float DP domain)
for n, k in [(0, 0), (4, 0), (5, 2), (6, 3), (10, 4), (7, 3), (12, 5), (20, 7), (50, 25), (4, 4), (15, 1), (15, 15)]:
    add('stirling2', [n, k], sp.stirling2(n, k))

out = {'module': 'special', 'scipy': __import__('scipy').__version__, 'specseq_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'special_seq: {len(cases)} sequence cases')
