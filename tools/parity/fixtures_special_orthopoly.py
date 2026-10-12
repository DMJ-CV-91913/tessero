#!/usr/bin/env python3
"""scipy.special orthogonal-polynomial factory fixtures (legendre, chebyt, jacobi, ...).

Each factory returns an ``orthopoly1d`` (a ``numpy.poly1d`` subclass). We record the power-basis
coefficients (``P.c``, highest degree first) and the polynomial evaluated at a few in-domain points,
which the runner reproduces with the pure-PHP ``Tessero\\OrthoPoly`` factories. These are exact
algebraic values from scipy, host-independent -- no pinned numpy/scipy environment required.
"""
import json
import os

import scipy.special as sp

OUT = os.environ.get('ORTHO_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'special_orthopoly.json')

# in-domain evaluation points per family
X_UNIT = [-0.8, -0.3, 0.4, 0.9]        # [-1, 1]
X_WIDE = [-1.5, -0.5, 0.7, 1.8]        # [-2, 2] (chebyc / chebys)
X_REAL = [-1.5, -0.3, 0.5, 2.0]        # hermite / laguerre families
X_SH = [0.1, 0.35, 0.6, 0.85]          # [0, 1] (shifted families)

# fn -> (scipy factory, extra numeric params list [] for none, eval points)
NOPARAM = {
    'legendre': (sp.legendre, X_UNIT),
    'chebyt': (sp.chebyt, X_UNIT),
    'chebyu': (sp.chebyu, X_UNIT),
    'chebyc': (sp.chebyc, X_WIDE),
    'chebys': (sp.chebys, X_WIDE),
    'hermite': (sp.hermite, X_REAL),
    'hermitenorm': (sp.hermitenorm, X_REAL),
    'laguerre': (sp.laguerre, X_REAL),
    'sh_legendre': (sp.sh_legendre, X_SH),
    'sh_chebyt': (sp.sh_chebyt, X_SH),
    'sh_chebyu': (sp.sh_chebyu, X_SH),
}
PARAM = {
    'genlaguerre': (sp.genlaguerre, [(0.5,), (1.5,), (2.0,)], X_REAL),
    'gegenbauer': (sp.gegenbauer, [(0.5,), (1.0,), (2.5,)], X_UNIT),
    'jacobi': (sp.jacobi, [(1.5, 0.5), (0.5, 2.0), (2.0, -0.5)], X_UNIT),
    'sh_jacobi': (sp.sh_jacobi, [(2.0, 1.0), (3.0, 1.5)], X_SH),
}

RTOL = 1e-9
cases = []


def add(fn, args, P, xs):
    cases.append({
        'fn': fn,
        'args': list(args),
        'c': [float(v) for v in P.c],
        'xs': xs,
        'ys': [float(P(x)) for x in xs],
        'rtol': RTOL,
    })


for fn, (fac, xs) in NOPARAM.items():
    for n in range(0, 7):
        add(fn, [n], fac(n), xs)
for fn, (fac, plist, xs) in PARAM.items():
    for pm in plist:
        for n in range(0, 6):
            add(fn, [n, *pm], fac(n, *pm), xs)

out = {'module': 'special', 'scipy': __import__('scipy').__version__, 'orthopoly_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'special_orthopoly: {len(cases)} orthopoly cases')
