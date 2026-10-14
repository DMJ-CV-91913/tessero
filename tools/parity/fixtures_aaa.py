#!/usr/bin/env python3
"""scipy.interpolate.AAA fixtures (pure-PHP Tessero\\Interpolate\\AAA, the adaptive Antoulas-Anderson greedy
barycentric rational approximation): construct from samples (x, y) then evaluate the approximant at query points.
Real-valued, well-conditioned regime (smooth functions, default rtol) where the smallest singular value of the
Loewner matrix is simple and no Froissart-doublet clean-up is triggered -- so the greedy support-point order and
the SVD-derived weights reproduce SciPy exactly (the weight vector is unique up to sign, but the evaluated value
N/D is sign-invariant). Each case records both the approximant values and the number of selected support points,
so parity covers the greedy iteration count as well as the values. Host-independent (value parity through the
shared LAPACK dgesdd SVD kernel, identical on both backends)."""
import json
import os

import numpy as np
from scipy.interpolate import AAA

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'interpolate_aaa.json')
cases = []


def case(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = AAA(x, y)
    # evaluate at generic interior points (not the sample abscissae) to exercise the barycentric form
    q = np.linspace(float(x.min()), float(x.max()), 41)
    cases.append({'x': x.tolist(), 'y': y.tolist(), 'query': q.tolist(),
                  'expect': [float(v) for v in r(q)],
                  'nsupport': int(len(r.support_points))})


x = np.linspace(-1.0, 1.0, 100);           case(x, np.exp(x))
x = np.linspace(0.1, 3.0, 80);             case(x, np.log(x))
x = np.linspace(-1.0, 1.0, 120);           case(x, 1.0 / (1.0 + 25.0 * x ** 2))   # Runge
x = np.linspace(-2.0, 2.0, 150);           case(x, np.tanh(5.0 * x))
x = np.linspace(0.0, 2.0 * np.pi, 90);     case(x, np.sin(3.0 * x))
x = np.linspace(-1.0, 1.0, 100);           case(x, np.sqrt(x ** 2 + 0.01))        # near-singular
x = np.linspace(0.5, 5.0, 110);            case(x, 1.0 / x)
x = np.linspace(-3.0, 3.0, 130);           case(x, np.cos(x) * np.exp(-0.2 * x ** 2))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as fh:
    json.dump({'aaa_cases': cases}, fh)
print(f'wrote {len(cases)} AAA cases -> {OUT}')
