#!/usr/bin/env python3
"""scipy.integrate.tanhsinh parity fixtures (finite limits, scalar, log=False).

The integrand is a NAMED function the PHP runner rebuilds via makeObjective (poly / gauss=exp(-x^2) /
sinx / lorentz / invsqrt=1/sqrt(x) / logx=log(x)); only result.integral is compared (rtol ~1e-13). This mirrors
the existing integrate.json 'quad' solver_cases pattern but carries a struct-result routine, so it is written to
a SEPARATE fixture file (integrate_tanhsinh.json) under the 'tanhsinh_cases' key to keep the quad fixtures intact.
Pure-PHP double-exponential quadrature, host-independent (no kernel, no LAPACK)."""
import json, math, os
import numpy as np
from scipy.integrate import tanhsinh

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'integrate_tanhsinh.json')


def polyf(c):
    return lambda x: sum(ck * x ** k for k, ck in enumerate(c))


# Named integrands matching makeObjective in run-tests.php (two new kinds added: invsqrt, logx).
# scipy.integrate.tanhsinh passes a numpy array of abscissae, so these use numpy ufuncs; the PHP
# makeObjective equivalents are scalar (the PHP routine calls f() one abscissa at a time).
FUNCS = {
    'gauss': lambda x: np.exp(-x * x),
    'sinx': np.sin,
    'lorentz': lambda x: 1.0 / (1.0 + x * x),
    'invsqrt': lambda x: 1.0 / np.sqrt(x),
    'logx': np.log,
}


def func_for(obj):
    if obj['kind'] == 'poly':
        return polyf(obj['c'])
    return FUNCS[obj['kind']]


def case(obj, a, b):
    f = func_for(obj)
    r = tanhsinh(f, a, b)                 # finite limits, scalar; compare .integral only
    return {'fn': 'tanhsinh', 'obj': obj, 'a': a, 'b': b, 'expect': float(r.integral), 'rtol': 1e-13}


cases = [
    case({'kind': 'poly', 'c': [1.0, 2.0, 3.0]}, 0.0, 2.0),        # 1 + 2x + 3x^2  -> smooth
    case({'kind': 'poly', 'c': [0.0, -2.0, 0.0, 1.0]}, -1.0, 2.0), # x^3 - 2x
    case({'kind': 'gauss'}, 0.0, 3.0),                             # exp(-x^2)
    case({'kind': 'sinx'}, 0.0, math.pi),                          # sin(x) -> 2
    case({'kind': 'lorentz'}, 0.0, 1.0),                           # 1/(1+x^2) -> pi/4
    case({'kind': 'invsqrt'}, 0.0, 1.0),                           # 1/sqrt(x) -> 2  (endpoint singular)
    case({'kind': 'logx'}, 0.0, 1.0),                              # log(x) -> -1    (endpoint singular)
]

out = {'module': 'integrate', 'scipy': __import__('scipy').__version__, 'tanhsinh_cases': cases}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'))
print(f'integrate tanhsinh: {len(cases)} cases -> {OUT}')
