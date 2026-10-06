#!/usr/bin/env python3
"""scipy.integrate.quad parity fixtures. The integrand is a named function the runner rebuilds in PHP (poly /
exp(-x^2) / sin / 1/(1+x^2)); only the integral value is compared. Pure-PHP adaptive quadrature, host-independent."""
import json, math, os
import scipy.integrate as si

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'integrate.json')


def polyf(c):
    return lambda x: sum(ck * x ** k for k, ck in enumerate(c))


def quad_case(obj, a, b, func):
    return {'fn': 'quad', 'obj': obj, 'a': a, 'b': b, 'expect': si.quad(func, a, b)[0], 'rtol': 1e-8}


cases = [
    quad_case({'kind': 'poly', 'c': [1.0, 2.0, 3.0]}, 0.0, 2.0, polyf([1.0, 2.0, 3.0])),      # 1 + 2x + 3x^2
    quad_case({'kind': 'poly', 'c': [0.0, 0.0, 0.0, 0.0, 1.0]}, -1.0, 2.0, polyf([0, 0, 0, 0, 1])),  # x^4
    quad_case({'kind': 'gauss'}, -3.0, 3.0, lambda x: math.exp(-x * x)),
    quad_case({'kind': 'gauss'}, 0.0, 1.0, lambda x: math.exp(-x * x)),
    quad_case({'kind': 'sinx'}, 0.0, math.pi, math.sin),
    quad_case({'kind': 'lorentz'}, -5.0, 5.0, lambda x: 1.0 / (1.0 + x * x)),
]

out = {'module': 'integrate', 'scipy': __import__('scipy').__version__, 'solver_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'integrate: {len(cases)} quad cases')
