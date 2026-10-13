#!/usr/bin/env python3
r"""scipy.integrate.nsum parity fixtures (FINITE limits only). Term function is a named kind the PHP runner
rebuilds; only the sum value is compared. Tessero\Integrate\Nsum evaluates the direct term-by-term sum,
matching scipy's _direct to ~1 ULP (sequential vs pairwise reduction). Host-independent."""
import json, os
import numpy as np
from scipy.integrate import nsum

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'integrate_nsum.json')

def case(kind, a, b, step, func, params=None, rtol=1e-10):
    r = nsum(func, a, b, step=step)
    c = {'fn': 'nsum', 'kind': kind, 'a': float(a), 'b': float(b), 'step': float(step), 'expect': float(r.sum), 'rtol': rtol}
    if params:
        c['params'] = params
    return c

cases = [
    case('inv_pow', 1, 1000, 1, lambda i: 1.0 / i ** 2, {'p': 2.0}),
    case('inv_pow', 1, 500, 1, lambda i: 1.0 / i ** 3, {'p': 3.0}),
    case('geom', 1, 60, 1, lambda i: 0.5 ** i, {'r': 0.5}),
    case('linear', 0, 100, 1, lambda i: 2.0 * i + 1.0, {'m': 2.0, 'c': 1.0}),
    case('inv_pow', 2, 200, 2, lambda i: 1.0 / i ** 2, {'p': 2.0}),
    case('expneg', 1, 300, 1, lambda i: np.exp(-0.1 * i), {'rate': 0.1}),
    case('inv_pow', 1, 50, 1, lambda i: 1.0 / i, {'p': 1.0}),
]
out = {'module': 'integrate', 'scipy': __import__('scipy').__version__, 'nsum_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print('integrate_nsum: %d cases' % len(cases))
