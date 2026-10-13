#!/usr/bin/env python3
r"""scipy.integrate.dblquad / tplquad / nquad parity fixtures. Nested adaptive Gauss-Kronrod quadrature over
named integrand + limit kinds the PHP runner rebuilds (Tessero\Integrate\NestedQuad reuses the verified
Quad::quad 1-D primitive). Only the integral value is compared, at scipy's adaptive tolerance. Pure-PHP,
host-independent."""
import json, os
import numpy as np
import scipy.integrate as si

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'integrate_nestedquad.json')

def dbl(kind, a, b, glo, ghi, func, gfun, hfun):
    v, _ = si.dblquad(func, a, b, gfun, hfun)
    return {'fn': 'dblquad', 'kind': kind, 'a': a, 'b': b, 'glo': glo, 'ghi': ghi, 'expect': float(v), 'rtol': 1e-7}

def tpl(kind, a, b, glo, ghi, qlo, qhi, func, gfun, hfun, qfun, rfun):
    v, _ = si.tplquad(func, a, b, gfun, hfun, qfun, rfun)
    return {'fn': 'tplquad', 'kind': kind, 'a': a, 'b': b, 'glo': glo, 'ghi': ghi, 'qlo': qlo, 'qhi': qhi, 'expect': float(v), 'rtol': 1e-7}

def nq(kind, ranges, func):
    v, _ = si.nquad(func, ranges)
    return {'fn': 'nquad', 'kind': kind, 'ranges': ranges, 'expect': float(v), 'rtol': 1e-7}

cases = [
    dbl('xy', 0.0, 1.0, {'const': 0.0}, {'const': 1.0}, lambda y, x: x * y, lambda x: 0, lambda x: 1),
    dbl('gauss2d', 0.0, 2.0, {'const': 0.0}, {'const': 2.0}, lambda y, x: np.exp(-(x * x + y * y)), lambda x: 0, lambda x: 2),
    dbl('triangle', 0.0, 1.0, {'const': 0.0}, {'affine': [0.0, 1.0]}, lambda y, x: 1.0, lambda x: 0, lambda x: x),
    dbl('affinef', 0.0, 3.0, {'affine': [0.0, 1.0]}, {'affine': [0.0, 2.0]}, lambda y, x: x + y, lambda x: x, lambda x: 2 * x),
    dbl('sincos', 0.0, np.pi, {'const': 0.0}, {'const': np.pi / 2}, lambda y, x: np.sin(x) * np.cos(y), lambda x: 0, lambda x: np.pi / 2),
    tpl('unitcube', 0.0, 1.0, {'const': 0.0}, {'const': 1.0}, {'const': 0.0}, {'const': 1.0}, lambda z, y, x: 1.0, lambda x: 0, lambda x: 1, lambda x, y: 0, lambda x, y: 1),
    tpl('xyzcube', 0.0, 1.0, {'const': 0.0}, {'const': 1.0}, {'const': 0.0}, {'const': 1.0}, lambda z, y, x: x * y * z, lambda x: 0, lambda x: 1, lambda x, y: 0, lambda x, y: 1),
    tpl('simplex', 0.0, 1.0, {'const': 0.0}, {'affine_x': [1.0, -1.0]}, {'const': 0.0}, {'affine_xy': [1.0, -1.0, -1.0]}, lambda z, y, x: 1.0, lambda x: 0, lambda x: 1 - x, lambda x, y: 0, lambda x, y: 1 - x - y),
    nq('xy', [[0.0, 1.0], [0.0, 1.0]], lambda x, y: x * y),
    nq('xyz', [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], lambda x, y, z: x + y + z),
    nq('gauss2', [[0.0, 2.0], [0.0, 2.0]], lambda x, y: np.exp(-(x * x + y * y))),
]

out = {'module': 'integrate', 'scipy': __import__('scipy').__version__, 'nestedquad_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print('integrate_nestedquad: %d cases' % len(cases))
