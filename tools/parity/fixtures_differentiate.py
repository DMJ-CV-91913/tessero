#!/usr/bin/env python3
"""Reference fixtures for scipy.differentiate.{derivative,jacobian,hessian}.

STRICT JSON only: every expected value is a finite float (smooth test functions on valid domains -- no
log(0), no 1/0). Cases are keyed by op + named function kind + evaluation point + expected result; the PHP
run-tests.php `differentiate_cases` branch rebuilds the same named callables and compares:
  derivative          rtol 1e-10   (scalar df)
  jacobian / hessian  rtol 1e-9    (matrix, component-wise)
"""
import json
import numpy as np
import scipy.differentiate as D

OUT = '../../tests/fixtures/parity/differentiate.json'


# --- scalar f: R -> R  (derivative) ---
def scalar_f(kind, c):
    if kind == 'poly':      return lambda x: sum(ck * x**i for i, ck in enumerate(c))
    if kind == 'expx':      return lambda x: np.exp(x)
    if kind == 'sinx':      return lambda x: np.sin(x)
    if kind == 'cosx':      return lambda x: np.cos(x)
    if kind == 'sincos':    return lambda x: np.sin(x) * np.cos(2.0 * x)
    if kind == 'expquad':   return lambda x: np.exp(0.3 * x) + x * x
    raise ValueError(kind)


# --- scalar f: R^m -> R  (hessian) ---
def scalarvec_f(kind):
    if kind == 'sphere':  return lambda v: sum(vi * vi for vi in v)
    if kind == 'qcross':  return lambda v: v[0]**2 + 3.0 * v[1]**2 + 2.0 * v[0] * v[1]
    if kind == 'trigq':   return lambda v: np.sin(v[0]) * np.cos(v[1]) + 0.5 * (v[0]**2 + v[1]**2)
    if kind == 'prodq':   return lambda v: v[0] * v[1] * v[2] + v[0]**2 * v[2]
    raise ValueError(kind)


# --- vector f: R^m -> R^k  (jacobian) ---
def vec_f(kind):
    if kind == 'vlin':   return lambda v: [2.0*v[0] + v[1], v[0] - 3.0*v[1], v[0]*v[1]]
    if kind == 'vtrig':  return lambda v: [np.sin(v[0]) + v[1]**2, np.cos(v[1]) * v[0]]
    if kind == 'vexp':   return lambda v: [np.exp(v[0]) * v[1], v[0] + v[1]**3, v[0]-v[1]]
    if kind == 'vquad3': return lambda v: [v[0]**2 + v[1], v[1]*v[2], v[0]+v[2]**2]
    raise ValueError(kind)


def finite(x):
    return np.all(np.isfinite(np.asarray(x, dtype=float)))


def deriv_case(kind, x, c=None):
    c = c or []
    df = float(D.derivative(scalar_f(kind, c), x).df)
    assert finite(df), (kind, x)
    return {'op': 'derivative', 'kind': kind, 'c': c, 'x': float(x), 'expect': df, 'rtol': 1e-10}


def hess_case(kind, x):
    f = scalarvec_f(kind)
    fh = lambda z: f([np.asarray(z, dtype=float)[i] for i in range(np.asarray(z).shape[0])])
    H = np.asarray(D.hessian(fh, np.asarray(x, dtype=float)).ddf, dtype=float)
    assert finite(H), (kind, x)
    return {'op': 'hessian', 'kind': kind, 'x': [float(v) for v in x], 'expect': H.tolist(), 'rtol': 1e-9}


def jac_case(kind, x):
    f = vec_f(kind)
    fj = lambda z: np.asarray(f([np.asarray(z, dtype=float)[i] for i in range(np.asarray(z).shape[0])]), dtype=float)
    J = np.asarray(D.jacobian(fj, np.asarray(x, dtype=float)).df, dtype=float)
    assert finite(J), (kind, x)
    return {'op': 'jacobian', 'kind': kind, 'x': [float(v) for v in x], 'expect': J.tolist(), 'rtol': 1e-9}


cases = [
    deriv_case('poly', 0.7, [0.0, 1.0, -2.0, 0.5, 3.0]),
    deriv_case('poly', -1.3, [2.0, -1.0, 4.0]),
    deriv_case('poly', 2.4, [1.0, 0.0, 0.0, 1.0]),
    deriv_case('expx', 0.4),
    deriv_case('expx', -1.1),
    deriv_case('sinx', 1.1),
    deriv_case('cosx', -0.6),
    deriv_case('sincos', 0.9),
    deriv_case('expquad', 1.4),

    hess_case('sphere', [1.0, -2.0]),
    hess_case('qcross', [0.5, 1.5]),
    hess_case('trigq', [0.7, -0.4]),
    hess_case('prodq', [1.1, 0.6, -0.8]),

    jac_case('vlin', [1.0, -0.5]),
    jac_case('vtrig', [0.6, 1.2]),
    jac_case('vexp', [0.3, -0.7]),
    jac_case('vquad3', [1.0, 2.0, -1.0]),
]

out = {'module': 'differentiate', 'scipy': __import__('scipy').__version__, 'differentiate_cases': cases}
with open(OUT, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'), allow_nan=False)   # allow_nan=False => NaN/Inf raise, never emitted
print('differentiate: %d cases (derivative/jacobian/hessian)' % len(cases))
