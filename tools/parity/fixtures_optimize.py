#!/usr/bin/env python3
"""scipy.optimize parity fixtures for the pure-PHP solvers (brentq, newton, minimize, fmin). The objective is a
named function the runner rebuilds in PHP (poly / cos(x)-x / Rosenbrock), so a callable need not be serialised.
These are backend-neutral iterative algorithms (no BLAS), host-independent; no pinned environment required."""
import json, math, os
import numpy as np
import scipy.optimize as so

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'optimize.json')


def rosen(x):
    return float(sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2 for i in range(len(x) - 1)))


POLY = [-5.0, -2.0, 0.0, 1.0]   # x^3 - 2x - 5


def polyf(c):
    return lambda x: float(sum(ck * x ** k for k, ck in enumerate(c)))


r, _ = so.brentq(polyf(POLY), 2, 3, full_output=True)
r2, _ = so.brentq(lambda x: math.cos(x) - x, -2, 4, full_output=True)
cases = [
    {'fn': 'brentq', 'obj': {'kind': 'poly', 'c': POLY}, 'a': 2.0, 'b': 3.0, 'expect': r, 'rtol': 1e-12},
    {'fn': 'brentq', 'obj': {'kind': 'cosx'}, 'a': -2.0, 'b': 4.0, 'expect': r2, 'rtol': 1e-12},
    {'fn': 'bisect', 'obj': {'kind': 'poly', 'c': POLY}, 'a': 2.0, 'b': 3.0, 'expect': float(so.bisect(polyf(POLY), 2, 3)), 'rtol': 1e-11},
    {'fn': 'bisect', 'obj': {'kind': 'cosx'}, 'a': -2.0, 'b': 4.0, 'expect': float(so.bisect(lambda x: math.cos(x) - x, -2, 4)), 'rtol': 1e-11},
    {'fn': 'brenth', 'obj': {'kind': 'poly', 'c': POLY}, 'a': 2.0, 'b': 3.0, 'expect': float(so.brenth(polyf(POLY), 2, 3)), 'rtol': 1e-12},
    {'fn': 'brenth', 'obj': {'kind': 'cosx'}, 'a': -2.0, 'b': 4.0, 'expect': float(so.brenth(lambda x: math.cos(x) - x, -2, 4)), 'rtol': 1e-12},
    {'fn': 'ridder', 'obj': {'kind': 'poly', 'c': POLY}, 'a': 2.0, 'b': 3.0, 'expect': float(so.ridder(polyf(POLY), 2, 3)), 'rtol': 1e-11},
    {'fn': 'ridder', 'obj': {'kind': 'cosx'}, 'a': -2.0, 'b': 4.0, 'expect': float(so.ridder(lambda x: math.cos(x) - x, -2, 4)), 'rtol': 1e-11},
    {'fn': 'toms748', 'obj': {'kind': 'poly', 'c': POLY}, 'a': 2.0, 'b': 3.0, 'expect': float(so.toms748(polyf(POLY), 2, 3)), 'rtol': 1e-9},
    {'fn': 'toms748', 'obj': {'kind': 'cosx'}, 'a': -2.0, 'b': 4.0, 'expect': float(so.toms748(lambda x: math.cos(x) - x, -2, 4)), 'rtol': 1e-9},
    {'fn': 'root_scalar', 'obj': {'kind': 'poly', 'c': POLY}, 'method': 'brentq', 'bracket': [2.0, 3.0], 'expect': float(so.root_scalar(polyf(POLY), method='brentq', bracket=[2.0, 3.0]).root), 'rtol': 1e-12},
    {'fn': 'root_scalar', 'obj': {'kind': 'cosx'}, 'method': 'ridder', 'bracket': [-2.0, 4.0], 'expect': float(so.root_scalar(lambda x: math.cos(x) - x, method='ridder', bracket=[-2.0, 4.0]).root), 'rtol': 1e-11},
    {'fn': 'root_scalar', 'obj': {'kind': 'poly', 'c': POLY}, 'method': 'newton', 'x0': 1.5, 'expect': float(so.root_scalar(polyf(POLY), method='newton', x0=1.5).root), 'rtol': 1e-10},
    {'fn': 'fminbound', 'obj': {'kind': 'poly', 'c': [2.0, -3.0, 1.0]}, 'a': 0.0, 'b': 3.0, 'expect': float(so.fminbound(polyf([2.0, -3.0, 1.0]), 0, 3)), 'rtol': 1e-7},
    {'fn': 'fminbound', 'obj': {'kind': 'poly', 'c': [5.0, -4.0, 1.0]}, 'a': 0.0, 'b': 5.0, 'expect': float(so.fminbound(polyf([5.0, -4.0, 1.0]), 0, 5)), 'rtol': 1e-7},
    {'fn': 'minimize_scalar', 'obj': {'kind': 'poly', 'c': [2.0, -3.0, 1.0]}, 'expect': float(so.minimize_scalar(polyf([2.0, -3.0, 1.0])).x), 'rtol': 1e-6},
    {'fn': 'minimize_scalar', 'obj': {'kind': 'poly', 'c': [5.0, -4.0, 1.0]}, 'bracket': [0.0, 3.0], 'expect': float(so.minimize_scalar(polyf([5.0, -4.0, 1.0]), bracket=(0.0, 3.0)).x), 'rtol': 1e-6},
    {'fn': 'minimize_scalar', 'obj': {'kind': 'poly', 'c': [5.0, -4.0, 1.0]}, 'bounds': [0.0, 5.0], 'method': 'bounded', 'expect': float(so.minimize_scalar(polyf([5.0, -4.0, 1.0]), bounds=(0.0, 5.0), method='bounded').x), 'rtol': 1e-6},
    {'fn': 'newton', 'obj': {'kind': 'poly', 'c': POLY}, 'x0': 1.5, 'expect': float(so.newton(polyf(POLY), 1.5)), 'rtol': 1e-12},
    {'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [1.3, 0.7, 0.8, 1.9, 1.2], 'method': 'Nelder-Mead',
     'expect': list(so.minimize(rosen, [1.3, 0.7, 0.8, 1.9, 1.2], method='Nelder-Mead').x), 'rtol': 1e-9},
    {'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [-1.2, 1.0], 'method': 'BFGS',
     'expect': list(so.minimize(rosen, [-1.2, 1.0], method='BFGS').x), 'rtol': 1e-4},
    {'fn': 'fmin', 'obj': {'kind': 'rosen'}, 'x0': [1.3, 0.7, 0.8, 1.9, 1.2],
     'expect': list(so.fmin(rosen, [1.3, 0.7, 0.8, 1.9, 1.2], disp=False)), 'rtol': 1e-9},
]

# leastsq: linear and exponential residuals (data embedded so the runner can rebuild the residual in PHP)
_lx = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
_ly = [1.2, 3.1, 5.0, 6.9, 9.1, 11.0]
cases.append({'fn': 'leastsq', 'obj': {'kind': 'linresid', 'x': _lx, 'y': _ly}, 'x0': [0.0, 0.0],
              'expect': list(so.leastsq(lambda p: [p[0] * x + p[1] - y for x, y in zip(_lx, _ly)], [0.0, 0.0])[0]), 'rtol': 1e-6})
_ex = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
_ey = [2.0 * math.exp(0.5 * x) + 1.0 for x in _ex]
cases.append({'fn': 'leastsq', 'obj': {'kind': 'expresid', 'x': _ex, 'y': _ey}, 'x0': [1.0, 0.3, 0.5],
              'expect': list(so.leastsq(lambda p: [p[0] * math.exp(p[1] * x) + p[2] - y for x, y in zip(_ex, _ey)], [1.0, 0.3, 0.5])[0]), 'rtol': 1e-4})

# least_squares (unbounded): same minimiser as leastsq, verified via .x
cases.append({'fn': 'least_squares', 'obj': {'kind': 'linresid', 'x': _lx, 'y': _ly}, 'x0': [0.0, 0.0],
              'expect': list(so.least_squares(lambda p: [p[0] * x + p[1] - y for x, y in zip(_lx, _ly)], [0.0, 0.0]).x), 'rtol': 1e-6})
cases.append({'fn': 'least_squares', 'obj': {'kind': 'expresid', 'x': _ex, 'y': _ey}, 'x0': [1.0, 0.3, 0.5],
              'expect': list(so.least_squares(lambda p: [p[0] * math.exp(p[1] * x) + p[2] - y for x, y in zip(_ex, _ey)], [1.0, 0.3, 0.5]).x), 'rtol': 1e-4})

# nnls: non-negative least squares (unique solution; A and b embedded, runner calls LeastSquares::nnls)
for _A, _b in (([[1., 0.], [1., 0.], [0., 1.]], [2., 1., 1.]),
               ([[1., 1., 0.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]], [1., 2., 3., 2.]),
               ([[0.6, 0.2, 0.1], [0.1, 0.8, 0.3], [0.4, 0.1, 0.9], [0.2, 0.5, 0.2], [0.7, 0.3, 0.4]], [0.3, 0.9, 1.1, 0.5, 0.8])):
    cases.append({'fn': 'nnls', 'A': _A, 'b': _b,
                  'expect': list(so.nnls(np.array(_A), np.array(_b))[0]), 'rtol': 1e-7})

# fixed_point: scalar fixed point via Aitken del2 (matches scipy iterate-for-iterate)
cases.append({'fn': 'fixed_point', 'obj': {'kind': 'lorentz'}, 'x0': 0.5,
              'expect': float(so.fixed_point(lambda x: 1.0 / (1.0 + x * x), 0.5)), 'rtol': 1e-9})
cases.append({'fn': 'fixed_point', 'obj': {'kind': 'gauss'}, 'x0': 0.4,
              'expect': float(so.fixed_point(lambda x: math.exp(-x * x), 0.4)), 'rtol': 1e-9})

# bracket: downhill minimum bracketing (port of scipy.optimize.bracket)
_br0, _br1, _br2 = so.bracket(polyf([2.0, -3.0, 1.0]), 0.0, 1.0)[:3]
cases.append({'fn': 'bracket', 'obj': {'kind': 'poly', 'c': [2.0, -3.0, 1.0]}, 'a': 0.0, 'b': 1.0,
              'expect': [float(_br0), float(_br1), float(_br2)], 'rtol': 1e-9})

# root / fsolve: a contractive 2-D nonlinear system (unique root)
def _vecsys(x):
    return [x[0] - 0.5 * math.cos(x[1]) - 0.5, x[1] - 0.5 * math.sin(x[0]) - 0.3]
cases.append({'fn': 'root', 'obj': {'kind': 'vecsys'}, 'x0': [0.0, 0.0], 'expect': list(so.root(_vecsys, [0.0, 0.0]).x), 'rtol': 1e-7})
cases.append({'fn': 'fsolve', 'obj': {'kind': 'vecsys'}, 'x0': [0.0, 0.0], 'expect': list(so.fsolve(_vecsys, [0.0, 0.0])), 'rtol': 1e-7})

# lsq_linear (unbounded): the ordinary least-squares solution
for _A, _b in (([[1., 0.], [1., 1.], [1., 2.], [1., 3.]], [1., 2., 2.5, 4.]),
               ([[0.6, 0.2, 0.1], [0.1, 0.8, 0.3], [0.4, 0.1, 0.9], [0.2, 0.5, 0.2], [0.7, 0.3, 0.4]], [0.3, 0.9, 1.1, 0.5, 0.8])):
    cases.append({'fn': 'lsq_linear', 'A': _A, 'b': _b,
                  'expect': list(so.lsq_linear(np.array(_A), np.array(_b)).x), 'rtol': 1e-6})

# linear_sum_assignment: optimal assignment (unique optimum -> exact index arrays)
for _C, _mx in (([[4., 1, 3], [2, 0, 5], [3, 2, 2]], False),
                ([[9., 2, 7, 8], [6, 4, 3, 7], [5, 8, 1, 8]], False),
                ([[0.6, 0.2, 0.9], [0.1, 0.8, 0.3], [0.4, 0.5, 0.2], [0.7, 0.1, 0.6]], False),
                ([[4., 1, 3], [2, 0, 5], [3, 2, 2]], True)):
    _r, _c = so.linear_sum_assignment(np.array(_C), maximize=_mx)
    cases.append({'fn': 'linear_sum_assignment', 'cost': _C, 'maximize': _mx,
                  'expect_row': list(map(int, _r)), 'expect_col': list(map(int, _c))})

# brent / golden: scalar minimisers on a valid downhill 3-point bracket; fmin_bfgs: Rosenbrock
for _fn in ('brent', 'golden'):
    cases.append({'fn': _fn, 'obj': {'kind': 'poly', 'c': [4.0, -4.0, 1.0]}, 'bracket': [1.0, 2.0, 3.0],
                  'expect': float(getattr(so, _fn)(polyf([4.0, -4.0, 1.0]), brack=(1.0, 2.0, 3.0))), 'rtol': 1e-6})
    cases.append({'fn': _fn, 'obj': {'kind': 'sinx'}, 'bracket': [-3.0, -1.5, 0.0],
                  'expect': float(getattr(so, _fn)(math.sin, brack=(-3.0, -1.5, 0.0))), 'rtol': 1e-6})
cases.append({'fn': 'fmin_bfgs', 'obj': {'kind': 'rosen'}, 'x0': [-1.2, 1.0],
              'expect': list(map(float, so.fmin_bfgs(rosen, [-1.2, 1.0], disp=False))), 'rtol': 1e-4})

# rosen family: pure-array test function, gradient, Hessian and Hessian-vector product
for _x in ([1.3, 0.7, 0.8, 1.9, 1.2], [-1.2, 1.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0]):
    cases.append({'fn': 'rosen', 'x': _x, 'expect': float(so.rosen(np.array(_x))), 'rtol': 1e-10})
    cases.append({'fn': 'rosen_der', 'x': _x, 'expect': list(map(float, so.rosen_der(np.array(_x)))), 'rtol': 1e-10})
    cases.append({'fn': 'rosen_hess', 'x': _x, 'expect': [list(map(float, r)) for r in so.rosen_hess(np.array(_x))], 'rtol': 1e-10})
    _p = [0.5, -1.0, 2.0, 0.3, 1.5][:len(_x)]
    cases.append({'fn': 'rosen_hess_prod', 'x': _x, 'p': _p, 'expect': list(map(float, so.rosen_hess_prod(np.array(_x), np.array(_p)))), 'rtol': 1e-10})

out = {'module': 'optimize', 'scipy': __import__('scipy').__version__, 'solver_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'optimize: {len(cases)} solver cases')
