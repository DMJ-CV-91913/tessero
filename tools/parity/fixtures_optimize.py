#!/usr/bin/env python3
"""scipy.optimize parity fixtures for the pure-PHP solvers (brentq, newton, minimize, fmin). The objective is a
named function the runner rebuilds in PHP (poly / cos(x)-x / Rosenbrock), so a callable need not be serialised.
These are backend-neutral iterative algorithms (no BLAS), host-independent; no pinned environment required."""
import json, math, os
import fixtures_optimize_classes as _oc
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
    {'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [-1.2, 1.0], 'method': 'CG',
     'expect': list(so.minimize(rosen, [-1.2, 1.0], method='CG', jac=so.rosen_der).x), 'rtol': 1e-4},
    {'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [1.3, 0.7, 0.8, 1.9, 1.2], 'method': 'CG',
     'expect': list(so.minimize(rosen, [1.3, 0.7, 0.8, 1.9, 1.2], method='CG', jac=so.rosen_der).x), 'rtol': 1e-4},
    {'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [-1.2, 1.0], 'method': 'Powell',
     'expect': list(so.minimize(rosen, [-1.2, 1.0], method='Powell').x), 'rtol': 1e-5},
    {'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [1.3, 0.7, 0.8, 1.9, 1.2], 'method': 'Powell',
     'expect': list(so.minimize(rosen, [1.3, 0.7, 0.8, 1.9, 1.2], method='Powell').x), 'rtol': 1e-5},
    {'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [-1.2, 1.0], 'method': 'Newton-CG',
     'expect': list(so.minimize(rosen, [-1.2, 1.0], method='Newton-CG', jac=so.rosen_der, hessp=lambda x, p: so.rosen_hess_prod(x, p)).x), 'rtol': 1e-4},
    {'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [1.3, 0.7, 0.8, 1.9, 1.2], 'method': 'Newton-CG',
     'expect': list(so.minimize(rosen, [1.3, 0.7, 0.8, 1.9, 1.2], method='Newton-CG', jac=so.rosen_der, hessp=lambda x, p: so.rosen_hess_prod(x, p)).x), 'rtol': 1e-4},
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

# isotonic_regression: Pool Adjacent Violators (runner calls LeastSquares::isotonicRegression)
for _y, _w, _inc in (([1., 0.5, 2., 1.5, 3.], None, True), ([8., 4., 6., 2., 9., 1.], None, True),
                     ([8., 4., 6., 2., 9., 1.], [1., 2., 3., 1., 1., 2.], True), ([5., 4., 3., 6., 2.], None, False),
                     ([3., 1., 4., 1., 5., 9., 2., 6.], None, True)):
    cases.append({'fn': 'isotonic_regression', 'y': _y, 'weights': _w, 'increasing': _inc,
                  'expect': list(so.isotonic_regression(_y, weights=_w, increasing=_inc).x), 'rtol': 1e-9})

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
cases.append({'fn': 'fmin_cg', 'obj': {'kind': 'rosen'}, 'x0': [-1.2, 1.0],
              'expect': list(map(float, so.fmin_cg(rosen, [-1.2, 1.0], fprime=so.rosen_der, disp=False))), 'rtol': 1e-4})
cases.append({'fn': 'fmin_powell', 'obj': {'kind': 'rosen'}, 'x0': [-1.2, 1.0],
              'expect': list(map(float, so.fmin_powell(rosen, [-1.2, 1.0], disp=False))), 'rtol': 1e-5})
cases.append({'fn': 'fmin_ncg', 'obj': {'kind': 'rosen'}, 'x0': [-1.2, 1.0],
              'expect': list(map(float, so.fmin_ncg(rosen, [-1.2, 1.0], fprime=so.rosen_der, fhess_p=so.rosen_hess_prod, disp=False))), 'rtol': 1e-4})

# rosen family: pure-array test function, gradient, Hessian and Hessian-vector product
for _x in ([1.3, 0.7, 0.8, 1.9, 1.2], [-1.2, 1.0], [0.5, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0]):
    cases.append({'fn': 'rosen', 'x': _x, 'expect': float(so.rosen(np.array(_x))), 'rtol': 1e-10})
    cases.append({'fn': 'rosen_der', 'x': _x, 'expect': list(map(float, so.rosen_der(np.array(_x)))), 'rtol': 1e-10})
    cases.append({'fn': 'rosen_hess', 'x': _x, 'expect': [list(map(float, r)) for r in so.rosen_hess(np.array(_x))], 'rtol': 1e-10})
    _p = [0.5, -1.0, 2.0, 0.3, 1.5][:len(_x)]
    cases.append({'fn': 'rosen_hess_prod', 'x': _x, 'p': _p, 'expect': list(map(float, so.rosen_hess_prod(np.array(_x), np.array(_p)))), 'rtol': 1e-10})

# check_grad: 2-norm of analytic vs forward-difference gradient (Rosenbrock)
for _x in ([1.3, 0.7, 0.8], [-1.2, 1.0], [0.5, 0.5, 0.5, 0.5]):
    cases.append({'fn': 'check_grad', 'obj': {'kind': 'rosen'}, 'x0': _x,
                  'expect': float(so.check_grad(so.rosen, so.rosen_der, np.array(_x, float))), 'rtol': 1e-9})

# brute (finish=None): grid argmin of Rosenbrock over a box
for _r in ([(-2.0, 2.0), (-2.0, 2.0)], [(-1.5, 2.5), (-1.5, 2.5), (-1.5, 2.5)]):
    _b = np.atleast_1d(so.brute(so.rosen, _r, Ns=20, finish=None))
    cases.append({'fn': 'brute', 'obj': {'kind': 'rosen'}, 'ranges': [list(r) for r in _r], 'Ns': 20,
                  'expect': list(map(float, _b)), 'rtol': 1e-9})

# ===== curve_fit cases — ADD to tools/parity/fixtures_optimize.py =====
# Place AFTER the existing least_squares block (after line ~79), BEFORE the
# `out = {...}` assembly line (~161). Reuses the obj.kind named-model pattern
# (expresid / linresid) already used by the leastsq/least_squares cases; the
# runner's curve_fit arm rebuilds the MODEL f(x,*p) from obj.kind inline
# (a*exp(b*x)+c for expresid, a*x+b for linresid) -- no model is serialised.
#
# Reuses the already-defined data arrays _ex,_ey (3-param exp) and _lx,_ly
# (2-param linear) from the leastsq block above. pcov compared rtol 1e-4,
# popt rtol 1e-6. An absolute_sigma=True case is included so pcov is a
# genuine nonzero matrix (the relative-sigma exp pcov collapses to ~0 on the
# noise-free exp data, which would not exercise the covariance path).

_cf_exp  = so.curve_fit(lambda x, a, b, c: a * np.exp(b * x) + c, np.array(_ex), np.array(_ey), p0=[1.0, 0.3, 0.5])
_cf_expA = so.curve_fit(lambda x, a, b, c: a * np.exp(b * x) + c, np.array(_ex), np.array(_ey), p0=[1.0, 0.3, 0.5], absolute_sigma=True)
_cf_lin  = so.curve_fit(lambda x, a, b: a * x + b, np.array(_lx), np.array(_ly), p0=[0.0, 0.0])

# a*exp(b*x)+c  (3-param, relative sigma -- popt is the real check, pcov ~0)
cases.append({'fn': 'curve_fit', 'obj': {'kind': 'expresid', 'x': _ex, 'y': _ey}, 'p0': [1.0, 0.3, 0.5],
              'absolute_sigma': False,
              'popt': [float(v) for v in _cf_exp[0]],
              'pcov': [[float(v) for v in row] for row in _cf_exp[1]],
              'rtol': 1e-6, 'rtol_pcov': 1e-4})

# a*exp(b*x)+c  (3-param, absolute_sigma=True -- nonzero pcov genuinely checked)
cases.append({'fn': 'curve_fit', 'obj': {'kind': 'expresid', 'x': _ex, 'y': _ey}, 'p0': [1.0, 0.3, 0.5],
              'absolute_sigma': True,
              'popt': [float(v) for v in _cf_expA[0]],
              'pcov': [[float(v) for v in row] for row in _cf_expA[1]],
              'rtol': 1e-6, 'rtol_pcov': 1e-4})

# a*x+b  (2-param model, relative sigma -- real (noisy) data, nonzero pcov)
cases.append({'fn': 'curve_fit', 'obj': {'kind': 'linresid', 'x': _lx, 'y': _ly}, 'p0': [0.0, 0.0],
              'absolute_sigma': False,
              'popt': [float(v) for v in _cf_lin[0]],
              'pcov': [[float(v) for v in row] for row in _cf_lin[1]],
              'rtol': 1e-6, 'rtol_pcov': 1e-4})
# ===== end curve_fit cases =====
cases += _oc.cases()

# --- pkg6 optimize methods (TNC/SLSQP/COBYLA + fmin_*) ---
def rosen(x):
    return float(sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2 for i in range(len(x) - 1)))


rosen_der = so.rosen_der

# Named problems (keyed by 'prob'): objective, optional jac, bounds, and constraints. The PHP runner holds
# the identical definitions in makeConObjective()/makeCons(); here we only need to produce the scipy optimum.
PROBS = {
    # bounds-only (TNC / SLSQP)
    'rosen2':  dict(f=rosen, jac=rosen_der, x0=[-1.2, 1.0], bounds=[(-2, 2), (-2, 2)]),
    'rosen4':  dict(f=rosen, jac=rosen_der, x0=[0.0, 0.0, 0.0, 0.0], bounds=[(-5, 5)] * 4),
    'quad2':   dict(f=lambda x: (x[0] - 2) ** 2 + (x[1] + 3) ** 2, jac=lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] + 3)]), x0=[0, 0], bounds=[(0, 1), (-1, 1)]),
    'quad3':   dict(f=lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[2] + 1) ** 2, jac=lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 2), 2 * (x[2] + 1)]), x0=[0, 0, 0], bounds=None),
    'sphere_lo': dict(f=lambda x: x[0] ** 2 + x[1] ** 2, jac=lambda x: np.array([2 * x[0], 2 * x[1]]), x0=[3, 3], bounds=[(1, None), (2, None)]),
}

# Equality / inequality constrained problems (SLSQP / COBYLA). 'eq'/'ineq' are name lists resolved in PHP.
CPROBS = {
    'sphere_eq':  dict(f=lambda x: x[0] ** 2 + x[1] ** 2, x0=[2, 2], eq=['sum_m1'], ineq=[], bounds=None),
    'qp_tri':     dict(f=lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2, x0=[2, 0], eq=[], ineq=['g1', 'g2', 'g3'], bounds=[(0, None), (0, None)]),
    'qp_tri0':    dict(f=lambda x: (x[0] - 1) ** 2 + (x[1] - 2.5) ** 2, x0=[0, 0], eq=[], ineq=['g1', 'g2', 'g3'], bounds=None),
    'rosen_con':  dict(f=rosen, x0=[0.5, 0.0], eq=[], ineq=['h1', 'h2', 'h3'], bounds=None),
    'sphere_line': dict(f=lambda x: x[0] ** 2 + x[1] ** 2, x0=[2, 2], eq=[], ineq=['line'], bounds=None),
    'uncon':      dict(f=lambda x: (x[0] - 3) ** 2 + (x[1] + 1) ** 2, x0=[0, 0], eq=[], ineq=[], bounds=None),
}

# constraint callables, referenced by name from CPROBS
CFUN = {
    'sum_m1': lambda x: x[0] + x[1] - 1,
    'g1': lambda x: x[0] - 2 * x[1] + 2,
    'g2': lambda x: -x[0] - 2 * x[1] + 6,
    'g3': lambda x: -x[0] + 2 * x[1] + 2,
    'h1': lambda x: 1 - x[0] - 2 * x[1],
    'h2': lambda x: x[0],
    'h3': lambda x: x[1],
    'line': lambda x: x[0] + x[1] - 1,
}


# ----- TNC: bounds-only problems
for prob in ('rosen2', 'rosen4', 'quad2', 'quad3', 'sphere_lo'):
    p = PROBS[prob]
    r = so.minimize(p['f'], p['x0'], method='TNC', jac=p['jac'], bounds=p['bounds'])
    cases.append({'fn': 'minimize_con', 'method': 'TNC', 'prob': prob, 'expect': list(map(float, r.x)), 'rtol': 1e-3})

# ----- SLSQP: bounds-only + constrained
for prob in ('rosen2', 'quad3'):
    p = PROBS[prob]
    r = so.minimize(p['f'], p['x0'], method='SLSQP', jac=p['jac'], bounds=p['bounds'])
    cases.append({'fn': 'minimize_con', 'method': 'SLSQP', 'prob': prob, 'expect': list(map(float, r.x)), 'rtol': 1e-3})
for prob in ('sphere_eq', 'qp_tri', 'uncon'):
    p = CPROBS[prob]
    cons = [{'type': 'eq', 'fun': CFUN[n]} for n in p['eq']] + [{'type': 'ineq', 'fun': CFUN[n]} for n in p['ineq']]
    r = so.minimize(p['f'], p['x0'], method='SLSQP', bounds=p['bounds'], constraints=cons)
    cases.append({'fn': 'minimize_con', 'method': 'SLSQP', 'prob': prob, 'expect': list(map(float, r.x)), 'rtol': 1e-4})

# ----- COBYLA: inequality-constrained (choose starts that converge cleanly, matching scipy's optimum)
for prob in ('sphere_line', 'qp_tri0', 'rosen_con'):
    p = CPROBS[prob]
    cons = [{'type': 'ineq', 'fun': CFUN[n]} for n in p['ineq']]
    r = so.minimize(p['f'], p['x0'], method='COBYLA', constraints=cons)
    cases.append({'fn': 'minimize_con', 'method': 'COBYLA', 'prob': prob, 'expect': list(map(float, r.x)), 'rtol': 2e-2})

# ----- fmin_* wrappers
r = so.fmin_tnc(lambda x: x[0] ** 2 + x[1] ** 2, [3, 3], approx_grad=1, bounds=[(1, None), (2, None)], disp=0)
cases.append({'fn': 'fmin_tnc', 'prob': 'sphere_lo', 'expect': list(map(float, r[0])), 'rtol': 1e-4})

r = so.fmin_slsqp(lambda x: x[0] ** 2 + x[1] ** 2, [2, 2], eqcons=[lambda x: x[0] + x[1] - 1], iprint=0)
cases.append({'fn': 'fmin_slsqp', 'prob': 'sphere_eq', 'expect': list(map(float, r)), 'rtol': 1e-4})

r = so.fmin_cobyla(lambda x: x[0] ** 2 + x[1] ** 2, [2, 2], [lambda x: x[0] + x[1] - 1], rhoend=1e-6, disp=0)
cases.append({'fn': 'fmin_cobyla', 'prob': 'sphere_line', 'expect': list(map(float, np.atleast_1d(r))), 'rtol': 1e-3})


# --- pkg6 trust-region methods ---
# === APPEND to tools/parity/fixtures_optimize.py (after the existing minimize cases) ===
# scipy.optimize.minimize trust-region methods (jac + hess required; rosen family analytic derivatives).
# dogleg needs a positive-definite Hessian, so its rosen start is near the minimum; at an indefinite
# start (e.g. [1.3,...]) scipy's dogleg itself returns status 3 and does not move -- not exercised here.
cases.append({'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [0.8, 0.6], 'method': 'dogleg',
              'expect': list(so.minimize(rosen, [0.8, 0.6], method='dogleg', jac=so.rosen_der, hess=so.rosen_hess).x), 'rtol': 1e-6})
cases.append({'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [-1.2, 1.0], 'method': 'trust-ncg',
              'expect': list(so.minimize(rosen, [-1.2, 1.0], method='trust-ncg', jac=so.rosen_der, hess=so.rosen_hess).x), 'rtol': 1e-4})
cases.append({'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [1.3, 0.7, 0.8, 1.9, 1.2], 'method': 'trust-ncg',
              'expect': list(so.minimize(rosen, [1.3, 0.7, 0.8, 1.9, 1.2], method='trust-ncg', jac=so.rosen_der, hessp=lambda x, p: so.rosen_hess_prod(x, p)).x), 'rtol': 1e-4})
cases.append({'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [-1.2, 1.0], 'method': 'trust-exact',
              'expect': list(so.minimize(rosen, [-1.2, 1.0], method='trust-exact', jac=so.rosen_der, hess=so.rosen_hess).x), 'rtol': 1e-6})
cases.append({'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [1.3, 0.7, 0.8, 1.9, 1.2], 'method': 'trust-exact',
              'expect': list(so.minimize(rosen, [1.3, 0.7, 0.8, 1.9, 1.2], method='trust-exact', jac=so.rosen_der, hess=so.rosen_hess).x), 'rtol': 1e-6})

# --- scipy.optimize.minimize(method='trust-krylov')  (GLTR, _trlib); compare converged x only ---
_tkC = (1.5, -0.5)
def _softbowl(x): return sum((1 + (x[i] - _tkC[i]) ** 2) ** 0.5 - 1 for i in range(2))
def _softbowl_der(x): return np.array([(x[i] - _tkC[i]) / (1 + (x[i] - _tkC[i]) ** 2) ** 0.5 for i in range(2)])
def _softbowl_hess(x): return np.array([[1.0 / (1 + (x[0]-_tkC[0])**2)**1.5, 0.0], [0.0, 1.0 / (1 + (x[1]-_tkC[1])**2)**1.5]])
def _bowl3(x): return (x[0]-1)**2 + (x[1]+2)**2 + 2*(x[2]-0.5)**2
def _bowl3_der(x): return np.array([2*(x[0]-1), 2*(x[1]+2), 4*(x[2]-0.5)])
def _bowl3_hess(x): return np.array([[2.,0,0],[0,2.,0],[0,0,4.]])
def _qbowl(x): return x[0]**2 + 2*x[1]**2 + x[0]*x[1]
def _qbowl_der(x): return np.array([2*x[0]+x[1], 4*x[1]+x[0]])
def _qbowl_hess(x): return np.array([[2.,1.],[1.,4.]])
cases.append({'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [1.3, 0.7], 'method': 'trust-krylov', 'hessp': False,
              'expect': list(map(float, so.minimize(rosen, [1.3, 0.7], method='trust-krylov', jac=so.rosen_der, hess=so.rosen_hess).x)), 'rtol': 1e-5})
cases.append({'fn': 'minimize', 'obj': {'kind': 'rosen'}, 'x0': [1.3, 0.7, 0.8, 1.9, 1.2], 'method': 'trust-krylov', 'hessp': True,
              'expect': list(map(float, so.minimize(rosen, [1.3, 0.7, 0.8, 1.9, 1.2], method='trust-krylov', jac=so.rosen_der, hessp=so.rosen_hess_prod).x)), 'rtol': 1e-5})
cases.append({'fn': 'minimize', 'obj': {'kind': 'softbowl'}, 'x0': [1.0, 1.0], 'method': 'trust-krylov', 'hessp': False,
              'expect': list(map(float, so.minimize(_softbowl, np.array([1.0, 1.0]), method='trust-krylov', jac=_softbowl_der, hess=_softbowl_hess).x)), 'rtol': 1e-5})
cases.append({'fn': 'minimize', 'obj': {'kind': 'bowl3'}, 'x0': [0.0, 0.0, 0.0], 'method': 'trust-krylov', 'hessp': False,
              'expect': list(map(float, so.minimize(_bowl3, np.array([0.0, 0.0, 0.0]), method='trust-krylov', jac=_bowl3_der, hess=_bowl3_hess).x)), 'rtol': 1e-5})
cases.append({'fn': 'minimize', 'obj': {'kind': 'qbowl'}, 'x0': [2.0, -1.0], 'method': 'trust-krylov', 'hessp': False,
              'expect': list(map(float, so.minimize(_qbowl, np.array([2.0, -1.0]), method='trust-krylov', jac=_qbowl_der, hess=_qbowl_hess).x)), 'rtol': 1e-5})

out = {'module': 'optimize', 'scipy': __import__('scipy').__version__, 'solver_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'optimize: {len(cases)} solver cases')
