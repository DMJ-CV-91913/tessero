#!/usr/bin/env python3
r"""scipy.optimize.broyden1 / broyden2 / anderson / diagbroyden parity fixtures.

Deterministic quasi-Newton nonlinear-system solvers (NO RNG). The PHP port
(Tessero\Optimize\NonlinSolve) reproduces scipy.optimize._nonlin.nonlin_solve
(inexact-Newton driver + Armijo line search) with each Jacobian approximation
(BroydenFirst / BroydenSecond / Anderson / DiagBroyden) iterate-for-iterate; the
de-risk (VERIFY.py) confirmed bit-exact iterate matching against scipy over the
well-behaved systems below.

Each residual function F(x) -> list[float] is NAMED (a 'vecsys_*' kind), rebuilt
in PHP by makeObjective() so no callable is serialised. The root of each covered
system is unique, so the comparison is on the converged x only (not iteration
internals). Tolerance is rtol 1e-6: scipy's own broyden/anderson/diagbroyden
default tolerance (f_tol = eps**(1/3)) leaves ~1e-6..1e-7 slack in x, and the
dense-collapse code path in BroydenFirst accumulates host-/libm-dependent FP
differences at that same level, so 1e-6 is the robust both-backend tolerance
(documented in wiring.txt). Most cases in fact agree with scipy to full double
precision; diagbroyden is the loosest.

Backend-neutral pure PHP (no BLAS), host-independent. Merged into optimize.json
under the existing 'solver_cases' key (consumed by run-tests.php via runSolverCases)."""
import json
import math
import os

from scipy.optimize import broyden1, broyden2, anderson, diagbroyden

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'optimize.json')


# Named residual systems. Each must mirror the matching makeObjective() 'vecsys_*' kind in run-tests.php.
def rotten(x):
    return [x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]]


def quad2(x):
    return [x[0] ** 2 + x[1] ** 2 - 4.0, x[0] - x[1]]


def lin2(x):
    return [2.0 * x[0] + x[1] - 5.0, x[0] + 3.0 * x[1] - 10.0]


def trig2(x):
    return [x[0] - 0.5 * math.cos(x[1]) - 0.5, x[1] - 0.5 * math.sin(x[0]) - 0.3]


def lin3(x):
    return [3 * x[0] + x[1] - x[2] - 2.0, x[0] + 4 * x[1] + x[2] - 6.0, -x[0] + x[1] + 5 * x[2] - 5.0]


SYSTEMS = {
    'vecsys_rotten': (rotten, [0.0, 0.0]),
    'vecsys_quad2':  (quad2,  [1.0, 1.0]),
    'vecsys_lin2':   (lin2,   [0.0, 0.0]),
    'vecsys_trig2':  (trig2,  [0.0, 0.0]),
    'vecsys_lin3':   (lin3,   [0.0, 0.0, 0.0]),
}

SOLVERS = {
    'broyden1': broyden1,
    'broyden2': broyden2,
    'anderson': anderson,
    'diagbroyden': diagbroyden,
}

nonlin_cases = []
for fn_name, solver in SOLVERS.items():
    for kind, (F, x0) in SYSTEMS.items():
        root = list(map(float, solver(F, list(x0))))
        nonlin_cases.append({
            'fn': fn_name,
            'obj': {'kind': kind},     # rebuilt in PHP via makeObjective() named kind
            'x0': x0,
            'expect': root,            # converged root (unique)
            'rtol': 1e-6,
        })

if os.path.exists(OUT):
    with open(OUT) as f:
        data = json.load(f)
    data.setdefault('solver_cases', [])
    nonlin_fns = set(SOLVERS)
    # drop any stale nonlin cases, then append the fresh ones (idempotent re-runs)
    data['solver_cases'] = [c for c in data['solver_cases'] if c.get('fn') not in nonlin_fns]
    data['solver_cases'].extend(nonlin_cases)
else:
    data = {'module': 'optimize', 'scipy': __import__('scipy').__version__, 'solver_cases': nonlin_cases}

with open(OUT, 'w') as f:
    json.dump(data, f, separators=(',', ':'))
print(f'optimize: +{len(nonlin_cases)} nonlin (broyden1/broyden2/anderson/diagbroyden) cases')
