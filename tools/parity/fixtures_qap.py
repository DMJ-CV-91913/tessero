#!/usr/bin/env python3
"""scipy.optimize.quadratic_assignment(method='faq') parity fixtures for the pure-PHP QuadraticAssignment facade.

FAQ with P0='barycenter' and no RNG is fully deterministic: a Frank-Wolfe relaxation on the Birkhoff polytope
whose only data-dependent branch is the linear-assignment subproblem. For NON-DEGENERATE inputs (asymmetric
integer graphs, random float matrices) that subproblem has a unique optimum at every iteration, so the trajectory
and the final `col_ind` reproduce bit-for-bit in pure PHP and are host-independent.

Symmetric integer inputs are deliberately avoided: they make the gradient degenerate (tied assignment
subproblems), and the tie resolution is BLAS/host-dependent -- scipy itself returns different col_ind across
OPENBLAS_CORETYPE on those -- so they are not portable parity cases.

These cases are appended to the existing optimize.json `solver_cases`; the runner compares col_ind exactly and fun
to rtol 1e-9 under the 'optimize.quadratic_assignment' branch. Run after fixtures_optimize.py.
"""
import json
import os

import numpy as np
import scipy.optimize as so

OUT = os.environ.get('QAP_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'optimize.json')


def rnd(n, seed):
    return np.random.default_rng(seed).random((n, n))


def asym(n, seed):
    # asymmetric integer matrix with zero diagonal (a weighted digraph): non-degenerate FAQ gradients
    g = np.random.default_rng(seed)
    M = g.integers(0, 10, (n, n)).astype(float)
    np.fill_diagonal(M, 0)
    return M


# (A, B, maximize). The first two pairs are scipy's own quadratic_assignment docstring examples.
specs = [
    (np.array([[0, 80, 150, 170], [80, 0, 130, 100], [150, 130, 0, 120], [170, 100, 120, 0]], float),
     np.array([[0, 5, 2, 7], [0, 0, 3, 8], [0, 0, 0, 3], [0, 0, 0, 0]], float), False),
    (np.array([[0, 5, 8, 6], [5, 0, 5, 1], [8, 5, 0, 2], [6, 1, 2, 0]], float),
     np.array([[0, 1, 8, 4], [1, 0, 5, 2], [8, 5, 0, 5], [4, 2, 5, 0]], float), False),
    (asym(5, 101), asym(5, 102), False),
    (asym(6, 103), asym(6, 104), False),
    (rnd(6, 11), rnd(6, 12), False),
    (rnd(7, 13), rnd(7, 14), False),
    (rnd(8, 21), rnd(8, 22), False),
    (rnd(6, 31), rnd(6, 32), True),   # maximize / graph matching
]

qap_cases = []
for A, B, mx in specs:
    res = so.quadratic_assignment(A, B, method='faq', options={'P0': 'barycenter', 'maximize': mx})
    qap_cases.append({
        'fn': 'quadratic_assignment',
        'A': A.tolist(),
        'B': B.tolist(),
        'maximize': bool(mx),
        'maxiter': 30,
        'tol': 0.03,
        'expect_col': [int(x) for x in res.col_ind],
        'fun': float(res.fun),
        'nit': int(res.nit),
        'rtol': 1e-9,
    })

with open(OUT) as f:
    data = json.load(f)
cases = data.setdefault('solver_cases', [])
cases = [c for c in cases if c.get('fn') != 'quadratic_assignment']   # idempotent re-generation
cases.extend(qap_cases)
data['solver_cases'] = cases
with open(OUT, 'w') as f:
    json.dump(data, f, separators=(',', ':'))
print(f'optimize: +{len(qap_cases)} quadratic_assignment cases ({len(cases)} solver cases total)')
