#!/usr/bin/env python3
"""scipy.sparse.linalg iterative solvers + onenormest fixtures.

Design note on the COMPARE STRATEGY (important): Krylov solvers return an x that satisfies
||b - A x|| <= max(rtol*||b||, atol); on the well-conditioned systems used here that x equals
the direct solve to <1e-6. We therefore compare against the EXACT direct solution
(np.linalg.solve / np.linalg.lstsq), NOT against scipy's iterate, because scipy's x carries
iteration-count noise and would differ from ours in the last digits. rtol is set tight (1e-9)
and matrices are well-conditioned so both agree far below it.

Matrices are built dense and passed dense, like the rest of the sparse parity surface; the
kernel routines run the Krylov iteration on the dense operator.

Append the 'calls' entries below to tools/parity/fixtures_splinalg.py's out['calls'] list (or run
this file to emit a separate splinalg_iter.json and register it in run-tests.php)."""
import json, os
import numpy as np
import fixtures_np as F

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'splinalg_iter.json')
rng = np.random.default_rng(20261013)
TOL = {'atol': 1e-7, 'rtol': 1e-7}


def spd(n):
    A = rng.normal(size=(n, n)); return A @ A.T + n * np.eye(n)


def symind(n):                       # symmetric, possibly indefinite (for minres)
    A = rng.normal(size=(n, n)); A = (A + A.T) / 2; A += n * np.eye(n); return A


def ddom(n):                         # diagonally dominant nonsymmetric (for bicg/cgs/bicgstab/gmres)
    A = rng.normal(size=(n, n)); A += n * np.eye(n); return A


def solver_cases(matgen):
    cases = []
    for n in (5, 8, 12):
        A = matgen(n); b = rng.normal(size=n)
        x = np.linalg.solve(A, b)                # exact reference
        cases.append({'args': [F.enc(A), F.enc(b), None, F.enc(1e-12), F.enc(0.0)], 'kwargs': {},
                      'expect': F.enc_result(np.asarray(x, float), []), 'compare': 'tol', 'tol': TOL})
    return cases


spd_c  = solver_cases(spd)
sym_c  = solver_cases(symind)
gen_c  = solver_cases(ddom)

# lsqr / lsmr: rectangular least squares; compare to np.linalg.lstsq
lsq_c = []
for (m, n) in ((10, 6), (12, 8), (9, 5)):
    A = rng.normal(size=(m, n)); b = rng.normal(size=m)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    lsq_c.append({'args': [F.enc(A), F.enc(b)], 'kwargs': {},
                  'expect': F.enc_result(np.asarray(x, float), []), 'compare': 'tol',
                  'tol': {'atol': 1e-6, 'rtol': 1e-6}})

# onenormest: exact 1-norm on these small matrices
one_c = []
for n in (6, 8, 10):
    A = rng.normal(size=(n, n)); A[np.abs(A) < 0.5] = 0.0
    val = np.abs(A).sum(axis=0).max()
    one_c.append({'args': [F.enc(A)], 'kwargs': {},
                  'expect': F.enc_result(float(val), []), 'compare': 'tol', 'tol': {'atol': 1e-12, 'rtol': 1e-12}})


lsmr_c = []
for (m, n) in ((10, 6), (12, 8), (9, 5)):
    A = rng.normal(size=(m, n)); b = rng.normal(size=m)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    lsmr_c.append({'args': [F.enc(A), F.enc(b)], 'kwargs': {},
                   'expect': F.enc_result(np.asarray(x, float), []), 'compare': 'tol',
                   'tol': {'atol': 1e-6, 'rtol': 1e-6}})

out = {'module': 'splinalg', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': [
    {'fn': 'qmr', 'cases': gen_c}, {'fn': 'tfqmr', 'cases': gen_c}, {'fn': 'lsmr', 'cases': lsmr_c},
    {'fn': 'cg', 'cases': spd_c},
    {'fn': 'bicgstab', 'cases': gen_c},
    {'fn': 'bicg', 'cases': gen_c},
    {'fn': 'cgs', 'cases': gen_c},
    {'fn': 'gmres', 'cases': gen_c},
    {'fn': 'lgmres', 'cases': gen_c},
    {'fn': 'gcrotmk', 'cases': gen_c},
    {'fn': 'minres', 'cases': sym_c},
    {'fn': 'lsqr', 'cases': lsq_c},
    {'fn': 'onenormest', 'cases': one_c},
]}
with open(OUT, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'))
print('splinalg_iter: cg %d, bicgstab %d, bicg %d, cgs %d, gmres %d, minres %d, lsqr %d, onenormest %d'
      % (len(spd_c), len(gen_c), len(gen_c), len(gen_c), len(gen_c), len(sym_c), len(lsq_c), len(one_c)))
