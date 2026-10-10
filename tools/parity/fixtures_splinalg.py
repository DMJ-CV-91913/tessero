#!/usr/bin/env python3
"""scipy.sparse.linalg fixtures: direct solves (spsolve, spsolve_triangular), matrix norms and structure
queries (spbandwidth, is_sptriangular). Matrices are built dense and passed dense, as elsewhere on the sparse
parity surface; the solves use LAPACK so this generator runs in the pinned environment."""
import json, os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import fixtures_np as F

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'splinalg.json')
rng = np.random.default_rng(20261010)
TOL = {'atol': 1e-9, 'rtol': 1e-9}


def spd(n):
    A = rng.normal(size=(n, n)); return A @ A.T + n * np.eye(n)       # well-conditioned


def tri(n, lower=True):
    A = rng.normal(size=(n, n)); A = np.tril(A) if lower else np.triu(A)
    A += np.sign(np.diag(A)) * (np.abs(np.diag(A)) + 1.0) * np.eye(n)  # non-zero diagonal
    return A


solve_c, trisolve_c, norm_c, band_c, trit_c = [], [], [], [], []

for n in (4, 6, 8):
    A = spd(n); As = sp.csr_array(A)
    b = rng.normal(size=n)
    x = sl.spsolve(As, b)
    solve_c.append({'args': [F.enc(A), F.enc(b)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(x, float), []), 'compare': 'tol', 'tol': TOL})
    B = rng.normal(size=(n, 2))
    X = np.linalg.solve(A, B)                                          # multi-RHS reference (dense, exact)
    solve_c.append({'args': [F.enc(A), F.enc(B)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(X, float), []), 'compare': 'tol', 'tol': TOL})

for n in (4, 6, 7):
    for lower in (True, False):
        A = tri(n, lower); As = sp.csr_array(A)
        b = rng.normal(size=n)
        x = sl.spsolve_triangular(As, b, lower=lower)
        trisolve_c.append({'args': [F.enc(A), F.enc(b), F.enc(bool(lower))], 'kwargs': {}, 'expect': F.enc_result(np.asarray(x, float), []), 'compare': 'tol', 'tol': TOL})

for n in (5, 6):
    A = rng.normal(size=(n, n)); A[np.abs(A) < 0.6] = 0.0
    As = sp.csr_array(A)
    for ordv, enc_ord in ((None, None), ('fro', F.enc('fro')), (1, F.enc(1.0)), (-1, F.enc(-1.0)),
                          (np.inf, F.enc(float('inf'))), (-np.inf, F.enc(float('-inf')))):
        args = [F.enc(A)] + ([enc_ord] if enc_ord is not None else [])
        val = sl.norm(As, ordv) if ordv is not None else sl.norm(As)
        norm_c.append({'args': args, 'kwargs': {}, 'expect': F.enc_result(float(val), []), 'compare': 'tol', 'tol': TOL})

def band_mat(n, kl, ku):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if -kl <= j - i <= ku and rng.uniform() < 0.7:
                A[i, j] = rng.normal()
    return A

for (n, kl, ku) in ((6, 1, 2), (7, 3, 0), (5, 0, 0), (8, 2, 2)):
    A = band_mat(n, kl, ku); As = sp.csr_array(A)
    lo, up = sl.spbandwidth(As)
    band_c.append({'args': [F.enc(A)], 'kwargs': {}, 'expect': {'dict': {'lower': int(lo), 'upper': int(up)}}, 'compare': 'tol'})
    isl, isu = sl.is_sptriangular(As)
    trit_c.append({'args': [F.enc(A)], 'kwargs': {}, 'expect': {'dict': {'lower': bool(isl), 'upper': bool(isu)}}, 'compare': 'tol'})

inv_c, pow_c, expm_c = [], [], []
for n in (4, 5, 6):
    A = spd(n); As = sp.csr_array(A)
    inv_c.append({'args': [F.enc(A)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(sl.inv(As).toarray(), float), []), 'compare': 'tol', 'tol': TOL})
for n in (4, 5):
    A = rng.normal(size=(n, n)); A[np.abs(A) < 0.5] = 0.0; A += np.eye(n)
    As = sp.csr_array(A)
    for p in (0, 1, 2, 3):
        Ap = sl.matrix_power(As, p)
        Ap = Ap.toarray() if hasattr(Ap, 'toarray') else np.asarray(Ap)
        pow_c.append({'args': [F.enc(A), F.enc(int(p))], 'kwargs': {}, 'expect': F.enc_result(np.asarray(Ap, float), []), 'compare': 'tol', 'tol': TOL})
for n in (3, 4, 5):
    A = rng.normal(size=(n, n)) * 0.5
    expm_c.append({'args': [F.enc(A)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(sl.expm(sp.csr_array(A)).toarray(), float), []), 'compare': 'tol', 'tol': {'atol': 1e-8, 'rtol': 1e-8}})

em_c = []
for n in (3, 4, 5):
    A = rng.normal(size=(n, n)) * 0.5; As = sp.csr_array(A)
    b = rng.normal(size=n)
    em_c.append({'args': [F.enc(A), F.enc(b)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(sl.expm_multiply(As, b), float), []), 'compare': 'tol', 'tol': {'atol': 1e-8, 'rtol': 1e-8}})
    B = rng.normal(size=(n, 2))
    em_c.append({'args': [F.enc(A), F.enc(B)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(sl.expm_multiply(As, B), float), []), 'compare': 'tol', 'tol': {'atol': 1e-8, 'rtol': 1e-8}})

out = {'module': 'splinalg', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(), 'calls': [
    {'fn': 'inv', 'cases': inv_c}, {'fn': 'matrix_power', 'cases': pow_c}, {'fn': 'expm', 'cases': expm_c},
    {'fn': 'expm_multiply', 'cases': em_c},
    {'fn': 'spsolve', 'cases': solve_c}, {'fn': 'spsolve_triangular', 'cases': trisolve_c},
    {'fn': 'norm', 'cases': norm_c}, {'fn': 'spbandwidth', 'cases': band_c},
    {'fn': 'is_sptriangular', 'cases': trit_c},
]}
with open(OUT, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'))
print('splinalg: spsolve %d, spsolve_triangular %d, norm %d, spbandwidth %d, is_sptriangular %d'
      % (len(solve_c), len(trisolve_c), len(norm_c), len(band_c), len(trit_c)))
