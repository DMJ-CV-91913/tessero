#!/usr/bin/env python3
"""scipy.sparse.linalg.lobpcg + funm_multiply_krylov fixtures.

COMPARE STRATEGY (important, matches the Krylov family in fixtures_splinalg_iter.py):

  - lobpcg returns k eigenpairs of a symmetric-PD matrix via the LOBPCG iteration.
    EigenVECTORS are NOT unique (sign/rotation within a degenerate block), so the
    kernel ROUTINE returns the EIGENVALUES only, and we compare them (ascending) to
    the EXACT dense reference np.linalg.eigvalsh(A) sliced to the k smallest/largest.
    This is the eigsh-style value-only check; on these well-separated SPD matrices
    LOBPCG reaches the exact eigenvalues far below the tolerance.

  - funm_multiply_krylov(f, A, b) ~= f(A) @ b via Arnoldi. With the full Krylov
    dimension m = n (dense, small) it equals the dense f(A) @ b to round-off, so we
    compare to the EXACT dense reference scipy.linalg.<f>(A) @ b. f is passed by NAME
    (a string) because the kernel dispatches the matrix function internally through
    the shared Schur-Parlett engine.

Matrices are built dense and passed dense, like the rest of the sparse surface."""
import json, os
import numpy as np
import scipy.linalg as la
import fixtures_np as F

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'splinalg_lobpcg.json')
rng = np.random.default_rng(20261013)


def spd(n):
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


# ---- lobpcg: eigenvalues only, vs the exact dense eigvalsh slice --------------
lob_c = []
for n, k, largest in [(6, 2, False), (8, 3, False), (10, 2, True),
                      (12, 3, True), (7, 2, False), (9, 4, True)]:
    A = spd(n)
    X0 = rng.normal(size=(n, k))
    we = np.sort(np.linalg.eigvalsh(A))
    ref = np.sort(we[-k:] if largest else we[:k])
    # facade signature: lobpcg(A, X, B, M, Y, tol, maxiter, largest)
    lob_c.append({'args': [F.enc(A), F.enc(X0), None, None, None, None, None, bool(largest)],
                  'kwargs': {},
                  'expect': F.enc_result(np.asarray(ref, float), []),
                  'compare': 'tol', 'tol': {'atol': 1e-8, 'rtol': 1e-8}})

# ---- funm_multiply_krylov: result vector, vs the dense f(A) @ b ---------------
FUNCS = {'exp': la.expm, 'sqrt': la.sqrtm, 'log': la.logm, 'cos': la.cosm, 'sin': la.sinm}
funm_c = []
for fname, func in FUNCS.items():
    for n in (5, 8, 10):
        if fname in ('sqrt', 'log'):
            A = spd(n)                      # SPD -> real principal sqrt/log
        else:
            A = rng.normal(size=(n, n)); A = (A + A.T) / 2
        b = rng.normal(size=n)
        ref = np.real(np.asarray(func(A)) @ b)
        # facade signature: funmMultiplyKrylov(f, A, b); f passed by name
        funm_c.append({'args': [F.enc(fname), F.enc(A), F.enc(b)], 'kwargs': {},
                       'expect': F.enc_result(np.asarray(ref, float), []),
                       'compare': 'tol', 'tol': {'atol': 1e-9, 'rtol': 1e-9}})

out = {'module': 'splinalg', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(),
       'calls': [
           {'fn': 'lobpcg', 'cases': lob_c},
           {'fn': 'funm_multiply_krylov', 'cases': funm_c},
       ]}
with open(OUT, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'))
print('splinalg_lobpcg: lobpcg %d, funm_multiply_krylov %d' % (len(lob_c), len(funm_c)))
