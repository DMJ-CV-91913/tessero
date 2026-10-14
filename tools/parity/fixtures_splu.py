#!/usr/bin/env python3
"""scipy.sparse.linalg sparse-LU object layer fixtures: splu / factorized / SuperLU.

The LU factors L, U and the row/column permutations of a SuperLU object are ordering-specific (COLAMD +
partial pivoting) and non-unique, so they are NOT part of the parity surface. What IS unique -- and what
these fixtures verify -- is the solve A^-1 b: splu(A).solve(b), factorized(A)(b) and the SuperLU object's
solve all reproduce it to machine precision. The expected x is taken from scipy's SuperLU.solve; it equals
np.linalg.solve(A.toarray(), b) (the dense reference) to ~1e-15, confirming the value is ordering-independent.

Single and multiple right-hand sides are covered; shape (n, n) is checked for the object. Matrices are built
dense and passed dense, as elsewhere on the sparse parity surface. spilu is intentionally absent: its solve is
an approximate ILUTP factorisation whose value is SuperLU-internal and not reproducible (reported BLOCKED).
"""
import json, os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import fixtures_np as F

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'splu.json')
rng = np.random.default_rng(20261014)
TOL = {'atol': 1e-9, 'rtol': 1e-9}


def spd(n):
    A = rng.normal(size=(n, n))
    return A @ A.T + n * np.eye(n)


def general(n, density):
    A = rng.normal(size=(n, n))
    A[rng.uniform(size=(n, n)) > density] = 0.0
    A += np.diag(np.abs(rng.normal(size=n)) + float(n))      # diagonally dominant -> nonsingular
    return A


systems = [
    spd(4), spd(8),
    general(6, 0.5), general(10, 0.3), general(12, 0.2), general(20, 0.15),
]

splu_c = []
for A in systems:
    n = A.shape[0]
    As = sp.csc_array(A)
    lu = sl.splu(As)
    assert lu.shape == (n, n)

    b = rng.normal(size=n)
    xb = lu.solve(b)
    # cross-check against the dense reference so the fixture's "expect" is provably the unique solve
    assert np.allclose(xb, np.linalg.solve(A, b), rtol=1e-10, atol=1e-12)
    splu_c.append({'A': F.enc(A), 'b': F.enc(b), 'shape': [n, n],
                   'expect': F.enc_result(np.asarray(xb, float), []),
                   'tol': TOL})

    B = rng.normal(size=(n, 3))                               # multi-RHS
    XB = lu.solve(B)
    assert np.allclose(XB, np.linalg.solve(A, B), rtol=1e-10, atol=1e-12)
    splu_c.append({'A': F.enc(A), 'b': F.enc(B), 'shape': [n, n],
                   'expect': F.enc_result(np.asarray(XB, float), []),
                   'tol': TOL})

out = {'splu_cases': splu_c}
with open(OUT, 'w') as fh:
    json.dump(out, fh)
# strict validation: reject NaN / Infinity (PHP json_decode rejects them)
json.loads(json.dumps(out), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f'bad constant {c}')))
print(f'wrote {OUT}: {len(splu_c)} splu cases over {len(systems)} systems')
