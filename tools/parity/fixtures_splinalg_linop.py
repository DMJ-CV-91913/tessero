#!/usr/bin/env python3
"""Generate parity fixtures for scipy.sparse.linalg.LinearOperator / aslinearoperator / LaplacianNd.

All three are deterministic linear algebra -> fixtures are matched to SciPy at
machine precision (verified here before writing; a mismatch aborts).

The LinearOperator / aslinearoperator matrices are encoded by a NAMED KIND so the
PHP harness can rebuild the same operator (mirrors makeObjective/makeOdeSystem):
each case carries {"op", "kind", ...} and the expected output. The named matrices
are defined identically here and in run-tests.php's makeLinOp() builder.

Case schemas:
  op=linop   : {op, kind, source(matrix|callback), act, x|X, expect}
  op=aslinop : {op, kind, source(matrix|csr), act, x|X, expect}
  op=laplacian: {op, grid, bc, method(toarray|eigenvalues|matvec), x?, m?, expect}
act in {matvec, rmatvec, matmat, dot, H.matvec, T.matvec, sum, scale, compose, scaledsum}
"""
import json
import numpy as np
import scipy.sparse.linalg as sla
from scipy.sparse import csr_matrix

RTOL = 1e-12
EPS = 1e-12  # abort threshold vs scipy

# ---- named matrices (MUST match makeLinOp() in run-tests.php) ----
NAMED = {
    "m33a": [[1.0, 2.0, 0.0], [0.0, 3.0, 1.0], [4.0, 0.0, 2.0]],
    "m33b": [[2.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 2.0, 3.0]],
    "m23":  [[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]],
    "m32":  [[2.0, 1.0], [-1.0, 3.0], [0.0, 2.0]],
    "m44":  [[1.0, 0.0, -1.0, 2.0], [3.0, 2.0, 0.0, 1.0],
             [0.0, 1.0, 4.0, -2.0], [2.0, -1.0, 1.0, 3.0]],
    "s33":  [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]],
}

# fixed test vectors / matrices (match run-tests.php constants)
VEC = {
    2: [1.0, 2.0],
    3: [1.0, 2.0, 3.0],
    4: [1.0, 2.0, 3.0, 4.0],
}
MAT = {  # (n, k) row-major
    2: [[1.0, 0.0], [0.0, 1.0]],
    3: [[1.0, 0.5], [2.0, -1.0], [0.0, 3.0]],
    4: [[1.0, 2.0], [0.0, 1.0], [3.0, -1.0], [1.0, 1.0]],
}

fails = []
cases = []


def A_of(kind):
    return np.asarray(NAMED[kind], dtype=float)


def add_linop(kind, source, act):
    A = A_of(kind)
    m, n = A.shape
    LA = sla.aslinearoperator(A)
    c = {"op": "linop", "kind": kind, "source": source, "act": act}
    if act == "matvec":
        x = VEC[n]; exp = LA.matvec(x); c["x"] = x
    elif act == "dot":
        x = VEC[n]; exp = LA.dot(np.asarray(x)); c["x"] = x
    elif act == "rmatvec":
        x = VEC[m]; exp = LA.rmatvec(x); c["x"] = x
    elif act == "H.matvec":
        x = VEC[m]; exp = LA.H.matvec(x); c["x"] = x
    elif act == "T.matvec":
        x = VEC[m]; exp = LA.T.matvec(x); c["x"] = x
    elif act == "matmat":
        X = MAT[n]; exp = LA.matmat(np.asarray(X)); c["X"] = X
    else:
        raise ValueError(act)
    exp = np.asarray(exp, dtype=float)
    c["expect"] = exp.tolist()
    # verify against the intended PHP semantics (A@x etc) too
    ref = {
        "matvec": A @ np.asarray(VEC[n]),
        "dot":    A @ np.asarray(VEC[n]),
        "rmatvec": A.T @ np.asarray(VEC[m]),
        "H.matvec": A.T @ np.asarray(VEC[m]),
        "T.matvec": A.T @ np.asarray(VEC[m]),
        "matmat": A @ np.asarray(MAT[n]),
    }[act]
    if not np.allclose(exp, ref, rtol=EPS, atol=EPS):
        fails.append((kind, source, act, "ref mismatch"))
    cases.append(c)


def add_arith(kinda, kindb, act):
    A = A_of(kinda)
    B = A_of(kindb) if kindb is not None else None
    n = A.shape[1]
    LA = sla.aslinearoperator(A)
    LB = sla.aslinearoperator(B) if B is not None else None
    x = VEC[n]
    s = 2.5
    c = {"op": "linop", "kind": kinda, "source": "matrix", "act": act, "x": x, "scalar": s}
    if kindb is not None:
        c["kind2"] = kindb
    if act == "sum":
        exp = (LA + LB).matvec(x); ref = (A + B) @ np.asarray(x)
    elif act == "scale":
        exp = (s * LA).matvec(x); ref = (s * A) @ np.asarray(x)
    elif act == "compose":
        exp = (LA @ LB).matvec(x); ref = (A @ B) @ np.asarray(x)
    elif act == "scaledsum":
        exp = (s * LA + LB).matvec(x); ref = (s * A + B) @ np.asarray(x)
    else:
        raise ValueError(act)
    exp = np.asarray(exp, float)
    if not np.allclose(exp, ref, rtol=EPS, atol=EPS):
        fails.append((kinda, kindb, act, "arith ref mismatch"))
    c["expect"] = exp.tolist()
    cases.append(c)


# ----- LinearOperator from matrix: all acts -----
for kind in ["m33a", "m23", "m44", "s33"]:
    for act in ["matvec", "rmatvec", "matmat", "dot", "H.matvec", "T.matvec"]:
        add_linop(kind, "matrix", act)

# ----- LinearOperator from callbacks (square only; matvec/rmatvec/H) -----
for kind in ["m33a", "m44", "s33"]:
    for act in ["matvec", "rmatvec", "H.matvec"]:
        add_linop(kind, "callback", act)

# ----- aslinearoperator on ndarray + csr -----
for kind in ["m33a", "m23", "m32"]:
    add_linop(kind, "matrix", "matvec")
    add_linop(kind, "matrix", "rmatvec")
for kind in ["m33a", "m44"]:
    add_linop(kind, "csr", "matvec")
    add_linop(kind, "csr", "rmatvec")

# ----- arithmetic / composition -----
add_arith("m33a", "m33b", "sum")
add_arith("m33a", None, "scale")
add_arith("m33a", "m33b", "compose")
add_arith("m33a", "m33b", "scaledsum")
add_arith("s33", "m33b", "sum")
add_arith("m44", "m44", "compose")

# ----- LaplacianNd -----
def add_lap(grid, bc, method, m=None):
    L = sla.LaplacianNd(tuple(grid), boundary_conditions=bc)
    c = {"op": "laplacian", "grid": list(grid), "bc": bc, "method": method}
    if method == "toarray":
        exp = L.toarray().astype(float)
    elif method == "tosparse":
        exp = L.tosparse().toarray().astype(float)
    elif method == "eigenvalues":
        exp = np.sort(L.eigenvalues() if m is None else L.eigenvalues(m))
        if m is not None:
            c["m"] = int(m)
    elif method == "matvec":
        p = int(np.prod(grid))
        x = [float((i % 5) - 2) for i in range(p)]
        c["x"] = x
        exp = L.matvec(np.asarray(x))
    else:
        raise ValueError(method)
    exp = np.asarray(exp, float)
    # cross-check eigenvalues against dense eigvalsh
    if method == "eigenvalues" and m is None:
        dref = np.sort(np.linalg.eigvalsh(L.toarray().astype(float)))
        if not np.allclose(exp, dref, rtol=1e-10, atol=1e-10):
            fails.append((grid, bc, method, "eig vs dense"))
    c["expect"] = exp.tolist()
    cases.append(c)


for grid in [[4], [6], [2, 3], [3, 4], [2, 2, 2], [2, 3, 2]]:
    for bc in ["dirichlet", "neumann", "periodic"]:
        add_lap(grid, bc, "toarray")
        add_lap(grid, bc, "tosparse")
        add_lap(grid, bc, "eigenvalues")
        add_lap(grid, bc, "matvec")
# eigenvalues(m) subset
add_lap([6], "periodic", "eigenvalues", m=2)
add_lap([3, 4], "dirichlet", "eigenvalues", m=3)

if fails:
    raise SystemExit("ABORT: fixture self-check mismatches: %r" % (fails[:10],))

OUT = "../../tests/fixtures/parity/splinalg_linop.json"
out = {
    "module": "splinalg",
    "scipy": __import__("scipy").__version__,
    "splinalg_linop_cases": cases,
}
with open(OUT, "w") as fh:
    json.dump(out, fh, separators=(",", ":"), allow_nan=False)
print("splinalg_linop: %d cases (LinearOperator/aslinearoperator/LaplacianNd)" % len(cases))
