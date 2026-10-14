#!/usr/bin/env python3
r"""scipy.optimize.BroydenFirst / InverseJacobian / KrylovJacobian parity fixtures.

These three public classes are the Jacobian-approximation helpers underlying the
already-landed nonlinear solvers (Tessero\Optimize\NonlinSolve drives them for
broyden1/newton_krylov). They expose a small numeric surface:

  * BroydenFirst   -- setup(x0, F(x0), func); update(x,f)*; solve(v) = J^-1 . v
                      ("good" Broyden low-rank inverse-Jacobian, LowRankMatrix).
  * InverseJacobian(jac) -- a wrapper exposing the inverse operator:
                      matvec(v) == jac.solve(v), update/setup delegated.
  * KrylovJacobian -- setup(x0, F(x0), func); matvec(v) = finite-difference J . v
                      (the matrix-free directional derivative; GMRES inner solve
                      is exercised separately by newton_krylov).

Each case scripts a deterministic op sequence (setup, zero-or-more update steps
over a fixed march of points, then one probe: solve for Broyden/Inverse, matvec
for Krylov) and records scipy's result. NO RNG -> fully reproducible. The de-risk
(VERIFY via drive.php) matched scipy to max_rel 2.5e-15 on BOTH backends (pure
PHP, no BLAS -> host-independent), so the tolerance is a tight rtol 1e-10.

Residual systems reuse the existing makeObjective() 'vecsys_*' kinds (run-tests.php).
Merged into optimize.json under a NEW 'nkclass_cases' key (consumed by
run-tests.php via runNkClassCases). Run AFTER fixtures_optimize.py; idempotent."""
import json
import math
import os

import numpy as np
from scipy.optimize import BroydenFirst, InverseJacobian, KrylovJacobian

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "optimize.json")


# Named residual systems -- each mirrors a makeObjective() 'vecsys_*' kind in run-tests.php.
def rotten(x): return [x[0] + 0.5 * (x[0] - x[1]) ** 3 - 1.0, 0.5 * (x[1] - x[0]) ** 3 + x[1]]
def quad2(x):  return [x[0] ** 2 + x[1] ** 2 - 4.0, x[0] - x[1]]
def lin2(x):   return [2.0 * x[0] + x[1] - 5.0, x[0] + 3.0 * x[1] - 10.0]
def trig2(x):  return [x[0] - 0.5 * math.cos(x[1]) - 0.5, x[1] - 0.5 * math.sin(x[0]) - 0.3]
def lin3(x):   return [3 * x[0] + x[1] - x[2] - 2.0, x[0] + 4 * x[1] + x[2] - 6.0, -x[0] + x[1] + 5 * x[2] - 5.0]
def exp2(x):   return [math.exp(x[0]) + x[1] - 2.0, x[0] + math.exp(x[1]) - 2.0]

# sys-name -> (vecsys kind in makeObjective, residual, x0)
SYS = {
    "rotten": ("vecsys_rotten", rotten, [0.1, 0.2]),
    "quad2":  ("vecsys_quad2",  quad2,  [1.0, 1.0]),
    "lin2":   ("vecsys_lin2",   lin2,   [0.3, 0.4]),
    "trig2":  ("vecsys_trig2",  trig2,  [0.2, 0.1]),
    "lin3":   ("vecsys_lin3",   lin3,   [0.1, 0.2, 0.3]),
    "exp2":   ("vecsys_exp2",   exp2,   [0.1, -0.1]),
}


def npfunc(F):
    return lambda z: np.asarray(F(list(np.atleast_1d(z))), float)


def march(x0, nup):
    """Deterministic sequence of update points starting at x0."""
    pts = [list(x0)]
    n = len(x0)
    for k in range(nup):
        p = list(pts[-1])
        p[k % n] += 0.05 * (k + 1)
        pts.append(p)
    return pts


def probe_vec(n, off):
    return [math.sin(1.0 + 0.3 * i + off) + 0.5 for i in range(n)]


cases = []

# BroydenFirst: setup -> update* -> solve(v)
for name, (kind, F, x0) in SYS.items():
    for nup in (0, 2, 3):
        jac = BroydenFirst()
        pts = march(x0, nup)
        jac.setup(np.array(pts[0], float), npfunc(F)(pts[0]), npfunc(F))
        for p in pts[1:]:
            jac.update(np.array(p, float), npfunc(F)(p))
        v = probe_vec(len(x0), 0.0)
        got = list(map(float, jac.solve(np.array(v, float))))
        cases.append({"cls": "BroydenFirst", "obj": {"kind": kind},
                      "x0": [float(z) for z in x0],
                      "updates": [[float(z) for z in p] for p in pts[1:]],
                      "probe": v, "op": "solve", "expect": got, "rtol": 1e-10})

# InverseJacobian(BroydenFirst): matvec(v) == underlying solve
for name, (kind, F, x0) in SYS.items():
    for nup in (0, 2):
        ij = InverseJacobian(BroydenFirst())
        pts = march(x0, nup)
        ij.setup(np.array(pts[0], float), npfunc(F)(pts[0]), npfunc(F))
        for p in pts[1:]:
            ij.update(np.array(p, float), npfunc(F)(p))
        v = probe_vec(len(x0), 0.7)
        got = list(map(float, ij.matvec(np.array(v, float))))
        cases.append({"cls": "InverseJacobian", "obj": {"kind": kind},
                      "x0": [float(z) for z in x0],
                      "updates": [[float(z) for z in p] for p in pts[1:]],
                      "probe": v, "op": "matvec", "expect": got, "rtol": 1e-10})

# KrylovJacobian: matvec(v) = finite-difference J . v at the setup point
for name, (kind, F, x0) in SYS.items():
    jac = KrylovJacobian()
    jac.setup(np.array(x0, float), npfunc(F)(x0), npfunc(F))
    for s in (0.0, 0.9, 1.7):
        v = probe_vec(len(x0), s)
        got = list(map(float, jac.matvec(np.array(v, float))))
        cases.append({"cls": "KrylovJacobian", "obj": {"kind": kind},
                      "x0": [float(z) for z in x0], "updates": [],
                      "probe": v, "op": "matvec", "expect": got, "rtol": 1e-10})

if os.path.exists(OUT):
    with open(OUT) as f:
        data = json.load(f)
    data["nkclass_cases"] = cases
else:
    data = {"module": "optimize", "scipy": __import__("scipy").__version__, "nkclass_cases": cases}

with open(OUT, "w") as f:
    json.dump(data, f, separators=(",", ":"), allow_nan=False)
print(f"optimize: +{len(cases)} nkclass (BroydenFirst/InverseJacobian/KrylovJacobian) cases")
