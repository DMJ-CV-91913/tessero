#!/usr/bin/env python3
r"""scipy.optimize.newton_krylov parity fixtures.

Jacobian-free (matrix-free) inexact-Newton with a restarted-GMRES inner solve.
The PHP port (Tessero\Optimize\NonlinSolve::newtonKrylov, with the finite-difference
Tessero\Optimize\KrylovJacobian) reproduces scipy.optimize._nonlin.nonlin_solve
driving scipy's KrylovJacobian (inner solver gmres, inner_maxiter=20). The inexact-
Newton forcing schedule is reproduced exactly, but the inner GMRES is our own port
(scipy's uses scipy.sparse.linalg.gmres), so the iterate PATH is numerically -- not
bit-for-bit -- identical. The root of each covered system is unique, so the comparison
is on the converged x only. Tolerance rtol 1e-6: scipy's default newton_krylov
f_tol = eps**(1/3) leaves ~1e-6..1e-7 slack in x, which is the robust both-backend
tolerance (the de-risk VERIFY_part1.py confirmed <=5e-9 rel vs scipy on every case; the
looser 1e-6 absorbs host/libm variation). Backend-neutral pure PHP (no BLAS),
host-independent.

Each residual F(x) -> list[float] is a NAMED 'vecsys_*' kind rebuilt in PHP by
makeObjective() (no callable serialised). Merged into optimize.json under the
existing 'solver_cases' key (consumed by run-tests.php via runSolverCases).

ORDER: run AFTER fixtures_optimize.py (which (re)writes optimize.json); the merge
pattern is idempotent (drops stale newton_krylov cases before re-appending)."""
import json
import math
import os

import numpy as np
from scipy.optimize import newton_krylov

EPS = 2.220446049250313e-16
FTOL = EPS ** (1.0 / 3.0)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "optimize.json")


# Named residual systems -- each mirrors a makeObjective() 'vecsys_*' kind in run-tests.php.
def nk_example(x):
    return [x[0] + 0.5 * x[1] - 1.0, 0.5 * (x[1] - x[0]) ** 2]


def quad2(x):
    return [x[0] ** 2 + x[1] ** 2 - 4.0, x[0] - x[1]]


def trig2(x):
    return [x[0] - 0.5 * math.cos(x[1]) - 0.5, x[1] - 0.5 * math.sin(x[0]) - 0.3]


def lin3(x):
    return [3 * x[0] + x[1] - x[2] - 2.0, x[0] + 4 * x[1] + x[2] - 6.0, -x[0] + x[1] + 5 * x[2] - 5.0]


def exp2(x):
    return [math.exp(x[0]) + x[1] - 2.0, x[0] + math.exp(x[1]) - 2.0]


def mix2(x):
    return [x[0] + math.sin(x[1]) - 1.0, x[1] + math.cos(x[0]) - 0.5]


# kind -> (residual, x0). The kind names must exist in makeObjective() (run-tests.php).
SYSTEMS = {
    "vecsys_nkexample": (nk_example, [0.0, 0.0]),
    "vecsys_quad2":     (quad2,      [1.0, 1.0]),
    "vecsys_trig2":     (trig2,      [0.0, 0.0]),
    "vecsys_lin3":      (lin3,       [0.0, 0.0, 0.0]),
    "vecsys_exp2":      (exp2,       [0.0, 0.0]),
    "vecsys_mix2":      (mix2,       [0.0, 0.0]),
}

nk_cases = []
for kind, (F, x0) in SYSTEMS.items():
    root = list(map(float, newton_krylov(lambda z: np.asarray(F(list(z))),
                                          np.array(x0, float), f_tol=FTOL)))
    nk_cases.append({
        "fn": "newton_krylov",
        "obj": {"kind": kind},
        "x0": x0,
        "expect": root,
        "rtol": 1e-6,
    })

if os.path.exists(OUT):
    with open(OUT) as f:
        data = json.load(f)
    data.setdefault("solver_cases", [])
    data["solver_cases"] = [c for c in data["solver_cases"] if c.get("fn") != "newton_krylov"]
    data["solver_cases"].extend(nk_cases)
else:
    data = {"module": "optimize", "scipy": __import__("scipy").__version__, "solver_cases": nk_cases}

with open(OUT, "w") as f:
    json.dump(data, f, separators=(",", ":"), allow_nan=False)
print(f"optimize: +{len(nk_cases)} newton_krylov cases")
