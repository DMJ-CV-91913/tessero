#!/usr/bin/env python3
r"""scipy.integrate.cubature parity fixtures. The integrand is a NAMED kind the PHP runner rebuilds; only
the scalar estimate is compared (rtol 1e-8, matching cubature default). Covers both the default product
Gauss-Kronrod rule ("gk21") and the Genz-Malik rule over 2-D and 3-D boxes. Tessero\Integrate\Cubature
is pure-PHP (heap-adaptive subdivision over hard-coded QUADPACK GK nodes + roots_legendre lower rule, and
the closed-form Genz-Malik rule); host-independent (no kernel, no LAPACK)."""
import json, os
import numpy as np
from scipy.integrate import cubature

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "integrate_cubature.json")

def mk(kind):
    return {
        "sumsq":  lambda x: np.sum(x**2, axis=-1),
        "prod":   lambda x: np.prod(x, axis=-1),
        "coscos": lambda x: np.sum(np.cos(x), axis=-1),
        "expsum": lambda x: np.exp(np.sum(x, axis=-1)),
        "gauss":  lambda x: np.exp(-np.sum(x**2, axis=-1)),
        "poly":   lambda x: np.sum(x**3, axis=-1) + np.prod(x, axis=-1),
        "invr":   lambda x: 1.0 / (1.0 + np.sum(x**2, axis=-1)),
    }[kind]

SPECS = [
    ("sumsq",  [0, 0],    [1, 1],    "gk21"),
    ("coscos", [0, 0],    [1, 1],    "gk21"),
    ("expsum", [0, 0, 0], [1, 1, 1], "gk21"),
    ("gauss",  [0, 0],    [1, 2],    "gk21"),
    ("prod",   [0, 0],    [2, 3],    "gk21"),
    ("invr",   [0, 0],    [1, 1],    "gk21"),
    ("sumsq",  [0, 0],    [1, 1],    "genz-malik"),
    ("coscos", [0, 0, 0], [1, 1, 1], "genz-malik"),
    ("gauss",  [0, 0],    [1, 2],    "genz-malik"),
    ("poly",   [0, 0],    [1, 1],    "genz-malik"),
    ("expsum", [0, 0, 0], [1, 1, 1], "genz-malik"),
    ("invr",   [0, 0, 0], [1, 1, 1], "genz-malik"),
]

cases = []
for kind, a, b, rule in SPECS:
    r = cubature(mk(kind), a, b, rule=rule)
    cases.append({
        "fn": "cubature", "kind": kind,
        "a": [float(v) for v in a], "b": [float(v) for v in b],
        "rule": rule, "expect": float(np.ravel(r.estimate)[0]), "rtol": 1e-8,
    })

out = {"module": "integrate", "scipy": __import__("scipy").__version__, "cubature_cases": cases}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(f"integrate cubature: {len(cases)} cases -> {OUT}")
