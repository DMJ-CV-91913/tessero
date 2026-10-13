#!/usr/bin/env python3
r"""scipy.optimize.line_search + scipy.optimize.approx_fprime parity fixtures (pure-PHP Tessero\Optimize\LineSearch).

line_search is scipy.optimize.line_search == line_search_wolfe2 / scalar_search_wolfe2 (strong Wolfe, the
deterministic MT-free interpolation search). The returned step alpha, fc, gc, new_fval and the new-slope
gradient array are bit-reproducible, so the runner compares alpha/new_fval (rtol 1e-9) and fc/gc exactly.
approx_fprime is the forward 2-point finite-difference gradient (default step sqrt(eps)); rtol 1e-9.

Objectives + analytic gradients are NAMED kinds rebuilt in PHP by makeObjective / makeGradient, so the same
f and grad cross into PHP (a Python callable cannot). Appended to optimize.json solver_cases.

Backend-neutral pure PHP (no BLAS); host-independent, no pinned environment needed."""
import json, os
import numpy as np
import scipy.optimize as so

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "optimize.json")


def f_of(kind):
    if kind == "vquad":
        return lambda x: float(sum(float(v) * float(v) for v in x))
    if kind == "wquad":
        return lambda x: x[0] ** 2 + 3.0 * x[1] ** 2
    if kind == "rosen":
        return lambda x: float(so.rosen(x))
    if kind == "shift3":
        return lambda x: (x[0] - 1.0) ** 2 + (x[1] - 2.5) ** 2 + (x[2] + 0.3) ** 2
    if kind == "expsum":
        return lambda x: float(sum(float(np.exp(0.5 * float(v))) + float(v) ** 2 for v in x))
    if kind == "trig":
        return lambda x: np.sin(x[0]) * np.cos(x[1]) + 0.5 * (x[0] ** 2 + x[1] ** 2)
    raise ValueError(kind)


def g_of(kind):
    if kind == "vquad":
        return lambda x: 2.0 * np.asarray(x, float)
    if kind == "wquad":
        return lambda x: np.array([2.0 * x[0], 6.0 * x[1]])
    if kind == "rosen":
        return lambda x: so.rosen_der(x)
    if kind == "shift3":
        return lambda x: np.array([2 * (x[0] - 1.0), 2 * (x[1] - 2.5), 2 * (x[2] + 0.3)])
    if kind == "expsum":
        return lambda x: 0.5 * np.exp(0.5 * np.asarray(x, float)) + 2.0 * np.asarray(x, float)
    if kind == "trig":
        return lambda x: np.array([np.cos(x[0]) * np.cos(x[1]) + x[0], -np.sin(x[0]) * np.sin(x[1]) + x[1]])
    raise ValueError(kind)


# approx_fprime: (kind, point). Default epsilon, rtol 1e-9.
FP = [
    ("vquad", [1.0, 2.0, 3.0]),
    ("wquad", [0.5, -1.5]),
    ("rosen", [1.3, 0.7, 0.8, 1.9, 1.2]),
    ("shift3", [0.0, 0.0, 0.0]),
    ("expsum", [-0.4, 0.9]),
    ("trig", [0.3, -0.7]),
    ("vquad", [0.0, 5.0]),
]

# line_search: (kind, xk, pk). alpha/fc/gc/new_fval/new_slope compared.
LS = [
    ("vquad", [1.5, 2.0], [-3.0, -4.0]),
    ("wquad", [1.0, 1.0], [-2.0, -6.0]),
    ("rosen", [-1.2, 1.0], [1.0, 0.0]),
    ("rosen", [0.0, 0.0], [1.0, 1.0]),
    ("shift3", [0.0, 0.0, 0.0], [1.0, 2.5, -0.3]),
    ("expsum", [1.0, 1.0], [-1.0, -1.0]),
    ("trig", [1.0, 1.0], [-0.8, -0.6]),
    ("vquad", [10.0, 0.0], [-1.0, 0.0]),
]

cases = []
for kind, x in FP:
    g = so.approx_fprime(np.asarray(x, float), f_of(kind))
    cases.append({"fn": "approx_fprime", "obj": {"kind": kind}, "x": [float(v) for v in x],
                  "expect": [float(v) for v in g], "rtol": 1e-9})

for kind, xk, pk in LS:
    r = so.line_search(f_of(kind), g_of(kind), np.asarray(xk, float), np.asarray(pk, float))
    alpha, fc, gc, newf, oldf, slope = r
    cases.append({"fn": "line_search", "obj": {"kind": kind}, "xk": [float(v) for v in xk],
                  "pk": [float(v) for v in pk],
                  "alpha": (None if alpha is None else float(alpha)),
                  "fc": int(fc), "gc": int(gc),
                  "new_fval": (None if newf is None else float(newf)),
                  "new_slope": (None if slope is None else [float(v) for v in np.atleast_1d(slope)]),
                  "rtol": 1e-9})

with open(OUT) as fh:
    data = json.load(fh)
data.setdefault("solver_cases", [])
data["solver_cases"] = [c for c in data["solver_cases"] if c.get("fn") not in ("approx_fprime", "line_search")]
data["solver_cases"].extend(cases)
with open(OUT, "w") as fh:
    json.dump(data, fh, indent=2)
print("wrote", len(cases), "line_search/approx_fprime cases to", OUT)
