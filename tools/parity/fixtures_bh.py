#!/usr/bin/env python3
import json, os
import numpy as np
import scipy.optimize as so

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "optimize.json")

def sphere(x):
    x = np.asarray(x, float); return float(np.sum(x * x))

def shifted_quadratic(x, shift):
    x = np.asarray(x, float); s = np.asarray(shift, float); return float(np.sum((x - s) ** 2))

def booth(x):
    return float((x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2)

def matyas(x):
    return float(0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1])

def rosen(x):
    return float(so.rosen(x))

SPECS = [
    {"kind": "sphere", "x0": [2.0, -1.5], "seed": 42, "niter": 8},
    {"kind": "shifted_quadratic", "x0": [0.0, 0.0], "shift": [1.5, -2.0], "seed": 7, "niter": 8},
    {"kind": "booth", "x0": [0.0, 0.0], "seed": 123, "niter": 10},
    {"kind": "matyas", "x0": [3.0, 4.0], "seed": 2024, "niter": 10},
    {"kind": "shifted_quadratic", "x0": [0.0, 0.0, 0.0], "shift": [1.0, -2.0, 0.5], "seed": 99, "niter": 8},
    {"kind": "rosen", "x0": [0.0, 0.0], "seed": 5, "niter": 12},
]

bh_cases = []
for spec in SPECS:
    kind = spec["kind"]
    if kind == "sphere":
        func = sphere
        obj = {"kind": "sphere"}
    elif kind == "shifted_quadratic":
        shift = spec["shift"]
        func = lambda x, _s=shift: shifted_quadratic(x, _s)
        obj = {"kind": "shifted_quadratic", "shift": shift}
    elif kind == "booth":
        func = booth
        obj = {"kind": "booth"}
    elif kind == "matyas":
        func = matyas
        obj = {"kind": "matyas"}
    elif kind == "rosen":
        func = rosen
        obj = {"kind": "rosen"}
    else:
        raise SystemExit("unknown kind " + kind)

    res = so.basinhopping(
        func, spec["x0"], niter=spec["niter"], T=1.0, stepsize=0.5,
        minimizer_kwargs={"method": "L-BFGS-B"},
        rng=np.random.default_rng(spec["seed"]),
    )
    bh_cases.append({
        "fn": "basinhopping",
        "obj": obj,
        "x0": [float(v) for v in spec["x0"]],
        "seed": spec["seed"],
        "niter": spec["niter"],
        "expect": list(map(float, res.x)),
        "expect_fun": float(res.fun),
        "rtol": 1e-4,
    })

if os.path.exists(OUT):
    with open(OUT) as f:
        data = json.load(f)
    data.setdefault("solver_cases", [])
    data["solver_cases"] = [c for c in data["solver_cases"] if c.get("fn") != "basinhopping"]
    data["solver_cases"].extend(bh_cases)
else:
    data = {"module": "optimize", "scipy": __import__("scipy").__version__, "solver_cases": bh_cases}

with open(OUT, "w") as f:
    json.dump(data, f, separators=(",", ":"))
print("optimize: +%d basinhopping cases" % len(bh_cases))
