#!/usr/bin/env python3
"""scipy.integrate.quad_vec parity fixtures.

The integrand is a NAMED function the PHP runner rebuilds (makeIntegrand): scalar kinds (poly / gauss /
sinx / lorentz) and a vector kind (xpow: x**alpha over an embedded alpha grid; osc: [sin(k x)] for k in
1..K). Only the integral result `res` is compared -- bit-exactly, per the de-risked prototype -- so the
fixtures are host-independent. Covers scalar + vector integrands under both gk15 (default) and gk21.
"""
import json, os
import numpy as np
from scipy.integrate import quad_vec

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "integrate_quadvec.json")


def polyf(c):
    return lambda x: float(sum(ck * x ** k for k, ck in enumerate(c)))


def make_scalar(kind, spec):
    if kind == "poly":
        return polyf(spec["c"])
    if kind == "gauss":
        return lambda x: float(np.exp(-x * x))
    if kind == "sinx":
        return lambda x: float(np.sin(x))
    if kind == "lorentz":
        return lambda x: float(1.0 / (1.0 + x * x))
    raise ValueError(kind)


def make_vector(kind, spec):
    if kind == "xpow":
        alpha = np.asarray(spec["alpha"], dtype=float)
        return lambda x: x ** alpha
    if kind == "osc":
        K = int(spec["k"])
        return lambda x: np.array([np.sin(k * x) for k in range(1, K + 1)])
    raise ValueError(kind)


def case(obj, a, b, norm, quadrature):
    kind = obj["kind"]
    if kind in ("xpow", "osc"):
        f = make_vector(kind, obj)
        vector = True
    else:
        f = make_scalar(kind, obj)
        vector = False
    res, err = quad_vec(f, a, b, norm=norm, quadrature=quadrature)
    res = np.asarray(res, dtype=float)
    expect = res.ravel().tolist() if vector else [float(res)]
    return {
        "obj": obj,
        "a": float(a),
        "b": float(b),
        "norm": norm,
        # emit the rule we PASSED; null means "scipy default" which the runner resolves to gk15
        "quadrature": quadrature,
        "vector": vector,
        "expect": expect,   # flat list<float>; scalar => 1-element
    }


alpha = np.linspace(0.0, 2.0, 30).tolist()

cases = [
    # scalar, default quadrature (None -> gk15)
    case({"kind": "sinx"}, 0.0, np.pi, "2", None),
    case({"kind": "gauss"}, -3.0, 3.0, "2", None),
    case({"kind": "poly", "c": [1.0, 2.0, 3.0]}, 0.0, 2.0, "2", None),
    case({"kind": "lorentz"}, -5.0, 5.0, "max", None),
    # scalar, explicit gk21
    case({"kind": "sinx"}, 0.0, np.pi, "2", "gk21"),
    case({"kind": "gauss"}, -3.0, 3.0, "2", "gk21"),
    case({"kind": "poly", "c": [0.0, 0.0, 0.0, 0.0, 1.0]}, -1.0, 2.0, "2", "gk21"),
    # scalar, explicit gk15
    case({"kind": "lorentz"}, -5.0, 5.0, "2", "gk15"),
    # vector x**alpha, gk15 default + gk21, 2-norm and max-norm
    case({"kind": "xpow", "alpha": alpha}, 0.0, 2.0, "2", None),
    case({"kind": "xpow", "alpha": alpha}, 0.0, 2.0, "max", None),
    case({"kind": "xpow", "alpha": alpha}, 0.0, 2.0, "2", "gk21"),
    case({"kind": "xpow", "alpha": alpha}, 0.0, 2.0, "max", "gk21"),
    # vector oscillatory [sin(k x)], gk15 + gk21
    case({"kind": "osc", "k": 5}, 0.0, 2.0 * np.pi, "2", None),
    case({"kind": "osc", "k": 5}, 0.0, 2.0 * np.pi, "2", "gk21"),
]

out = {"module": "integrate", "scipy": __import__("scipy").__version__, "quadvec_cases": cases}
with open(OUT, "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(f"integrate_quadvec: {len(cases)} quad_vec cases -> {OUT}")
