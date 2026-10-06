"""
Reference values for Tessero\\Ext\\Math loop ufuncs, computed by NumPy and
SciPy (scipy.special for erf, erfc, gamma, gammaln).

    bash tools/parity/gen-fixtures.sh legacy   ->  tests/fixtures/ufunc.json (pinned environment, ADR 0012)
"""
import json
import os
import sys
import pathlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "parity"))
import fixture_env  # noqa: E402  (ADR 0012: the pinned reference environment)
import scipy
import scipy.special as sp

rng = np.random.default_rng(20260927)

UNARY = {
    "cbrt": (np.cbrt, (-50, 50)),
    "exp2": (np.exp2, (-30, 30)),
    "trunc": (np.trunc, (-10, 10)),
    "arcsinh": (np.arcsinh, (-20, 20)),
    "arccosh": (np.arccosh, (1, 40)),
    "arctanh": (np.arctanh, (-0.99, 0.99)),
    "degrees": (np.degrees, (-7, 7)),
    "radians": (np.radians, (-400, 400)),
    "erf": (sp.erf, (-4, 4)),
    "erfc": (sp.erfc, (-4, 8)),
    "gamma": (sp.gamma, (0.1, 20)),
    "lgamma": (sp.gammaln, (0.1, 200)),
    "signbit": (np.signbit, (-5, 5)),
}
BINARY = {
    "fmax": (np.fmax, (-5, 5)),
    "fmin": (np.fmin, (-5, 5)),
    "copysign": (np.copysign, (-5, 5)),
    "nextafter": (np.nextafter, (-5, 5)),
    "heaviside": (np.heaviside, (-5, 5)),
    "floatPower": (np.float_power, (0.1, 5)),
    "logaddexp": (np.logaddexp, (-50, 50)),
    "logaddexp2": (np.logaddexp2, (-50, 50)),
    "fmod": (np.fmod, (-20, 20)),
}


def enc(a):
    a = np.asarray(a)
    if a.dtype == np.bool_:
        return {"dtype": "bool", "shape": list(a.shape), "data": a.ravel().tolist()}
    return {"dtype": str(a.dtype), "shape": list(a.shape),
            "data": [special(v) if a.dtype.kind == "f" and not np.isfinite(v) else float(v) if a.dtype.kind == "f" else int(v)
                     for v in a.ravel()]}


def special(v):
    return "nan" if np.isnan(v) else ("inf" if v > 0 else "-inf")


cases = []
for name, (fn, (lo, hi)) in UNARY.items():
    x = rng.uniform(lo, hi, size=(3, 5))
    cases.append({"f": name, "args": [enc(x)], "expect": enc(fn(x))})
    x32 = x.astype(np.float32)
    cases.append({"f": name, "args": [enc(x32)], "expect": enc(fn(x32)), "tol": 2e-6})
    xi = np.round(x[0]).astype(np.int64)
    if name in ("arctanh",):
        continue
    r = fn(xi)
    cases.append({"f": name, "args": [enc(xi)], "expect": enc(r)})
# special values
cases.append({"f": "signbit", "args": [enc(np.array([-0.0, 0.0, -np.inf, 3.0]))], "expect": enc(np.signbit(np.array([-0.0, 0.0, -np.inf, 3.0])))})
cases.append({"f": "cbrt", "args": [enc(np.array([8.0, -27.0, 64.0, 1e-9]))], "expect": enc(np.cbrt(np.array([8.0, -27.0, 64.0, 1e-9]))), "tol": 0})

for name, (fn, (lo, hi)) in BINARY.items():
    a = rng.uniform(lo, hi, size=(4, 3))
    b = rng.uniform(lo, hi, size=(3,))              # broadcast along rows
    cases.append({"f": name, "args": [enc(a), enc(b)], "expect": enc(fn(a, b))})
    cases.append({"f": name, "args": [enc(a.astype(np.float32)), enc(b.astype(np.float32))],
                  "expect": enc(fn(a.astype(np.float32), b.astype(np.float32))), "tol": 2e-6})
    ai = np.round(a).astype(np.int64)
    bi = np.round(b).astype(np.int64)
    bi[bi == 0] = 3
    cases.append({"f": name, "args": [enc(ai), enc(bi)], "expect": enc(fn(ai, bi))})
# gcd / lcm are integer-only (NumPy raises on floats): test int64, int32 and uint8, with broadcasting.
for name, fn in (("gcd", np.gcd), ("lcm", np.lcm)):
    a = rng.integers(-30, 30, size=(4, 3))
    b = rng.integers(-30, 30, size=(3,))
    for dt in (np.int64, np.int32):
        cases.append({"f": name, "args": [enc(a.astype(dt)), enc(b.astype(dt))], "expect": enc(fn(a.astype(dt), b.astype(dt)))})
    au = rng.integers(0, 30, size=(4, 3)).astype(np.uint8)
    bu = rng.integers(1, 30, size=(3,)).astype(np.uint8)
    cases.append({"f": name, "args": [enc(au), enc(bu)], "expect": enc(fn(au, bu))})
# NaN handling of fmax/fmin, and logaddexp on equal / infinite inputs
nan_a = np.array([np.nan, 1.0, np.nan])
nan_b = np.array([2.0, np.nan, np.nan])
for name in ("fmax", "fmin"):
    cases.append({"f": name, "args": [enc(nan_a), enc(nan_b)], "expect": enc(BINARY[name][0](nan_a, nan_b))})
eq = np.array([3.0, -np.inf, 700.0])
cases.append({"f": "logaddexp", "args": [enc(eq), enc(eq)], "expect": enc(np.logaddexp(eq, eq))})

out = {"numpy": np.__version__, "scipy": scipy.__version__, "env": fixture_env.env(), "cases": cases}
fixture_env.require_pinned()
path = pathlib.Path(__file__).with_name("ufunc.json")
path.write_text(json.dumps(out))
print(f"{len(cases)} ufunc cases -> {path}")
