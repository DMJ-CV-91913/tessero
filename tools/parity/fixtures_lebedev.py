#!/usr/bin/env python3
r"""scipy.integrate.lebedev_rule parity fixtures. For each supported order n the fixture stores the exact
nodes x (shape (3, m)) and weights w (shape (m,)); Tessero\Integrate\Lebedev::lebedevRule(n) reproduces
them bit-for-bit (pure-PHP port of scipy._lebedev; PHP M_PI/sqrt are bit-identical to NumPy here). Compared
with tol ~1e-13 (the port is actually exact). Host-independent (no kernel, no LAPACK)."""
import json, os
from scipy.integrate import lebedev_rule

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "integrate_lebedev.json")

ORDERS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47,
          53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131]

cases = []
for n in ORDERS:
    x, w = lebedev_rule(n)
    cases.append({
        "fn": "lebedev_rule",
        "n": int(n),
        "x": [[float(v) for v in x[i]] for i in range(3)],
        "w": [float(v) for v in w],
        "rtol": 1e-13,
    })

out = {"module": "integrate", "scipy": __import__("scipy").__version__, "lebedev_cases": cases}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(f"integrate lebedev: {len(cases)} orders -> {OUT}")
