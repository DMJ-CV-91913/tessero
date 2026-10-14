#!/usr/bin/env python3
"""Generate tests/fixtures/parity/optimize_trustconstr.json.

scipy.optimize.minimize(method='trust-constr') verified converged x for a set of
non-degenerate problems (unconstrained, bound, linear-eq, linear-ineq+bounds,
nonlinear-eq, mixed, and a 5-D user-jac case). The two degenerate boundary cases
from the de-risk set (circle-tangent / interval-ring) are intentionally excluded:
their minimizer is weakly determined along an active-constraint arc, so scipy and
any independent reimplementation only agree to ~1e-4 at the default gtol (the
objective value matches to machine eps). Each prob is rebuilt inline on the PHP
side (the trust_constr branch in run-tests.php); this file carries only the
scipy-converged x (scipy 1.17.1) and the comparison tolerance.
"""
import json
import os

ROOT = os.path.join(os.environ["HOME"], "tessero")
OUT = os.path.join(ROOT, "tests", "fixtures", "parity", "optimize_trustconstr.json")

# scipy 1.17.1 minimize(method='trust-constr') converged x per problem.
EXPECT = {
    "p1_rosen3":      [0.9999973115201615, 0.9999946245242296, 0.9999892415889771],
    "p2_boundquad":   [0.9999311174800009, -0.9999742346243773],
    "p3_lineq":       [0.49999999741410905, 0.5000000025858906],
    "p4_linineq_bnd": [2.4999091376723332, 1.4999304747638902],
    "p6_nleq":        [1.3660253978837673, 1.8660253876490869],
    "p7_mixed":       [0.7058586741126519, 0.7058586741126519],
    "p8_rosen5_jac":  [0.9999999999999559, 0.9999999999998627, 0.9999999999997544,
                       0.9999999999994668, 0.9999999999989759],
}

cases = [
    {"fn": "trust_constr", "prob": prob, "expect": x, "rtol": 1e-5}
    for prob, x in EXPECT.items()
]

data = {
    "module": "optimize",
    "_meta": {
        "scipy": "1.17.1",
        "method": "trust-constr",
        "note": "non-degenerate problems; degenerate boundary cases excluded (see docstring)",
    },
    "solver_cases": cases,
}

with open(OUT, "w") as f:
    json.dump(data, f, indent=2, allow_nan=False)
print(f"wrote {OUT}: {len(cases)} trust_constr cases")
