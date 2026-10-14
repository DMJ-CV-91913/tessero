#!/usr/bin/env python3
r"""scipy.integrate.qmc_quad parity fixtures.

Tessero\Integrate\QmcQuad is a pure-PHP facade over Tessero's bit-exact PCG64 Generator and a
reproduction of NumPy's SeedSequence.spawn (Tessero\Random\SeedSequence). qmc_quad runs
`n_estimates` scrambled-Halton estimates and averages them; each estimate's Halton engine is a
SPAWNED child of the seed, so the whole result is determined by (a) the spawn tree and (b) the
Owen-scramble shuffle draws. Both are reproduced bit-for-bit, giving rel == 0 vs scipy.

The spawn tree (verified against numpy): default_rng(SEED) is SeedSequence(SEED) with spawn_key ();
Halton(seed=rng) spawns once -> self.rng key (0,); qmc_quad's _rng_spawn gives children (0,i); each
subsequent Halton spawns once more -> (0,i,0). Estimate 0 uses the base Halton (key (0,)).

Expected integrals MUST come from qmc_quad with qrng=qmc.Halton(d, seed=np.random.default_rng(SEED)),
NEVER an int seed (an int builds a legacy MT19937 RandomState whose stream differs from PCG64 and
would be unmatchable). The PHP side receives the int SEED and rebuilds the identical PCG64/
SeedSequence tree via Tessero\Random\SeedSequence(SEED).

The integrand is a NAMED kind the PHP runner rebuilds; only the scalar integral is compared. Observed
max rel diff is ~1e-15 (machine precision; the maths is identical, the tiny residual is float
summation order). Comparison rtol 1e-10. Written to integrate_qmcquad.json under 'qmcquad_cases'."""
import json
import os

import numpy as np
from scipy.integrate import qmc_quad
from scipy.stats import qmc

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "integrate_qmcquad.json")


def integrand(kind):
    return {
        "sum":    lambda x: np.sum(x, axis=0),
        "gauss":  lambda x: np.exp(-np.sum(x ** 2, axis=0)),
        "cosprod": lambda x: np.prod(np.cos(x), axis=0),
        "sumsq1": lambda x: np.sum(x ** 2, axis=0) + 1.0,
        "invsum": lambda x: 1.0 / (1.0 + np.sum(x, axis=0)),
    }[kind]


# (kind, a, b, n_estimates, n_points, seed)
SPECS = [
    ("sum",     [0.0, 0.0],      [1.0, 1.0],      8, 64,  12345),
    ("gauss",   [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 6, 128, 7),
    ("cosprod", [0.0],           [1.0],           4, 32,  99),
    ("sumsq1",  [0.0, 0.0],      [2.0, 1.0],      8, 256, 2024),
    ("invsum",  [0.0, 0.0],      [1.0, 1.0],      5, 100, 42),
    ("gauss",   [0.0, 0.0],      [1.0, 1.0],      8, 128, 2026),
    ("sum",     [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 6, 64,  5),
]

cases = []
for kind, a, b, ne, npn, seed in SPECS:
    qr = qmc.Halton(d=len(a), seed=np.random.default_rng(seed))
    res = qmc_quad(integrand(kind), a, b, n_estimates=ne, n_points=npn, qrng=qr)
    cases.append({
        "kind": kind, "a": a, "b": b,
        "n_estimates": ne, "n_points": npn, "seed": seed,
        "expect": float(res.integral),
        "rtol": 1e-10,
    })

doc = {"module": "scipy.integrate",
       "numpy": np.__version__, "scipy": __import__("scipy").__version__,
       "qmcquad_cases": cases}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(doc, f, separators=(",", ":"), allow_nan=False)
print(f"wrote {len(cases)} cases -> {OUT}")
