#!/usr/bin/env python3
r"""scipy.stats.qmc.Sobol + scipy.stats.sobol_indices parity fixtures.

Tessero\Stats\Qmc\Sobol is a pure-PHP Joe&Kuo Sobol' engine (direction-number table embedded, Gray-code
construction) over Tessero's bit-exact PCG64 Generator + SeedSequence spawn (for the LMS+shift scramble).
The UNSCRAMBLED sequence is deterministic and matches scipy to the last bit; the SCRAMBLED variant draws
from the single spawned child of default_rng(seed) (spawn_key (0,)) exactly as scipy's _rng_spawn does, so
it is bit-exact too. Verified at bits=30 and bits=64 across dims and seeds (rel == 0).

Tessero\Stats\SobolIndices reproduces scipy.stats.sobol_indices: a Saltelli A/B/AB design from a scrambled
64-bit Sobol' sequence (d=2*dim) + the default saltelli_2010 estimator with the Sobol'-Levitan mean
normalisation. The Sobol' points are bit-exact; the final estimator reductions agree to ~1e-15 (PHP's
sequential summation vs numpy's pairwise). Comparison rtol 1e-12.

Two case groups are written to stats_sobol.json:
  - 'sobol_cases'        : engine draws (d, bits, n, scramble, seed) -> expected (n, d) points.
  - 'sobolindices_cases' : (kind, n, seed, dists) -> expected first_order/total_order indices.

The sobol_indices model functions are NAMED kinds the PHP runner rebuilds; the dists are uniform(loc,scale)
pairs (uniform.ppf(q) = loc + scale*q, applied exactly by both sides)."""
import json
import math
import os

import numpy as np
import scipy
from scipy.stats import qmc, sobol_indices, uniform
from scipy.stats._sensitivity_analysis import f_ishigami

from fixture_env import require_pinned

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "stats_sobol.json")

require_pinned()

# ---- engine cases ----------------------------------------------------------------------------
# (dim, bits, n, scramble, seed)
ENGINE_SPECS = [
    (1, 30, 1, False, None), (1, 30, 8, False, None),
    (2, 30, 16, False, None), (3, 30, 256, False, None),
    (6, 30, 64, False, None), (10, 30, 128, False, None),
    (2, 64, 16, False, None), (6, 64, 64, False, None), (8, 64, 8, False, None),
    (2, 30, 32, True, 0), (2, 30, 32, True, 1), (3, 30, 64, True, 42),
    (6, 30, 16, True, 2026), (4, 30, 64, True, 1),
    (2, 64, 128, True, 0), (6, 64, 64, True, 2024), (3, 64, 256, True, 7),
    (8, 64, 32, True, 99), (12, 64, 16, True, 2026), (1, 30, 8, True, 5),
]
engine_cases = []
for dim, bits, n, scr, seed in ENGINE_SPECS:
    rng = np.random.default_rng(seed) if scr else 0
    eng = qmc.Sobol(d=dim, scramble=scr, rng=rng, bits=bits)
    pts = eng.random(n)
    engine_cases.append({
        "dim": dim, "bits": bits, "n": n, "scramble": scr,
        "seed": (seed if scr else None),
        "expect": pts.tolist(),
    })

# ---- sobol_indices cases ---------------------------------------------------------------------
def f_linear(x):
    x = np.atleast_2d(x)
    return x[0] + 2 * x[1] + 3 * x[2]


def f_poly(x):
    x = np.atleast_2d(x)
    return x[0] ** 2 + x[1] * x[2] + np.sin(x[0])


def f_sum4(x):
    x = np.atleast_2d(x)
    return x[0] + x[1] + x[2] + x[3]


SI_FUNCS = {
    "ishigami": (f_ishigami, [(-math.pi, 2 * math.pi)] * 3),
    "linear": (f_linear, [(0.0, 1.0)] * 3),
    "poly": (f_poly, [(0.0, 1.0)] * 3),
    "sum4": (f_sum4, [(0.0, 1.0)] * 4),
}
# (kind, n, seed)
SI_SPECS = [
    ("ishigami", 256, 0), ("ishigami", 1024, 42), ("ishigami", 512, 7),
    ("linear", 256, 1), ("linear", 1024, 0),
    ("poly", 512, 2024), ("poly", 256, 99),
    ("sum4", 256, 0), ("sum4", 1024, 5),
]
si_cases = []
for kind, n, seed in SI_SPECS:
    fn, dspec = SI_FUNCS[kind]
    dists = [uniform(loc=l, scale=s) for l, s in dspec]
    res = sobol_indices(func=fn, n=n, dists=dists, rng=np.random.default_rng(seed))
    si_cases.append({
        "kind": kind, "n": n, "seed": seed,
        "dists": [[l, s] for l, s in dspec],
        "first_order": np.atleast_2d(res.first_order).tolist(),
        "total_order": np.atleast_2d(res.total_order).tolist(),
        "rtol": 1e-12,
    })

doc = {
    "module": "scipy.stats",
    "numpy": np.__version__, "scipy": scipy.__version__,
    "tol": {"rtol": 1e-12, "atol": 1e-14},
    "sobol_cases": engine_cases,
    "sobolindices_cases": si_cases,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(doc, f, separators=(",", ":"), allow_nan=False)
print(f"wrote {len(engine_cases)} engine + {len(si_cases)} sobol_indices cases -> {OUT}")
