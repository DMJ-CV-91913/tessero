#!/usr/bin/env python3
r"""scipy.linalg.clarkson_woodruff_transform parity fixtures.

Tessero\Linalg\ClarksonWoodruff is a pure-PHP facade over Tessero's bit-exact PCG64 Generator. The
CountSketch matrix S (sketch_size x n) is built from exactly two RNG draws, in scipy's order:
    rows  = rng.integers(0, sketch_size, n)     # one target row per column
    signs = rng.choice([1, -1], n)              # +-1 per column
and the sketch is S @ A. Generator reproduces integers and choice-with-replacement bit-for-bit, so
the sketch matches scipy to machine precision (observed max abs diff 0.0; residual bounded by float
matmul ordering).

Expected sketches MUST come from clarkson_woodruff_transform(..., rng=np.random.default_rng(SEED)),
NEVER an int seed (an int builds a legacy MT19937 RandomState and would be unmatchable). The PHP side
receives the int SEED and rebuilds the identical PCG64 stream via Generator::defaultRng(SEED).

Input matrices are rounded to 6 decimals so the PHP-side input is exact. Comparison rtol 1e-9, atol
1e-11. Written to slinalg_clarkson.json under 'clarkson_cases'."""
import json
import os

import numpy as np
from numpy.random import default_rng
from scipy.linalg import clarkson_woodruff_transform as cwt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "tests", "fixtures", "parity", "slinalg_clarkson.json")


def mk(n, d, s):
    return np.round(np.random.RandomState(s).standard_normal((n, d)), 6)


# (A, sketch_size, seed) — a range of shapes, sketch sizes, seeds
SPECS = [
    (mk(6, 2, 0), 3, 11),
    (mk(5, 4, 1), 2, 7),
    (mk(8, 3, 2), 4, 42),
    (mk(10, 5, 3), 3, 2024),
    (mk(7, 1, 4), 2, 99),
    (mk(9, 3, 5), 5, 12345),
    (mk(12, 4, 6), 6, 2026),
]

cases = []
for A, k, seed in SPECS:
    S = cwt(A, k, rng=default_rng(seed))
    cases.append({
        "A": A.tolist(),
        "sketch_size": int(k),
        "seed": int(seed),
        "expect": np.asarray(S).tolist(),
        "rtol": 1e-9, "atol": 1e-11,
    })

doc = {"module": "scipy.linalg",
       "numpy": np.__version__, "scipy": __import__("scipy").__version__,
       "clarkson_cases": cases}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(doc, f, separators=(",", ":"), allow_nan=False)
print(f"wrote {len(cases)} cases -> {OUT}")
