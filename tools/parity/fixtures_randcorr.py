#!/usr/bin/env python3
r"""scipy.stats.random_correlation + random_table parity fixtures (seeded PCG64 / default_rng).

Tessero\Stats\RandomCorrelation and Tessero\Stats\RandomTable are pure-PHP facades over Tessero's
bit-exact PCG64 Generator. random_correlation draws a single ortho_group matrix and rotates it to
unit diagonal with Givens/BLAS-drot (near-bit, ~1e-15). random_table reproduces scipy's _rcont C
exactly: Boyett (rcont1) via next_uint32 masked-rejection and Patefield (rcont2) via next_double,
drawing from the SAME PCG64 stream through the kernel's tsr_pcg64_uint64 / tsr_pcg64_random
primitives — so the sampled integer table is bit-identical to scipy.

Expected matrices/tables MUST come from random_state=np.random.default_rng(SEED), NEVER an int seed
(an int builds a legacy MT19937 RandomState whose stream differs from PCG64). The PHP side receives
the int seed and rebuilds the identical PCG64 stream via Generator::defaultRng(seed).

Comparison: random_correlation TIGHT near-bit (rtol 1e-9, atol 1e-11; observed ~9e-16);
random_table integer-exact. Written to tests/fixtures/parity/stats_randcorr.json with keys
'randcorr_cases' and 'randtable_cases' consumed by run-tests.php. Strict JSON (allow_nan=False)."""
import json
import os

import numpy as np
from scipy.stats import random_correlation, random_table

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests", "fixtures", "parity", "stats_randcorr.json",
)

# ----- random_correlation: (eigs, seed). eigs must be >= 0 and sum to len(eigs). -----
RANDCORR_SPECS = [
    ([0.5, 0.8, 1.2, 1.5], 0),
    ([0.5, 0.8, 1.2, 1.5], 42),
    ([1.0, 1.0, 1.0], 1),
    ([0.2, 0.4, 1.4, 2.0], 123),
    ([1.5, 0.5], 7),
    ([0.5, 0.5, 1.0, 1.0, 2.0], 2024),
    ([0.3, 0.3, 0.3, 1.0, 1.0, 3.1], 99),
    ([0.1, 0.9, 1.0, 2.0], 2026),
]

randcorr_cases = []
for eigs, seed in RANDCORR_SPECS:
    m = random_correlation.rvs(eigs, random_state=np.random.default_rng(seed))
    randcorr_cases.append({"eigs": [float(e) for e in eigs], "seed": seed, "matrix": m.tolist()})

# ----- random_table: (row, col, method, seed). method in auto/boyett/patefield. -----
RANDTABLE_SPECS = [
    # small -> auto selects boyett
    ([1, 5], [2, 3, 1], "auto", 0),
    ([1, 5], [2, 3, 1], "auto", 123),
    ([2, 2], [2, 2], "auto", 42),
    ([3, 3, 4], [2, 4, 4], "auto", 7),
    ([6, 4], [5, 5], "auto", 2024),
    # explicit boyett
    ([1, 5], [2, 3, 1], "boyett", 1),
    ([3, 3, 4], [2, 4, 4], "boyett", 99),
    # large -> auto selects patefield
    ([10, 20, 30], [25, 15, 20], "auto", 0),
    ([10, 20, 30], [25, 15, 20], "auto", 2026),
    ([50, 50], [30, 40, 30], "auto", 5),
    ([100, 200, 300, 400], [250, 350, 400], "auto", 42),
    # explicit patefield (incl. margins beyond the logfactorial table of 126)
    ([10, 20, 30], [25, 15, 20], "patefield", 1),
    ([300, 400, 300], [200, 400, 400], "patefield", 7),
    ([1000, 2000], [1500, 1500], "patefield", 2024),
]

randtable_cases = []
for row, col, method, seed in RANDTABLE_SPECS:
    kw = {} if method == "auto" else {"method": method}
    t = random_table.rvs(row, col, random_state=np.random.default_rng(seed), **kw)
    # scipy returns float64 with exact-integer entries; store as ints.
    randtable_cases.append({
        "row": [int(x) for x in row],
        "col": [int(x) for x in col],
        "method": method,
        "seed": seed,
        "table": [[int(v) for v in r] for r in t.tolist()],
    })

doc = {
    "module": "scipy.stats",
    "tol": {"rtol": 1e-9, "atol": 1e-11},
    "randcorr_cases": randcorr_cases,
    "randtable_cases": randtable_cases,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(doc, f, allow_nan=False)
print("wrote %d randcorr + %d randtable cases -> %s"
      % (len(randcorr_cases), len(randtable_cases), OUT))
