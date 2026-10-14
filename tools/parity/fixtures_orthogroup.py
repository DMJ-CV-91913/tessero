#!/usr/bin/env python3
r"""scipy.stats.ortho_group / special_ortho_group parity fixtures.

Tessero\Stats\OrthoGroup is a pure-PHP facade over Tessero's bit-exact PCG64 Generator and the
backend-neutral Linalg::qr. ortho_group.rvs(dim) draws one dim x dim standard-normal matrix in
NumPy's exact C-order, QR-decomposes it, and multiplies each column of Q by sign(diag(R)); this
positive-diagonal normalisation makes the result independent of the QR implementation's internal
sign choice, so any correct QR (either backend) reproduces scipy bit-for-bit. special_ortho_group
additionally scales row 0 by 1/det(q) so det = +1.

Expected matrices MUST come from a numpy Generator (random_state=np.random.default_rng(SEED)),
NEVER an int seed (an int builds a legacy MT19937 RandomState whose stream differs from PCG64 and
would be unmatchable). The PHP side DOES receive the int seed and rebuilds the identical PCG64
stream via Generator::defaultRng(seed), matching default_rng(seed).

Comparison is TIGHT (rtol 1e-9, atol 1e-11): observed max abs diff is 0.0 (byte-exact) across all
cases, but the tolerance leaves room for a ~1e-13 last-digit QR drift on other hosts. Written to
stats_orthogroup.json under the 'orthogroup_cases' key consumed by run-tests.php."""
import json, os
import numpy as np
from scipy.stats import ortho_group, special_ortho_group

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'stats_orthogroup.json')

# (dim, seed) covering dim 1..8, odd/even, small + large seeds
SPECS = [(1, 3), (2, 0), (2, 12345), (3, 1), (3, 99), (4, 7),
         (5, 42), (6, 2026), (7, 5), (8, 101)]

cases = []
for dim, seed in SPECS:
    o = ortho_group.rvs(dim, random_state=np.random.default_rng(seed))
    s = special_ortho_group.rvs(dim, random_state=np.random.default_rng(seed))
    cases.append({'fn': 'ortho_group', 'dim': dim, 'seed': seed,
                  'matrix': o.tolist()})
    cases.append({'fn': 'special_ortho_group', 'dim': dim, 'seed': seed,
                  'matrix': s.tolist()})

doc = {'module': 'scipy.stats',
       'tol': {'rtol': 1e-9, 'atol': 1e-11},
       'orthogroup_cases': cases}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(doc, f)
print(f"wrote {len(cases)} cases -> {OUT}")
