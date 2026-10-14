#!/usr/bin/env python3
r"""scipy.stats.unitary_group parity fixtures.

Tessero\Stats\UnitaryGroup is a pure-PHP facade over Tessero's bit-exact PCG64 Generator and the
complex QR kernel routine linalg.qr_complex (LAPACKE_zgeqrf + zungqr). unitary_group.rvs(dim) draws
two dim x dim standard-normal matrices (real then imaginary block, in NumPy's exact C-order),
forms z = (re + 1j*im)/sqrt(2), complex-QR-decomposes it, and multiplies each column of Q by
phase(diag(R)) = d/|d|; this phase normalisation makes the result independent of the complex-QR
implementation's internal reflector phase, so any correct complex QR (either backend) reproduces
scipy bit-for-bit.

Expected matrices MUST come from a numpy Generator (random_state=np.random.default_rng(SEED)),
NEVER an int seed (an int builds a legacy MT19937 RandomState whose stream differs from PCG64 and
would be unmatchable). The PHP side DOES receive the int seed and rebuilds the identical PCG64
stream via Generator::defaultRng(seed), matching default_rng(seed).

Comparison is TIGHT (rtol 1e-9, atol 1e-11): observed max abs diff is ~1.57e-16 (machine epsilon)
across all cases. Complex expectations are stored as [re, im] pairs. Written to
stats_unitarygroup.json under the 'unitarygroup_cases' key consumed by run-tests.php."""
import json, os
import numpy as np
from scipy.stats import unitary_group

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'stats_unitarygroup.json')

# (dim, seed) covering dim 1..8, odd/even, small + large seeds
SPECS = [(1, 3), (2, 0), (2, 12345), (3, 1), (3, 99), (4, 7),
         (5, 42), (6, 2026), (7, 5), (8, 101)]

cases = []
for dim, seed in SPECS:
    u = unitary_group.rvs(dim, random_state=np.random.default_rng(seed))
    matrix = [[[float(u[i, j].real), float(u[i, j].imag)] for j in range(dim)] for i in range(dim)]
    cases.append({'fn': 'unitary_group', 'dim': dim, 'seed': seed, 'matrix': matrix})

doc = {'module': 'scipy.stats',
       'tol': {'rtol': 1e-9, 'atol': 1e-11},
       'unitarygroup_cases': cases}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(doc, f)
print(f"wrote {len(cases)} cases -> {OUT}")
