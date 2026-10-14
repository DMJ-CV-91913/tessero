#!/usr/bin/env python3
r"""scipy.sparse.random_array / random / rand parity fixtures.

Tessero\Sparse\SparseRandom is a pure-PHP facade over Tessero's bit-exact PCG64 Generator. It
reproduces scipy.sparse._construct._random exactly: it draws `size = round(density*prod(shape))`
distinct raveled positions with numpy.random.Generator.choice(replace=False, p=None), unravels them
in Fortran (column-major) order, then draws the data with rng.uniform(size=size) AFTER the index
choice. Because Tessero's Generator reproduces NumPy's PCG64 stream (Lemire bounded integers and the
uniform doubles) bit-for-bit, the resulting COO (row, col, data) triplets match scipy byte-for-byte.

numpy's choice-without-replacement uses TWO algorithms, both covered here: Floyd's hash-set algorithm
(pop <= 10000 or size <= pop/50) and the tail-shuffle branch (pop > 10000 and size > pop/50). The
150x100 @ 0.05 and 200x200 @ 0.03 cases exercise the tail-shuffle branch; the rest exercise Floyd.

Expected values MUST come from rng=np.random.default_rng(SEED), NEVER an int seed (an int builds a
legacy MT19937 RandomState whose stream differs from PCG64 and would be unmatchable). The PHP side
receives the int seed and rebuilds the identical PCG64 stream via Generator::defaultRng(seed).

Comparison is TIGHT: row/col indices are integer-exact; data is compared at rtol 1e-12, atol 0 (the
observed max abs diff is 0.0 -- byte-exact). Written to sparse_random.json under the
'sparse_random_cases' key consumed by run-tests.php."""
import json
import os

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'fixtures', 'parity', 'sparse_random.json')

# (fn, rows, cols, density, seed). fn in {random_array, random, rand}.
# random/rand take (m, n); random_array takes shape=(rows, cols). Same underlying math.
SPECS = [
    ('random_array', 5, 4, 0.5, 0),
    ('random_array', 10, 10, 0.3, 1),
    ('random_array', 8, 8, 0.25, 3),
    ('random_array', 3, 7, 0.4, 5),
    ('random_array', 7, 7, 1.0, 13),        # full density (size == pop)
    ('random_array', 20, 15, 0.1, 17),
    ('random_array', 100, 100, 0.001, 7),   # large pop, tiny density -> Floyd
    ('random_array', 150, 100, 0.05, 9),    # tot=15000>10000, size=750>300 -> tail-shuffle
    ('random_array', 200, 200, 0.03, 11),   # tot=40000, size=1200>800    -> tail-shuffle
    ('random', 6, 9, 0.35, 2),
    ('random', 12, 4, 0.5, 8),
    ('rand', 5, 5, 0.4, 4),
    ('rand', 9, 6, 0.2, 21),
    ('random_array', 4, 4, 0.0, 6),         # zero density -> empty
]

cases = []
for fn, r, c, d, seed in SPECS:
    rng = np.random.default_rng(seed)
    if fn == 'random_array':
        M = sp.random_array((r, c), density=d, rng=rng, format='coo')
    elif fn == 'random':
        M = sp.random(r, c, density=d, rng=rng, format='coo')
    else:  # rand
        M = sp.rand(r, c, density=d, rng=rng, format='coo')
    M = M.tocoo()
    cases.append({
        'fn': fn,
        'rows': r,
        'cols': c,
        'density': d,
        'seed': seed,
        'nnz': int(M.nnz),
        'row': M.row.astype(int).tolist(),
        'col': M.col.astype(int).tolist(),
        'data': M.data.astype(float).tolist(),
    })

doc = {
    'module': 'scipy.sparse',
    'tol': {'rtol': 1e-12, 'atol': 0.0},
    'sparse_random_cases': cases,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(doc, f, allow_nan=False)
print(f"wrote {len(cases)} cases -> {OUT}")
