#!/usr/bin/env python3
"""scipy.stats.uniform_direction parity fixtures (seeded PCG64 / default_rng).

Pure-PHP Tessero\\Stats\\UniformDirection reproduces scipy.stats.uniform_direction.rvs
bit-for-bit: it draws an array of standard normals of shape (*size, dim) from Tessero's
PCG64 Generator (numpy.random.default_rng equivalent, same ziggurat and draw order), then
normalises each trailing length-`dim` vector by its Euclidean norm (Marsaglia 1972 — exactly
scipy.stats._multivariate._sample_uniform_direction).

Because the expected values come from a seeded Generator (rng=np.random.default_rng(SEED)),
the fixture is host-independent (the PCG64 integer stream and the ziggurat are deterministic),
so no pinned-environment block is required.

Writes tests/fixtures/parity/stats_unifdir.json with a unifdir_cases block:
  {fn, dim, size, rng:{rng:SEED}, expect:<nested array>, rtol, atol}
The rng is encoded with the harness's {"rng": SEED} seed form (decodeValue -> defaultRng).
"""
import json
import os

import numpy as np
from scipy.stats import uniform_direction

OUT = os.environ.get('STATSUNIFDIR_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_unifdir.json')

cases = []


def add(dim, size, seed, rtol=1e-9, atol=1e-11):
    v = uniform_direction.rvs(dim, size, random_state=np.random.default_rng(seed))
    cases.append({
        'fn': 'uniform_direction',
        'dim': int(dim),
        'size': size,                 # None, int, or list -> JSON null / int / array
        'rng': {'rng': int(seed)},    # harness seed form -> Generator::defaultRng(seed)
        'expect': np.asarray(v, float).tolist(),
        'rtol': rtol,
        'atol': atol,
    })


# size None -> a single length-dim vector; int -> (m, dim); list -> (*list, dim).
add(1, None, 0)
add(2, None, 0)
add(3, None, 1)
add(3, 5, 2)
add(4, 3, 3)
add(5, None, 4)
add(10, 4, 5)
add(2, 8, 6)
add(7, 2, 7)
add(3, [2, 3], 8)
add(3, 3, 9)
add(6, 4, 10)

payload = json.dumps({'module': 'stats', 'scipy': '1.17.1', 'unifdir_cases': cases}, indent=2)
outdir = os.path.dirname(OUT)
if os.path.isdir(outdir):
    _w = open(OUT, 'w').write(payload)
    print('wrote', len(cases), 'cases to', OUT)
else:
    print(payload)
