#!/usr/bin/env python3
"""scipy.spatial.cKDTree fixtures: build a tree from points, query nearest neighbours, compare distances and
indices. Exact nearest-neighbour search (no BLAS), host-independent; no pinned environment required."""
import json, os
import numpy as np
from scipy.spatial import cKDTree

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'spatial.json')
rng = np.random.default_rng(20261001)


def case(data, query, k):
    d, i = cKDTree(data).query(query, k=k)
    return {'fn': 'cKDTree', 'data': data.tolist(), 'query': query.tolist(), 'k': k,
            'expect': {'distances': np.asarray(d, dtype=float).tolist(), 'indices': np.asarray(i).tolist()}}


cases = [
    case(rng.uniform(-5, 5, (20, 2)), rng.uniform(-5, 5, (5, 2)), 1),
    case(rng.uniform(-5, 5, (20, 2)), rng.uniform(-5, 5, (5, 2)), 3),
    case(rng.uniform(-5, 5, (30, 3)), rng.uniform(-5, 5, (4, 3)), 1),
    case(rng.uniform(-5, 5, (30, 3)), rng.uniform(-5, 5, (4, 3)), 2),
    case(rng.uniform(0, 10, (15, 1)), rng.uniform(0, 10, (6, 1)), 1),
    case(rng.uniform(-2, 2, (12, 4)), rng.uniform(-2, 2, (3, 4)), 2),
]

out = {'module': 'spatial', 'scipy': __import__('scipy').__version__, 'spatial_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'spatial: cKDTree {len(cases)} cases')
