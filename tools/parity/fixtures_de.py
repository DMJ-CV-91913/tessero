#!/usr/bin/env python3
"""scipy.optimize.differential_evolution parity fixtures (strategy='best1bin', polish=False).

The PHP port drives Tessero's bit-exact PCG64 Generator from the SAME seed, so it reproduces SciPy's exact
iterate sequence. Expected values therefore MUST come from a numpy Generator (rng=np.random.default_rng(SEED)),
never from an int seed - with an int SciPy builds a legacy RandomState whose draw stream differs from PCG64.

Objectives are convex with a known minimiser (sphere / shifted-quadratic), rebuilt in PHP by named kind, so
comparison is a LOOSE x check (rtol ~1e-4): on a convex basin DE converges to the optimum regardless of any
residual draw mismatch, while the exact-stream port also matches nfev/nit (see VERIFY.txt).

These are backend-neutral pure-PHP iterative solvers (no BLAS); host-independent, no pinned environment needed.
Merged into optimize.json under the existing 'solver_cases' key (consumed by run-tests.php)."""
import json, os
import numpy as np
import scipy.optimize as so

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'optimize.json')


def sphere(x):
    return float(np.sum(np.asarray(x, float) ** 2))


def shifted_quadratic(x, shift):
    x = np.asarray(x, float)
    s = np.asarray(shift, float)
    return float(np.sum((x - s) ** 2))


# (kind, bounds, shift, seed, known optimum)
SPECS = [
    {'kind': 'sphere', 'bounds': [[-5.0, 5.0], [-5.0, 5.0]], 'shift': [0.0, 0.0], 'seed': 42},
    {'kind': 'shifted_quadratic', 'bounds': [[-5.0, 5.0], [-5.0, 5.0]], 'shift': [1.5, -2.0], 'seed': 7},
    {'kind': 'sphere', 'bounds': [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]], 'shift': [0.0, 0.0, 0.0], 'seed': 123},
    {'kind': 'shifted_quadratic', 'bounds': [[-3.0, 6.0], [-4.0, 4.0], [-2.0, 8.0]], 'shift': [2.0, -1.0, 3.5], 'seed': 2024},
]

de_cases = []
for spec in SPECS:
    shift = spec['shift']
    if spec['kind'] == 'sphere':
        func = sphere
    else:
        func = lambda x, _s=shift: shifted_quadratic(x, _s)
    res = so.differential_evolution(
        func, spec['bounds'], strategy='best1bin', polish=False,
        rng=np.random.default_rng(spec['seed']),          # Generator (PCG64), NOT an int seed
    )
    de_cases.append({
        'fn': 'differential_evolution',
        'obj': {'kind': spec['kind'], 'shift': shift},     # rebuilt in PHP via named kind
        'bounds': spec['bounds'],
        'seed': spec['seed'],
        'expect': list(map(float, res.x)),
        'expect_fun': float(res.fun),
        'expect_nfev': int(res.nfev),
        'expect_nit': int(res.nit),
        'rtol': 1e-4,                                      # LOOSE x compare on convex basin
    })

# Merge into the existing optimize.json (append to solver_cases) so run-tests.php picks them up in one module.
if os.path.exists(OUT):
    with open(OUT) as f:
        data = json.load(f)
    data.setdefault('solver_cases', [])
    # drop any stale DE cases, then append the fresh ones (idempotent re-runs)
    data['solver_cases'] = [c for c in data['solver_cases'] if c.get('fn') != 'differential_evolution']
    data['solver_cases'].extend(de_cases)
else:
    data = {'module': 'optimize', 'scipy': __import__('scipy').__version__, 'solver_cases': de_cases}

with open(OUT, 'w') as f:
    json.dump(data, f, separators=(',', ':'))
print(f'optimize: +{len(de_cases)} differential_evolution cases')
