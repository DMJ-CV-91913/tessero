#!/usr/bin/env python3
r"""scipy.optimize.dual_annealing parity fixtures (no_local_search=True ONLY).

The PHP port (Tessero\Optimize\DualAnnealing) drives Tessero's bit-exact PCG64 Generator from the SAME seed,
so it reproduces SciPy's exact VisitingDistribution / StrategyChain draw stream. Expected values therefore MUST
come from a numpy Generator (rng=np.random.default_rng(SEED)), NEVER an int seed passed to scipy - with an int
scipy builds a legacy RandomState (MT19937) whose stream differs from PCG64. (The PHP side DOES receive the int
seed and rebuilds the identical PCG64 stream via Generator::defaultRng(seed), matching default_rng(seed).)

Objectives are NAMED (sphere/rastrigin/ackley/rosen), rebuilt in PHP by kind in makeObjective(), so no callable is
serialised. Because the draw stream matches exactly, comparison is TIGHT: x and fun to rtol 1e-12, plus nfev/nit.
no_local_search=True is the only covered path: the default L-BFGS-B refinement is not bit-exact (Tessero lbfgsb
diverges from scipy), so dual annealing WITH local search is not yet achievable. Backend-neutral pure-PHP (no BLAS),
host-independent. Merged into optimize.json under the existing 'solver_cases' key (consumed by run-tests.php)."""
import json, os
import numpy as np
from scipy.optimize import dual_annealing

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'optimize.json')


def rastrigin(x):
    x = np.asarray(x, float)
    return float(np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)) + 10.0 * x.size)


def sphere(x):
    x = np.asarray(x, float)
    return float(np.sum(x * x))


def ackley(x):
    x = np.asarray(x, float); n = x.size
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.sum(x * x) / n))
                 - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / n) + 20.0 + np.e)


def rosen(x):
    x = np.asarray(x, float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


OBJS = {'rastrigin': rastrigin, 'sphere': sphere, 'ackley': ackley, 'rosen': rosen}

# (obj_kind, dim, bound, seed, maxiter)
SPECS = [
    ('rastrigin', 1, 5.12, 1234, 50),
    ('rastrigin', 2, 5.12, 1234, 50),
    ('rastrigin', 2, 5.12, 7, 80),
    ('rastrigin', 3, 5.12, 7, 60),
    ('sphere', 2, 5.0, 42, 50),
    ('sphere', 3, 2.0, 99, 40),
    ('ackley', 2, 5.0, 11, 60),
    ('rosen', 2, 2.5, 3, 80),
]

da_cases = []
for kind, dim, bnd, seed, maxiter in SPECS:
    func = OBJS[kind]
    bounds = [(-bnd, bnd)] * dim
    res = dual_annealing(func, bounds, rng=np.random.default_rng(seed),  # Generator (PCG64), NOT an int seed
                         maxiter=maxiter, no_local_search=True)
    da_cases.append({
        'fn': 'dual_annealing',
        'obj': {'kind': kind},                              # rebuilt in PHP via named kind
        'bounds': [list(b) for b in bounds],
        'seed': seed,
        'maxiter': maxiter,
        'expect': list(map(float, res.x)),                  # 'expect' == x, matching the DE arm's key
        'expect_fun': float(res.fun),
        'expect_nfev': int(res.nfev),
        'expect_nit': int(res.nit),
        'rtol': 1e-12,                                      # TIGHT: exact PCG64 stream
    })

# Merge into the existing optimize.json (append to solver_cases) so run-tests.php picks them up in one module.
if os.path.exists(OUT):
    with open(OUT) as f:
        data = json.load(f)
    data.setdefault('solver_cases', [])
    # drop any stale DA cases, then append the fresh ones (idempotent re-runs)
    data['solver_cases'] = [c for c in data['solver_cases'] if c.get('fn') != 'dual_annealing']
    data['solver_cases'].extend(da_cases)
else:
    data = {'module': 'optimize', 'scipy': __import__('scipy').__version__, 'solver_cases': da_cases}

with open(OUT, 'w') as f:
    json.dump(data, f, separators=(',', ':'))
print(f'optimize: +{len(da_cases)} dual_annealing cases')
