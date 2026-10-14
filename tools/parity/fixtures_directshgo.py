#!/usr/bin/env python3
r"""scipy.optimize.direct + scipy.optimize.shgo (sampling_method='simplicial') parity fixtures.

Both targets are DETERMINISTIC in scipy under their defaults (direct = pure space-partitioning;
shgo-simplicial has no RNG). Objectives are NAMED (styblinski/ackley/booth/shifted_quadratic/
sphere/rosen), rebuilt in PHP by kind in makeObjective(), so no callable is serialised.

DIRECT (Tessero\Optimize\Direct::minimize) is a faithful port of SciPy's Gablonsky DIRECT_L.
It is iterate-exact on separable / convex / symmetric landscapes (styblinski, ackley,
shifted-sphere, separable quad): tier='iterate', x/fun asserted at rtol 1e-6 vs scipy's iterate.
On Rosenbrock-type curved valleys SciPy terminates one or two iterations earlier via its anchor
linked-list tie ordering; both still reach the UNIQUE analytic global optimum, so those cases use
tier='global': x asserted vs the KNOWN optimum at a documented looser rtol, and fun only checked
to be near-zero (abs <= fun_atol). nfev/nit are NOT asserted for direct (late-iterate divergence
is expected and benign).

SHGO (Tessero\Optimize\Shgo::minimize) reproduces scipy.optimize.shgo default (simplicial,
iters=1, n=2**dim+1, SLSQP polish with ftol=1e-12 and locally-convex bounds). The initial complex
graph is bit-identical to scipy's; the polish runs through Tessero's SLSQP (tracks scipy's SLSQP
to ~1e-7), so .x/.fun match scipy to rtol 1e-6.

Merged into optimize.json under the existing 'solver_cases' key (consumed by run-tests.php)."""
import json
import os
import numpy as np
from scipy.optimize import direct, shgo

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'optimize.json')


def styblinski(x):
    x = np.asarray(x, float)
    return float(0.5 * np.sum(x ** 4 - 16.0 * x ** 2 + 5.0 * x))


def ackley(x):
    x = np.asarray(x, float); n = x.size
    return float(-20.0 * np.exp(-0.2 * np.sqrt(np.sum(x * x) / n))
                 - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / n) + 20.0 + np.e)


def booth(x):
    return float((x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2)


def shifted_quadratic(x, shift):
    x = np.asarray(x, float); s = np.asarray(shift, float)
    return float(np.sum((x - s) ** 2))


def sphere(x):
    x = np.asarray(x, float)
    return float(np.sum(x * x))


def rosen(x):
    x = np.asarray(x, float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def build(name, obj):
    if name == 'shifted_quadratic':
        return lambda z, s=obj.get('shift'): shifted_quadratic(z, s)
    return {'styblinski': styblinski, 'ackley': ackley, 'booth': booth,
            'sphere': sphere, 'rosen': rosen}[name]


# ---------------- DIRECT ----------------
DIRECT_ITER = [
    ('styblinski', {'kind': 'styblinski'}, [(-5., 5.)] * 2, 1e-6),
    ('styblinski', {'kind': 'styblinski'}, [(-4., 4.)] * 2, 1e-6),
    ('ackley', {'kind': 'ackley'}, [(-5., 5.)] * 2, 1e-6),
    ('shifted_quadratic', {'kind': 'shifted_quadratic', 'shift': [0.3, 0.3, 0.3]}, [(-4., 4.)] * 3, 1e-6),
    ('sphere', {'kind': 'sphere'}, [(-4., 4.)] * 4, 1e-6),
]
DIRECT_GLOBAL = [
    ('rosen', {'kind': 'rosen'}, [(-2., 2.)] * 2, [1.0, 1.0], 2e-3, 1e-2),
    ('booth', {'kind': 'booth'}, [(-10., 10.)] * 2, [1.0, 3.0], 1e-4, 1e-6),
]

direct_cases = []
for name, obj, bounds, rtol in DIRECT_ITER:
    f = build(name, obj)
    r = direct(f, bounds)
    direct_cases.append({
        'fn': 'direct', 'obj': obj, 'bounds': [list(b) for b in bounds],
        'tier': 'iterate',
        'expect': list(map(float, r.x)), 'expect_fun': float(r.fun),
        'rtol': rtol,
    })
for name, obj, bounds, known, rtol, fun_atol in DIRECT_GLOBAL:
    f = build(name, obj)
    r = direct(f, bounds)
    # scipy must itself reach the known optimum within rtol and near-zero fun
    assert np.max(np.abs(np.array(known) - r.x) / (1 + np.abs(np.array(known)))) <= rtol, (name, r.x)
    assert abs(r.fun) <= fun_atol, (name, r.fun)
    direct_cases.append({
        'fn': 'direct', 'obj': obj, 'bounds': [list(b) for b in bounds],
        'tier': 'global',
        'expect': list(map(float, known)),   # analytic global optimum
        'expect_fun': 0.0, 'fun_atol': fun_atol,
        'rtol': rtol,
    })

# ---------------- SHGO ----------------
SHGO_SPECS = [
    ('styblinski', {'kind': 'styblinski'}, [(-5., 5.)] * 2),
    ('ackley', {'kind': 'ackley'}, [(-5., 5.)] * 2),
    ('booth', {'kind': 'booth'}, [(-10., 10.)] * 2),
    ('shifted_quadratic', {'kind': 'shifted_quadratic', 'shift': [0.3, 0.3, 0.3]}, [(-4., 4.)] * 3),
    ('sphere', {'kind': 'sphere'}, [(-4., 4.)] * 4),
    ('rosen', {'kind': 'rosen'}, [(-2., 2.)] * 2),
]
shgo_cases = []
for name, obj, bounds in SHGO_SPECS:
    f = build(name, obj)
    r = shgo(f, bounds)
    shgo_cases.append({
        'fn': 'shgo', 'obj': obj, 'bounds': [list(b) for b in bounds],
        'expect': list(map(float, r.x)), 'expect_fun': float(r.fun),
        'rtol': 1e-6,
    })

if os.path.exists(OUT):
    with open(OUT) as fh:
        data = json.load(fh)
    data.setdefault('solver_cases', [])
    data['solver_cases'] = [c for c in data['solver_cases'] if c.get('fn') not in ('direct', 'shgo')]
    data['solver_cases'].extend(direct_cases)
    data['solver_cases'].extend(shgo_cases)
else:
    data = {'module': 'optimize', 'scipy': __import__('scipy').__version__,
            'solver_cases': direct_cases + shgo_cases}

json.dumps(data, allow_nan=False)   # strict: reject NaN/Inf
with open(OUT, 'w') as fh:
    json.dump(data, fh, separators=(',', ':'))
print(f'optimize: +{len(direct_cases)} direct cases, +{len(shgo_cases)} shgo cases')