#!/usr/bin/env python3
r"""scipy.optimize linearmixing / excitingmixing parity fixtures + the
scipy.optimize.fmin_l_bfgs_b wrapper.

linearmixing / excitingmixing are quasi-Newton root solvers driven by the same
inexact-Newton + Armijo engine as broyden1/diagbroyden (Tessero\Optimize\NonlinSolve),
but with the scalar (LinearMixing) / tuned-diagonal (ExcitingMixing) Jacobian
approximations. Both are notoriously fragile (scipy documents that convergence is
strongly problem-dependent), so the covered systems are weakly-coupled trig fixed
points that converge under the stock autoscaled alpha for BOTH methods. The de-risk
(VERIFY_MIX5.py) confirmed scipy success for trig1/trig3; the PHP port reproduces the
iterate sequence and converged root. rtol 1e-6 matches the broyden-family tolerance.

fmin_l_bfgs_b is a thin wrapper over minimize(method='L-BFGS-B'), exactly as the other
fmin_* wrappers (Min::fminLbfgsb forwards to minimize and returns ->x). The de-risk
(VERIFY.py) confirmed its solution equals minimize(method='L-BFGS-B') / scipy's
fmin_l_bfgs_b over 6 problems (bounds / no-bounds, approx_grad True/False) to rtol 1e-5.

Named residual systems mirror makeObjective() 'vecsys_*' kinds; fmin_l_bfgs_b rides the
existing named-problem (makeConProblem) path. Merged into optimize.json 'solver_cases'
(idempotent). Backend-neutral pure PHP, host-independent."""
import json
import math
import os

import numpy as np
from scipy.optimize import root, fmin_l_bfgs_b, minimize, rosen, rosen_der

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'optimize.json')


# ---- linearmixing / excitingmixing: weakly-coupled trig fixed points (converge for both)
def trig1(x):
    return [math.cos(x[0]) + 0.1 * x[1] - 1.0, 0.1 * x[0] + math.sin(x[1]) - 0.5]


def trig3(x):
    return [math.cos(x[0]) + 0.15 * x[1] - 0.9, 0.15 * x[0] + math.sin(x[1]) - 0.4]


MIX_SYSTEMS = {
    'vecsys_mtrig1': (trig1, [0.5, 0.5]),
    'vecsys_mtrig3': (trig3, [0.5, 0.5]),
}
MIX_SOLVERS = ('linearmixing', 'excitingmixing')

mix_cases = []
for fn_name in MIX_SOLVERS:
    for kind, (F, x0) in MIX_SYSTEMS.items():
        sol = root(F, list(x0), method=fn_name)
        assert sol.success, f'{fn_name} {kind} did not converge'
        mix_cases.append({
            'fn': fn_name,
            'obj': {'kind': kind},
            'x0': x0,
            'expect': list(map(float, sol.x)),
            'rtol': 1e-6,
        })

# ---- fmin_l_bfgs_b: named problems resolved in PHP via makeConProblem (reuses existing probs)
def _rosen(x):
    return float(sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2 for i in range(len(x) - 1)))


# problem 'sphere_lo' (bounds, approx_grad), 'rosen2' (bounds+jac), 'quad3' (no-bounds+jac)
lbfgsb_cases = []
# sphere_lo: bounds [(1,None),(2,None)], approx_grad -> no fprime
r = fmin_l_bfgs_b(lambda x: x[0] ** 2 + x[1] ** 2, [3., 3.], approx_grad=1, bounds=[(1, None), (2, None)])
lbfgsb_cases.append({'fn': 'fmin_l_bfgs_b', 'prob': 'sphere_lo', 'approx_grad': True, 'expect': list(map(float, r[0])), 'rtol': 1e-4})
# rosen2: bounds [(-2,2),(-2,2)], analytic grad
r = fmin_l_bfgs_b(_rosen, [-1.2, 1.0], fprime=rosen_der, bounds=[(-2, 2), (-2, 2)])
lbfgsb_cases.append({'fn': 'fmin_l_bfgs_b', 'prob': 'rosen2', 'approx_grad': False, 'expect': list(map(float, r[0])), 'rtol': 1e-4})
# quad3: no bounds, analytic grad
r = fmin_l_bfgs_b(lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2 + (x[2] + 1) ** 2,
                  [0., 0., 0.], fprime=lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 2), 2 * (x[2] + 1)]))
lbfgsb_cases.append({'fn': 'fmin_l_bfgs_b', 'prob': 'quad3', 'approx_grad': False, 'expect': list(map(float, r[0])), 'rtol': 1e-5})

new_cases = mix_cases + lbfgsb_cases
new_fns = set(MIX_SOLVERS) | {'fmin_l_bfgs_b'}

if os.path.exists(OUT):
    with open(OUT) as f:
        data = json.load(f)
    data.setdefault('solver_cases', [])
    data['solver_cases'] = [c for c in data['solver_cases'] if c.get('fn') not in new_fns]
    data['solver_cases'].extend(new_cases)
else:
    data = {'module': 'optimize', 'scipy': __import__('scipy').__version__, 'solver_cases': new_cases}

with open(OUT, 'w') as f:
    json.dump(data, f, separators=(',', ':'))
print(f'optimize: +{len(mix_cases)} mixing + {len(lbfgsb_cases)} fmin_l_bfgs_b cases')
