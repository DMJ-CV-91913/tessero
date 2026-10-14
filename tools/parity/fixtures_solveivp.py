import json
import numpy as np
from scipy.integrate import solve_ivp


def make_rhs(kind, p):
    if kind == 'exp_decay':
        k = p['k']
        return lambda t, y: -k * np.asarray(y)
    if kind == 'harmonic':
        w = p['w']
        return lambda t, y: np.array([y[1], -w * w * y[0]])
    if kind == 'lotka':
        a, b, c, d = p['a'], p['b'], p['c'], p['d']
        return lambda t, y: np.array([a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]])
    if kind == 'linear2x2':
        M = np.asarray(p['M'], dtype=float)
        return lambda t, y: M.dot(np.asarray(y))
    if kind == 'logistic':
        r, K = p['r'], p['K']
        return lambda t, y: r * np.asarray(y) * (1.0 - np.asarray(y) / K)
    if kind == 'vanderpol':
        mu = p['mu']
        return lambda t, y: np.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]])
    raise ValueError(kind)


OUT = '../../tests/fixtures/parity/integrate_solveivp.json'


def case(kind, params, t_span, y0, t_eval, method='RK45', rtol=1e-6, atol=1e-9):
    f = make_rhs(kind, params)
    teval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(f, t_span, y0, method=method, t_eval=teval, rtol=rtol, atol=atol)
    assert sol.success, (kind, method, sol.message)
    return {
        'kind': kind,
        'params': params,
        't_span': [float(t_span[0]), float(t_span[1])],
        'y0': [float(v) for v in y0],
        't_eval': teval.tolist(),
        'method': method,
        'rtol': rtol,
        'atol': atol,
        'expect': sol.y.tolist(),
    }


cases = [
    case('exp_decay', {'k': 0.7}, (0.0, 5.0), [2.0], np.linspace(0, 5, 11).tolist()),
    case('harmonic', {'w': 2.0}, (0.0, 6.0), [1.0, 0.0], np.linspace(0, 6, 13).tolist()),
    case('lotka', {'a': 1.5, 'b': 1.0, 'c': 3.0, 'd': 1.0}, (0.0, 10.0), [10.0, 5.0], np.linspace(0, 10, 21).tolist()),
    case('linear2x2', {'M': [[-0.5, 1.0], [-1.0, -0.5]]}, (0.0, 8.0), [1.0, 1.0], np.linspace(0, 8, 17).tolist()),
    case('logistic', {'r': 1.0, 'K': 10.0}, (0.0, 6.0), [1.0], np.linspace(0, 6, 13).tolist()),
    case('vanderpol', {'mu': 0.5}, (0.0, 10.0), [2.0, 0.0], np.linspace(0, 10, 21).tolist()),
    case('exp_decay', {'k': 2.0}, (0.0, 4.0), [5.0], np.linspace(0, 4, 9).tolist(), rtol=1e-8, atol=1e-11),
    case('harmonic', {'w': 1.5}, (0.0, -5.0), [0.0, 1.0], np.linspace(0, -5, 11).tolist()),
    case('harmonic', {'w': 2.0}, (0.0, 6.0), [1.0, 0.0], np.linspace(0, 6, 13).tolist(), method='RK23'),
    case('lotka', {'a': 1.5, 'b': 1.0, 'c': 3.0, 'd': 1.0}, (0.0, 10.0), [10.0, 5.0], np.linspace(0, 10, 21).tolist(), method='RK23'),
    case('exp_decay', {'k': 0.7}, (0.0, 5.0), [2.0], np.linspace(0, 5, 11).tolist(), method='RK23'),
    case('logistic', {'r': 1.0, 'K': 10.0}, (0.0, 6.0), [1.0], np.linspace(0, 6, 13).tolist(), method='RK23'),
    # --- method='DOP853' (Dormand-Prince 8(5,3)); mirror the RK45 cases + a backward-time case, rtol 1e-8 ---
    case('exp_decay', {'k': 0.7}, (0.0, 5.0), [2.0], np.linspace(0, 5, 11).tolist(), method='DOP853', rtol=1e-8, atol=1e-11),
    case('harmonic', {'w': 2.0}, (0.0, 6.0), [1.0, 0.0], np.linspace(0, 6, 13).tolist(), method='DOP853', rtol=1e-8, atol=1e-11),
    case('lotka', {'a': 1.5, 'b': 1.0, 'c': 3.0, 'd': 1.0}, (0.0, 10.0), [10.0, 5.0], np.linspace(0, 10, 21).tolist(), method='DOP853', rtol=1e-8, atol=1e-11),
    case('linear2x2', {'M': [[-0.5, 1.0], [-1.0, -0.5]]}, (0.0, 8.0), [1.0, 1.0], np.linspace(0, 8, 17).tolist(), method='DOP853', rtol=1e-8, atol=1e-11),
    case('logistic', {'r': 1.0, 'K': 10.0}, (0.0, 6.0), [1.0], np.linspace(0, 6, 13).tolist(), method='DOP853', rtol=1e-8, atol=1e-11),
    case('vanderpol', {'mu': 0.5}, (0.0, 10.0), [2.0, 0.0], np.linspace(0, 10, 21).tolist(), method='DOP853', rtol=1e-8, atol=1e-11),
    case('harmonic', {'w': 1.5}, (0.0, -5.0), [0.0, 1.0], np.linspace(0, -5, 11).tolist(), method='DOP853', rtol=1e-8, atol=1e-11),
]


out = {'module': 'integrate', 'scipy': __import__('scipy').__version__, 'solveivp_cases': cases}
with open(OUT, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'))
print('integrate_solveivp: %d solve_ivp cases' % len(cases))
