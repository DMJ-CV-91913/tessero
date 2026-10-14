import json
import numpy as np
from scipy.integrate import solve_ivp, odeint


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
    if kind == 'robertson':
        return lambda t, y: np.array([
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2,
        ])
    if kind == 'linear3x3':
        M = np.asarray(p['M'], dtype=float)
        return lambda t, y: M.dot(np.asarray(y))
    if kind == 'stiff_linear':
        return lambda t, y: np.array([-1000.0 * y[0] + y[1], -y[1]])
    if kind == 'stiff_linear3':
        A = np.array([[-21.0, 19.0, -20.0], [19.0, -21.0, 20.0], [40.0, -40.0, -40.0]])
        return lambda t, y: A.dot(np.asarray(y))
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
    # --- method='BDF' (variable-order 1-5 implicit backward-differentiation; stiff systems) ---
    case('exp_decay', {'k': 50.0}, (0.0, 1.0), [3.0], np.linspace(0, 1, 20).tolist(), method='BDF'),
    case('robertson', {}, (0.0, 100.0), [1.0, 0.0, 0.0], np.linspace(0, 100, 40).tolist(), method='BDF'),
    case('linear2x2', {'M': [[-1000.0, 1.0], [0.0, -1.0]]}, (0.0, 5.0), [1.0, 1.0], np.linspace(0, 5, 30).tolist(), method='BDF'),
    case('linear3x3', {'M': [[-100.0, 1.0, 0.0], [0.0, -10.0, 1.0], [0.0, 0.0, -1.0]]}, (0.0, 2.0), [1.0, 1.0, 1.0], np.linspace(0, 2, 25).tolist(), method='BDF'),
    case('robertson', {}, (0.0, 40.0), [1.0, 0.0, 0.0], np.linspace(0, 40, 30).tolist(), method='BDF', rtol=1e-8, atol=1e-11),
    case('linear3x3', {'M': [[-100.0, 1.0, 0.0], [0.0, -10.0, 1.0], [0.0, 0.0, -1.0]]}, (0.0, 3.0), [2.0, -1.0, 0.5], np.linspace(0, 3, 25).tolist(), method='BDF', rtol=1e-8, atol=1e-11),
    case('exp_decay', {'k': 100.0}, (0.0, 0.5), [1.0, 2.0, 3.0], np.linspace(0, 0.5, 15).tolist(), method='BDF', rtol=1e-8, atol=1e-11),
    # --- method='Radau' (implicit Radau IIA order 5, stiff ODEs) ---
    case('vanderpol', {'mu': 5.0},   (0.0, 20.0),  [2.0, 0.0],   np.linspace(0, 20, 21).tolist(),   method='Radau'),
    case('vanderpol', {'mu': 50.0},  (0.0, 30.0),  [2.0, 0.0],   np.linspace(0, 30, 16).tolist(),   method='Radau'),
    case('vanderpol', {'mu': 200.0}, (0.0, 40.0),  [2.0, 0.0],   np.linspace(0, 40, 21).tolist(),   method='Radau'),
    case('robertson', {},            (0.0, 100.0), [1.0, 0.0, 0.0], [0.0, 0.4, 4.0, 40.0, 100.0],    method='Radau'),
    case('robertson', {},            (0.0, 1e3),   [1.0, 0.0, 0.0], [0.0, 1.0, 10.0, 100.0, 1000.0], method='Radau', rtol=1e-8, atol=1e-10),
    case('stiff_linear', {},         (0.0, 5.0),   [1.0, 1.0],   np.linspace(0, 5, 11).tolist(),    method='Radau'),
    case('stiff_linear3', {},        (0.0, 2.0),   [1.0, 0.0, -1.0], np.linspace(0, 2, 11).tolist(), method='Radau'),
    case('harmonic', {'w': 2.0},     (0.0, 6.0),   [1.0, 0.0],   np.linspace(0, 6, 13).tolist(),    method='Radau'),
    case('harmonic', {'w': 1.5},     (0.0, -5.0),  [0.0, 1.0],   np.linspace(0, -5, 11).tolist(),   method='Radau'),
    case('exp_decay', {'k': 0.7},    (0.0, 5.0),   [2.0],        np.linspace(0, 5, 11).tolist(),    method='Radau'),
    # --- method='LSODA' (ODEPACK dlsoda: Adams-Moulton <-> BDF automatic stiffness switching) ---
    # scipy's LSODA wraps compiled Fortran whose exact step/order/switch sequence is not reproducible
    # in pure PHP; at rtol<=1e-10 the solver has converged, so Tessero's switching driver (BDF for the
    # detected-stiff regime, DOP853 for the nonstiff regime) matches dlsoda's output within the harness
    # threshold. Mix of nonstiff (exp_decay/harmonic/lotka/logistic) and stiff (robertson/vanderpol).
    case('exp_decay', {'k': 0.7},    (0.0, 5.0),   [2.0],           np.linspace(0, 5, 11).tolist(),   method='LSODA', rtol=1e-10, atol=1e-12),
    case('harmonic',  {'w': 2.0},    (0.0, 6.0),   [1.0, 0.0],      np.linspace(0, 6, 13).tolist(),   method='LSODA', rtol=1e-10, atol=1e-12),
    case('lotka', {'a': 1.5, 'b': 1.0, 'c': 3.0, 'd': 1.0}, (0.0, 10.0), [10.0, 5.0], np.linspace(0, 10, 21).tolist(), method='LSODA', rtol=1e-10, atol=1e-12),
    case('logistic',  {'r': 1.0, 'K': 10.0}, (0.0, 6.0), [1.0],     np.linspace(0, 6, 13).tolist(),   method='LSODA', rtol=1e-10, atol=1e-12),
    case('harmonic',  {'w': 1.5},    (0.0, -5.0),  [0.0, 1.0],      np.linspace(0, -5, 11).tolist(),  method='LSODA', rtol=1e-10, atol=1e-12),
    case('robertson', {},            (0.0, 100.0), [1.0, 0.0, 0.0], [0.0, 0.4, 4.0, 40.0, 100.0],     method='LSODA', rtol=1e-10, atol=1e-12),
    case('vanderpol', {'mu': 5.0},   (0.0, 20.0),  [2.0, 0.0],      np.linspace(0, 20, 21).tolist(),  method='LSODA', rtol=1e-10, atol=1e-12),
    case('vanderpol', {'mu': 50.0},  (0.0, 30.0),  [2.0, 0.0],      np.linspace(0, 30, 16).tolist(),  method='LSODA', rtol=1e-10, atol=1e-12),
    case('exp_decay', {'k': 50.0},   (0.0, 1.0),   [3.0],           np.linspace(0, 1, 20).tolist(),   method='LSODA', rtol=1e-10, atol=1e-12),
]


# --- scipy.integrate.odeint (legacy interface over the SAME ODEPACK dlsoda engine) ---
# odeint(func, y0, t, tfirst=False) returns y at every t (shape len(t) x n). We mirror LSODA's
# matchable-tolerance regime (rtol=atol=1e-10..1e-12). The runner reconstructs func via makeOdeSystem
# (func is called as func(y, t) by default; our PHP facade supports tfirst for the func(t, y) order).
def odeint_case(kind, params, y0, t, rtol=1e-10, atol=1e-12, tfirst=False):
    f = make_rhs(kind, params)
    tv = np.asarray(t, dtype=float)
    if tfirst:
        y = odeint(f, y0, tv, rtol=rtol, atol=atol, tfirst=True)
    else:
        y = odeint(lambda yy, tt: f(tt, yy), y0, tv, rtol=rtol, atol=atol)
    return {
        'kind': kind,
        'params': params,
        'y0': [float(v) for v in y0],
        't': tv.tolist(),
        'tfirst': tfirst,
        'rtol': rtol,
        'atol': atol,
        'expect': y.tolist(),   # shape (len(t), n), matching scipy odeint orientation
    }


odeint_cases = [
    odeint_case('exp_decay', {'k': 0.7},    [2.0],           np.linspace(0, 5, 11).tolist()),
    odeint_case('harmonic',  {'w': 2.0},    [1.0, 0.0],      np.linspace(0, 6, 13).tolist()),
    odeint_case('lotka', {'a': 1.5, 'b': 1.0, 'c': 3.0, 'd': 1.0}, [10.0, 5.0], np.linspace(0, 10, 21).tolist()),
    odeint_case('logistic',  {'r': 1.0, 'K': 10.0}, [1.0],   np.linspace(0, 6, 13).tolist()),
    odeint_case('robertson', {},            [1.0, 0.0, 0.0], [0.0, 0.4, 4.0, 40.0, 100.0]),
    odeint_case('vanderpol', {'mu': 50.0},  [2.0, 0.0],      np.linspace(0, 30, 16).tolist()),
    odeint_case('exp_decay', {'k': 2.0},    [5.0],           np.linspace(0, 4, 9).tolist(), tfirst=True),
]


out = {'module': 'integrate', 'scipy': __import__('scipy').__version__, 'solveivp_cases': cases}
with open(OUT, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'))
print('integrate_solveivp: %d solve_ivp cases' % len(cases))

OUT_ODEINT = '../../tests/fixtures/parity/integrate_odeint.json'
out2 = {'module': 'integrate', 'scipy': __import__('scipy').__version__, 'odeint_cases': odeint_cases}
with open(OUT_ODEINT, 'w') as fh:
    json.dump(out2, fh, separators=(',', ':'))
print('integrate_odeint: %d odeint cases' % len(odeint_cases))
