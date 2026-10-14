import json
import numpy as np
from scipy.integrate import solve_bvp


# Named BVP systems. The PHP side rebuilds fun/bc from `kind` (+ params); a Python
# callable can't cross into PHP, so fixtures carry NAMED systems only.
def make_bvp(kind, params):
    if kind == 'bratu':
        def fun(x, y):
            return np.vstack((y[1], -np.exp(y[0])))
        def bc(ya, yb):
            return np.array([ya[0], yb[0]])
        return fun, bc, None
    if kind == 'linear_sin':   # y'' = -y, y(0)=0, y(b)=1
        def fun(x, y):
            return np.vstack((y[1], -y[0]))
        def bc(ya, yb):
            return np.array([ya[0], yb[0] - 1.0])
        return fun, bc, None
    if kind == 'nonlin_reaction':  # y'' = -3 y y', y(0)=0, y(1)=1
        def fun(x, y):
            return np.vstack((y[1], -3 * y[0] * y[1]))
        def bc(ya, yb):
            return np.array([ya[0], yb[0] - 1.0])
        return fun, bc, None
    if kind == 'sturm_liouville':  # y'' = -k^2 y, y(0)=y(1)=0, y'(0)=k ; unknown p=[k]
        def fun(x, y, p):
            k = p[0]
            return np.vstack((y[1], -k ** 2 * y[0]))
        def bc(ya, yb, p):
            k = p[0]
            return np.array([ya[0], yb[0], ya[1] - k])
        return fun, bc, 'p'
    if kind == 'exp_shrink':   # y'' = y, y(0)=1, y(1)=exp(1)  -> y=exp(x)
        def fun(x, y):
            return np.vstack((y[1], y[0]))
        def bc(ya, yb):
            return np.array([ya[0] - 1.0, yb[0] - np.e])
        return fun, bc, None
    raise ValueError(kind)


OUT = '../../tests/fixtures/parity/integrate_solvebvp.json'


def case(kind, params, x, y_init, p_init=None, tol=1e-3, bc_tol=None, x_eval=None):
    fun, bc, mode = make_bvp(kind, params)
    x = np.asarray(x, dtype=float)
    y_init = np.asarray(y_init, dtype=float)
    if mode == 'p':
        res = solve_bvp(fun, bc, x, y_init, p=p_init, tol=tol, bc_tol=bc_tol)
    else:
        res = solve_bvp(fun, bc, x, y_init, tol=tol, bc_tol=bc_tol)
    assert res.status == 0, (kind, res.status, res.message)
    if x_eval is None:
        x_eval = np.linspace(float(x[0]), float(x[-1]), 50)
    x_eval = np.asarray(x_eval, dtype=float)
    ys = res.sol(x_eval)
    return {
        'kind': kind,
        'params': params,
        'x': x.tolist(),
        'y_init': y_init.tolist(),
        'p_init': (list(map(float, p_init)) if p_init is not None else None),
        'tol': tol,
        'bc_tol': bc_tol,
        'x_eval': x_eval.tolist(),
        'expect': ys.tolist(),
        'p_expect': (list(map(float, res.p)) if res.p is not None else None),
        'status': int(res.status),
    }


cases = [
    # Bratu, small-solution branch (zero guess)
    case('bratu', {}, np.linspace(0, 1, 5), np.zeros((2, 5))),
    # Bratu, large-solution branch (y0 guess = 3)
    case('bratu', {}, np.linspace(0, 1, 5), [[3.0]*5, [0.0]*5]),
    # Bratu tighter tol
    case('bratu', {}, np.linspace(0, 1, 9), np.zeros((2, 9)), tol=1e-6),
    # linear y'' = -y
    case('linear_sin', {}, np.linspace(0, np.pi/2, 5),
         [np.linspace(0, 1, 5).tolist(), [1.0]*5], tol=1e-6),
    # linear tighter tol
    case('linear_sin', {}, np.linspace(0, np.pi/2, 7),
         [np.sin(np.linspace(0, np.pi/2, 7)).tolist(), np.cos(np.linspace(0, np.pi/2, 7)).tolist()], tol=1e-8),
    # nonlinear reaction
    case('nonlin_reaction', {}, np.linspace(0, 1, 6),
         [np.linspace(0, 1, 6).tolist(), [1.0]*6], tol=1e-4),
    # exp growth two-point
    case('exp_shrink', {}, np.linspace(0, 1, 5),
         [[1.0]*5, [1.0]*5], tol=1e-6),
    # Sturm-Liouville eigenvalue with unknown p (k ~ 2*pi)
    case('sturm_liouville', {}, np.linspace(0, 1, 5),
         [[0.0, 1.0, 0.0, -1.0, 0.0], [0.0]*5], p_init=[6.0], tol=1e-3),
    # Sturm-Liouville tighter
    case('sturm_liouville', {}, np.linspace(0, 1, 7),
         [[0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0]*7], p_init=[6.0], tol=1e-6),
]

out = {'module': 'integrate', 'scipy': __import__('scipy').__version__, 'solvebvp_cases': cases}
with open(OUT, 'w') as fh:
    json.dump(out, fh, separators=(',', ':'))
print('integrate_solvebvp: %d solve_bvp cases' % len(cases))
