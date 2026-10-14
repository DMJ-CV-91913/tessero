"""scipy.optimize value classes (Bounds, LinearConstraint, NonlinearConstraint,
RootResults) as optimize.json solver_cases entries. Finite bounds only (JSON has no inf)."""
import numpy as np
import scipy.optimize as O


def cases():
    out = []
    b = O.Bounds([0.0, -1.0], [1.0, 2.0])
    out.append({'fn': 'Bounds', 'ctor': {'lb': [0.0, -1.0], 'ub': [1.0, 2.0]},
                'lb': np.asarray(b.lb, float).tolist(), 'ub': np.asarray(b.ub, float).tolist()})
    bs = O.Bounds(0.0, 5.0)
    lb, ub = np.broadcast_to(bs.lb, (3,)), np.broadcast_to(bs.ub, (3,))
    out.append({'fn': 'Bounds', 'ctor': {'lb': 0.0, 'ub': 5.0}, 'n': 3,
                'lb': lb.astype(float).tolist(), 'ub': ub.astype(float).tolist()})
    lc = O.LinearConstraint([[1.0, 2.0], [0.0, 1.0]], [0.0, -1.0], [1.0, 1.0])
    out.append({'fn': 'LinearConstraint', 'ctor': {'A': [[1.0, 2.0], [0.0, 1.0]], 'lb': [0.0, -1.0], 'ub': [1.0, 1.0]},
                'A': np.asarray(lc.A, float).tolist(), 'lb': np.asarray(lc.lb, float).tolist(), 'ub': np.asarray(lc.ub, float).tolist()})
    nlc = O.NonlinearConstraint(lambda x: x, 0.0, 5.0)
    out.append({'fn': 'NonlinearConstraint', 'ctor': {'lb': 0.0, 'ub': 5.0},
                'lb': float(np.asarray(nlc.lb)), 'ub': float(np.asarray(nlc.ub))})
    rr = O.root_scalar(lambda x: x * x - 2.0, bracket=[0.0, 2.0])
    out.append({'fn': 'RootResults', 'ctor': {'bracket': [0.0, 2.0], 'method': 'brentq'},
                'root': float(rr.root), 'converged': bool(rr.converged), 'flag': str(rr.flag)})
    def _steps(seed, n, k):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(n); g = rng.standard_normal(n); st = []
        for _ in range(k):
            xn = rng.standard_normal(n); gn = rng.standard_normal(n)
            st.append([list(map(float, xn - x)), list(map(float, gn - g))]); x, g = xn, gn
        return st

    for _cls, _name, _seed in [(O.BFGS, 'BFGS', 7), (O.SR1, 'SR1', 13)]:
        for _n, _k in [(3, 4), (4, 5)]:
            _st = _steps(_seed + _n, _n, _k)
            _obj = _cls(); _obj.initialize(_n, 'hess')
            _seq = []
            for _dx, _dg in _st:
                _obj.update(_dx, _dg); _seq.append(_obj.get_matrix().tolist())
            _p = list(np.arange(1.0, _n + 1.0))
            out.append({'fn': _name, 'n': _n, 'steps': _st, 'seq': _seq,
                        'p': _p, 'dot': _obj.dot(np.asarray(_p)).tolist(), 'rtol': 1e-12})
    return out
