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
    return out
