#!/usr/bin/env python3
r"""scipy.stats matrix-variate / normal-inverse-gamma density fixtures: the pure-PHP facades
Tessero\Stats\MatrixT and Tessero\Stats\NormalInverseGamma, verified against the deterministic
density surface of scipy.stats.matrix_t and scipy.stats.normal_inverse_gamma.

Only the closed-form, RNG-free surface is covered (matrix_t: pdf/logpdf at a single matrix X;
normal_inverse_gamma: pdf/logpdf at (x, s2) plus mean/var); .rvs is RNG-only and out of scope.

Writes one JSON into tests/fixtures/parity/, read by run-tests.php's matvardist_cases branch:
  stats_matvardist.json

Each matrix_t case carries the 2-D mean, row-spread, col-spread, scalar df, the evaluation point X,
the method and the scipy-produced expected scalar. Each normal_inverse_gamma case carries the four
shape parameters, the method, its (x, s2) argument where relevant, and the expected value(s). Matched
to rtol 1e-10 (stored exact so the gate re-derives the tolerance).
"""
import json
import os

import numpy as np
import scipy
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIXDIR = os.path.join(ROOT, 'tests', 'fixtures', 'parity')
SCIPY = scipy.__version__


def v(x):
    f = float(np.asarray(x).item())
    if np.isposinf(f):
        return 'inf'
    if np.isneginf(f):
        return '-inf'
    if np.isnan(f):
        return 'nan'
    return f


def main():
    rr = np.random.default_rng(20261014)
    cases = []

    # ---- scipy.stats.matrix_t: pdf / logpdf at a single matrix X ----
    for (m, n) in ((2, 2), (3, 2), (2, 3), (3, 3), (4, 2), (2, 4)):
        M = rr.normal(size=(m, n))
        Ar = rr.normal(size=(m, m)); R = Ar @ Ar.T + m * np.eye(m)
        Ac = rr.normal(size=(n, n)); C = Ac @ Ac.T + n * np.eye(n)
        df = float(rr.uniform(n + 1, n + 6))
        X = rr.normal(size=(m, n))
        d = st.matrix_t(M, R, C, df)
        common = {'kind': 'matrix_t', 'mean': M.tolist(), 'row': R.tolist(),
                  'col': C.tolist(), 'df': df, 'X': X.tolist()}
        cases.append({**common, 'method': 'logpdf', 'expect': v(d.logpdf(X))})
        cases.append({**common, 'method': 'pdf', 'expect': v(d.pdf(X))})

    # ---- scipy.stats.normal_inverse_gamma: pdf / logpdf at (x, s2) + mean/var ----
    for _ in range(6):
        mu = float(rr.normal())
        lmbda = float(rr.uniform(0.5, 5))
        a = float(rr.uniform(2.5, 6))      # a > 2 so mean and var are both finite
        b = float(rr.uniform(0.5, 5))
        x = float(rr.normal())
        s2 = float(rr.uniform(0.1, 5))
        d = st.normal_inverse_gamma(mu, lmbda, a, b)
        base = {'kind': 'normal_inverse_gamma', 'mu': mu, 'lmbda': lmbda, 'a': a, 'b': b}
        cases.append({**base, 'method': 'logpdf', 'x': x, 's2': s2, 'expect': v(d.logpdf(x, s2))})
        cases.append({**base, 'method': 'pdf', 'x': x, 's2': s2, 'expect': v(d.pdf(x, s2))})
        mx, ms2 = d.mean()
        cases.append({**base, 'method': 'mean', 'expect': [v(mx), v(ms2)]})
        vx, vs2 = d.var()
        cases.append({**base, 'method': 'var', 'expect': [v(vx), v(vs2)]})

    out = {'module': 'stats', 'scipy': SCIPY, 'matvardist_cases': cases}
    path = os.path.join(FIXDIR, 'stats_matvardist.json')
    with open(path, 'w') as f:
        json.dump(out, f, separators=(',', ':'), allow_nan=False)
    print(f'stats_matvardist.json: {len(cases)} cases -> {path}')


if __name__ == '__main__':
    main()
