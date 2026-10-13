#!/usr/bin/env python3
"""SciPy reference fixtures for scipy.stats.boschloo_exact (pure-PHP Tessero\\Boschloo).

    python3 tools/parity/fixtures_boschloo.py

Writes tests/fixtures/parity/stats_boschloo.json. Each case records a 2x2 table, the
alternative, and n; `expect` is {statistic, pvalue} from scipy.stats.boschloo_exact. The
run-tests `boschloo_cases` arm checks both backends against these values.

Format mirrors the other stats fixtures (module/kind/numpy/scipy/env header); the payload key
`boschloo_cases` is dispatched by tools/parity/run-tests.php.
"""
import json
import os

import numpy as np
import scipy
import scipy.stats as st

import fixture_env

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'fixtures', 'parity')
ALTS = ['two-sided', 'less', 'greater']

# A spread of 2x2 tables: balanced, lopsided, small counts, a zero cell, and a larger table.
TABLES = [
    [[74, 31], [43, 32]],   # the scipy docstring example
    [[7, 12], [8, 3]],
    [[2, 7], [8, 2]],
    [[20, 14], [12, 22]],
    [[5, 0], [1, 4]],
    [[8, 2], [1, 5]],
    [[3, 4], [5, 9]],
    [[1, 9], [11, 3]],
    [[10, 10], [10, 10]],
    [[6, 3], [1, 10]],
    [[12, 5], [3, 20]],
    [[0, 5], [5, 0]],
]


def _num(x):
    """JSON-safe float (NaN -> the string 'nan', matching how PHP serialises NAN expectations)."""
    x = float(x)
    return 'nan' if np.isnan(x) else x


def main():
    cases = []
    for t in TABLES:
        for alt in ALTS:
            res = st.boschloo_exact(np.asarray(t, dtype=np.int64), alternative=alt)
            cases.append({
                'fn': 'boschloo_exact',
                'table': t,
                'alternative': alt,
                'n': 32,
                'expect': {'statistic': _num(res.statistic), 'pvalue': _num(res.pvalue)},
                'rtol': 1e-9,
                'atol': 1e-11,
            })
    # a degenerate column (both column-0 cells zero) -> scipy returns (nan, nan)
    for alt in ALTS:
        res = st.boschloo_exact(np.array([[0, 3], [0, 4]], dtype=np.int64), alternative=alt)
        cases.append({
            'fn': 'boschloo_exact',
            'table': [[0, 3], [0, 4]],
            'alternative': alt,
            'n': 32,
            'expect': {'statistic': _num(res.statistic), 'pvalue': _num(res.pvalue)},
            'rtol': 1e-9,
            'atol': 1e-11,
        })

    out = {
        'module': 'stats',
        'kind': 'functions',
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'env': fixture_env.env(),
        'boschloo_cases': cases,
    }
    path = os.environ.get('FIXTURE_OUT') or os.path.join(OUTDIR, 'stats_boschloo.json')
    fixture_env.require_pinned()
    with open(path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'stats_boschloo: {len(cases)} cases -> {path}')


if __name__ == '__main__':
    main()
