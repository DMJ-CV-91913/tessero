#!/usr/bin/env python3
"""scipy.stats.ppcc_plot: Tukey-lambda probability-plot correlation coefficient over a shape grid.

The pure-PHP Tessero\\StatsFns::ppccPlot reproduces scipy.stats.ppcc_plot (dist='tukeylambda'): it
forms the uniform order-statistic medians of the sorted sample, then for each shape in
linspace(a, b, N) computes the Pearson correlation between the Tukey-lambda quantiles and the sorted
sample. It reuses the same helpers as the already-verified ppcc_max, so the recorded scipy
expectation is host-independent (all data is embedded in the fixture).

Writes tests/fixtures/parity/stats_ppccplot.json with a `ppccplot_cases` block:
  {fn:'ppcc_plot', x:[...], a, b, n, expect:{svals:[...], ppcc:[...]}, rtol}
"""
import json
import os

import numpy as np
import scipy
import scipy.stats as st

OUT = os.environ.get('PPCCPLOT_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_ppccplot.json')

cases = []


def add(x, a, b, n=80, rtol=1e-9):
    x = np.asarray(x, float)
    svals, ppcc = st.ppcc_plot(x, a, b, N=n)
    cases.append({
        'fn': 'ppcc_plot',
        'x': [float(v) for v in x],
        'a': float(a),
        'b': float(b),
        'n': int(n),
        'expect': {'svals': [float(v) for v in svals], 'ppcc': [float(v) for v in ppcc]},
        'rtol': rtol,
    })


rng = np.random.default_rng(11)
# Weibull-ish samples (the scipy docstring use-case) over several shape ranges / grid sizes.
add(rng.weibull(2.5, size=60) * 3 + 1, -2, 2, 80)
add(rng.weibull(1.5, size=40) * 2 + 0.5, -3, 3, 50)
add(rng.normal(10, 2, size=100), -2, 2, 80)       # near-normal -> peak around lambda ~ 0.14
add(rng.standard_t(3, size=75) + 5, -5, 5, 61)     # heavy-tailed
add(rng.uniform(0, 1, size=30), -1, 1, 20)         # short-tailed, small N grid
add(rng.exponential(2.0, size=50) + 0.1, 0.0, 1.0, 11)

out = {'module': 'stats', 'scipy': scipy.__version__, 'ppccplot_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'stats_ppccplot: {len(cases)} cases -> {OUT}')
