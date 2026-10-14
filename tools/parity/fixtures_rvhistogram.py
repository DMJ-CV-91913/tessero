#!/usr/bin/env python3
"""scipy.stats.rv_histogram: a continuous distribution from a (counts, bin_edges) histogram.

The pure-PHP Tessero\\Stats\\RvHistogram reproduces scipy.stats.rv_histogram: it normalises the
histogram to a piecewise-constant pdf, forms the piecewise-linear cdf over the bin edges, and serves
pdf/cdf/sf/ppf plus the closed-form mean/var/median of the step density. All data is embedded in the
fixture, so the recorded scipy expectation is host-independent (no pinned environment required).

Writes tests/fixtures/parity/stats_rvhistogram.json with a `rvhistogram_cases` block:
  {fn:'rv_histogram', counts:[...], bin_edges:[...], density:bool, method:'pdf'|'cdf'|'sf'|'ppf'|
   'mean'|'var'|'median'|'std', arg:[...] (omitted for scalar stats), expect:[...] (list for the
   array methods, a single-element list for the scalar stats), rtol}
"""
import json
import os

import numpy as np
import scipy
import scipy.stats as st

OUT = os.environ.get('RVHISTOGRAM_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'stats_rvhistogram.json')

cases = []


def add(counts, edges, density, method, arg=None, rtol=1e-9):
    d = st.rv_histogram((np.asarray(counts, float), np.asarray(edges, float)), density=density)
    case = {
        'fn': 'rv_histogram',
        'counts': [float(v) for v in counts],
        'bin_edges': [float(v) for v in edges],
        'density': bool(density),
        'method': method,
        'rtol': rtol,
    }
    if method in ('pdf', 'cdf', 'sf', 'ppf'):
        x = np.asarray(arg, float)
        got = getattr(d, method)(x)
        case['arg'] = [float(v) for v in x]
        case['expect'] = [float(v) for v in np.atleast_1d(got)]
    else:  # scalar statistic
        got = getattr(d, method)()
        case['expect'] = [float(got)]
    cases.append(case)


def sweep(counts, edges, density):
    lo, hi = float(edges[0]), float(edges[-1])
    pad = (hi - lo) * 0.1 + 1.0
    xs = list(np.linspace(lo - pad, hi + pad, 23)) + list(map(float, edges))
    qs = list(np.linspace(0.0, 1.0, 21)) + [0.001, 0.123, 0.5, 0.999]
    add(counts, edges, density, 'pdf', xs)
    add(counts, edges, density, 'cdf', xs)
    add(counts, edges, density, 'sf', xs)
    add(counts, edges, density, 'ppf', qs)
    for m in ('mean', 'var', 'std', 'median'):
        add(counts, edges, density, m)


# Case set 1: uniform-ish counts, constant bins, density=False (the np.histogram default use).
data1 = np.array([0., 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 6, 7, 8, 9, 9, 10, 10.])
c1, e1 = np.histogram(data1, bins=5)
sweep(c1.tolist(), e1.tolist(), False)

# Case set 2: skewed counts, constant bins, density=False.
sweep([10., 6, 3, 1, 0, 2], np.linspace(-3, 3, 7).tolist(), False)

# Case set 3: negative + nonuniform bin widths, density=False.
sweep([3., 5, 2, 8, 1], [-2., -1, 1, 2, 5, 6], False)

# Case set 4: nonuniform widths, density=True (counts already a density).
sweep([0.1, 0.4, 0.05, 0.3, 0.02], [0., 1, 2, 4, 7, 10], True)

# Case set 5: a zero-count interior bin -> flat cdf segment (exercises the np.interp tie path).
sweep([4., 0, 3, 0, 5], [0., 2, 4, 6, 8, 10], False)

# Case set 6: single bin.
sweep([7.], [1., 4.], False)

# Case set 7: many fine bins from a deterministic sample.
rng = np.random.default_rng(20261014)
sample = rng.normal(5.0, 2.0, size=500)
c7, e7 = np.histogram(sample, bins=20)
sweep(c7.tolist(), e7.tolist(), False)

out = {'module': 'stats', 'scipy': scipy.__version__, 'rvhistogram_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'stats_rvhistogram: {len(cases)} cases -> {OUT}')
