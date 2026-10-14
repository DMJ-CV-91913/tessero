#!/usr/bin/env python3
"""SciPy reference fixtures for scipy.stats.dunnett (pure-PHP Tessero\\StatsDunnett).

    python3 tools/parity/fixtures_dunnett.py

Writes tests/fixtures/parity/stats_dunnett.json with a `dunnett_cases` list. Each case
records the experimental-group samples, the control sample, the alternative, and the PCG64
seed (scipy's `rng`); `expect` is {statistic, pvalue} (both 1-D lists) from
scipy.stats.dunnett. The run-tests `dunnett_cases` arm checks both backends.

Exactness note
--------------
dunnett's `.statistic` is deterministic and reproduces to full double precision.
dunnett's `.pvalue` is scipy's randomized-QMC estimate of the multivariate-t CDF; given the
same `rng` seed it is reproducible, and Tessero\\StatsDunnett matches it bit-for-bit EXCEPT
at a Fast-CBC-lattice `argmin` tie whose outcome is decided by pocketfft's exact rounding
(scipy.fft vs numpy.fft even disagree there). For a subset of group counts the minimiser is a
numerical tie and the p-value is not portable to full precision. These fixtures therefore
exercise `.pvalue` only on the TIE-FREE group counts (1, 2, 4, 5, 6, 7, 9 -- verified against
scipy.fft), where Tessero matches scipy to ~1e-15. `.statistic` is checked for every case.
"""
import json
import os

import numpy as np
import scipy
import scipy.stats as st

import fixture_env

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'fixtures', 'parity')
ALTS = ['two-sided', 'less', 'greater']

# Group counts whose CBC lattice construction is bit-exact vs scipy.fft (no argmin tie).
# Verified by comparing a tie-free direct-convolution lattice against scipy._qmvnt._cbc_lattice
# at n_qmc = 100*ngroups for ngroups in 1..12: ties occur at 3, 8, 10, 11; the rest are exact.
TIE_FREE = [1, 2, 4, 5, 6, 7, 9]


def _row(v):
    return [float(x) for x in np.atleast_1d(v)]


def main():
    cases = []
    rng = np.random.default_rng(20251014)

    def add(samples, control, alt, seed):
        res = st.dunnett(*samples, control=control, alternative=alt,
                         rng=np.random.default_rng(seed))
        cases.append({
            'fn': 'dunnett',
            'samples': [[float(x) for x in s] for s in samples],
            'control': [float(x) for x in control],
            'alternative': alt,
            'seed': int(seed),
            'expect': {'statistic': _row(res.statistic), 'pvalue': _row(res.pvalue)},
            'rtol': 1e-9,
            'atol': 1e-11,
        })

    # The canonical scipy docstring example (Thomson & Short 1969, Table 1), all alternatives.
    samples0 = [[3.8, 2.7, 4.0, 2.4], [2.8, 3.4, 3.7, 2.2, 2.0]]
    control0 = [2.9, 3.0, 2.5, 2.6, 3.2]
    add(samples0, control0, 'two-sided', 12345)
    add(samples0, control0, 'greater', 777)
    add(samples0, control0, 'less', 2024)

    # A single experimental group (reduces to a univariate-t comparison).
    add([[5.1, 5.3, 5.0, 5.4, 5.2]], [4.9, 5.0, 5.1, 4.8], 'two-sided', 42)
    add([[5.1, 5.3, 5.0, 5.4, 5.2]], [4.9, 5.0, 5.1, 4.8], 'greater', 43)

    # A spread over the tie-free group counts, all alternatives, varied seeds and sample sizes.
    for ng in TIE_FREE:
        for alt in ALTS:
            seed = int(rng.integers(1, 10 ** 7))
            samples = [list(rng.normal(float(rng.normal()), 1.0,
                                       size=int(3 + rng.integers(0, 6))))
                       for _ in range(ng)]
            control = list(rng.normal(0.0, 1.0, size=int(4 + rng.integers(0, 6))))
            add(samples, control, alt, seed)

    out = {
        'module': 'stats',
        'kind': 'functions',
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'env': fixture_env.env(),
        'dunnett_cases': cases,
    }
    path = os.environ.get('FIXTURE_OUT') or os.path.join(OUTDIR, 'stats_dunnett.json')
    fixture_env.require_pinned()
    with open(path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'stats_dunnett: {len(cases)} cases -> {path}')


if __name__ == '__main__':
    main()
