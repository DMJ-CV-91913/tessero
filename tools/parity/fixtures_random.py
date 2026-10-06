#!/usr/bin/env python3
"""Generate NumPy reference fixtures for the numpy.random.Generator methods in the registry (random.*).

    python3 tools/parity/fixtures_random.py

Writes tests/fixtures/parity/random.json in the call format of fixtures_np.py, with a `seed` per case: the
runner builds Generator::defaultRng(seed), replays the recorded calls on it in order (so the stream position
carries over between calls, as in NumPy) and compares each result. Parameter errors must raise.
"""
import json, os

import numpy as np
import fixture_env

import fixtures_np as F

OUT = os.path.join(F.ROOT, 'tests', 'fixtures', 'parity', 'random.json')

# method -> list of (args, kwargs); each list is replayed on one generator per seed
CASES = {
    'random': [([], {}), ([], {'size': 5}), ([], {'size': (2, 3)}), ([], {'size': 4, 'dtype': np.float32})],
    'standard_normal': [([], {}), ([], {'size': 7}), ([], {'size': (3, 2), 'dtype': np.float32})],
    'standard_exponential': [([], {'size': 6}), ([], {'size': 4, 'method': 'inv'}), ([], {'size': 3, 'dtype': np.float32}),
                             ([], {'size': 3, 'dtype': np.float32, 'method': 'inv'})],
    'standard_gamma': [([0.5], {'size': 5}), ([1.0], {'size': 3}), ([3.7], {'size': 4}), ([np.array([0.2, 1.0, 9.0])], {}),
                       ([2.0], {'size': 3, 'dtype': np.float32}), ([0.0], {'size': 2}), ([-1.0], {'size': 2})],
    'gamma': [([2.0], {'size': 5}), ([np.array([0.5, 2.0]), 3.0], {}), ([1.5, np.array([[1.0], [2.0]])], {'size': (2, 3)}),
              ([-1.0], {})],
    'beta': [([0.5, 0.5], {'size': 5}), ([2.0, 3.0], {'size': 4}), ([np.array([0.1, 1.0, 5.0]), 2.0], {}), ([0.0, 1.0], {})],
    'exponential': [([], {'size': 4}), ([2.5], {'size': 3}), ([np.array([1.0, 10.0])], {}), ([-1.0], {})],
    'uniform': [([], {'size': 5}), ([-2.0, 3.0], {'size': 4}), ([np.array([0.0, 10.0]), 20.0], {}),
                ([1.0, np.array([[2.0], [3.0]])], {'size': (2, 2)}), ([2.0, 1.0], {}), ([0.0, np.inf], {})],
    'normal': [([], {'size': 5}), ([10.0, 2.0], {'size': 3}), ([np.array([0.0, 100.0]), np.array([1.0, 0.1])], {}),
               ([0.0, -1.0], {})],
    'f': [([2.0, 5.0], {'size': 5}), ([np.array([1.0, 30.0]), 10.0], {}), ([0.0, 1.0], {})],
    'noncentral_f': [([3.0, 20.0, 3.0], {'size': 5}), ([2.0, 5.0, 0.0], {'size': 2}), ([1.0, 1.0, -1.0], {})],
    'chisquare': [([2.0], {'size': 5}), ([np.array([0.5, 1.0, 50.0])], {}), ([0.0], {})],
    'noncentral_chisquare': [([3.0, 2.0], {'size': 5}), ([0.5, 1.0], {'size': 3}), ([3.0, 0.0], {'size': 2}),
                             ([1.0, -1.0], {})],
    'standard_cauchy': [([], {'size': 5})],
    'standard_t': [([3.0], {'size': 5}), ([np.array([1.0, 1e6])], {}), ([0.0], {})],
    'vonmises': [([0.0, 4.0], {'size': 5}), ([1.0, 0.0], {'size': 3}), ([np.array([-1.0, 1.0]), 1e-9], {}),
                 ([0.0, 1e7], {'size': 2}), ([0.0, -1.0], {})],
    'pareto': [([3.0], {'size': 5}), ([0.0], {})],
    'weibull': [([1.5], {'size': 5}), ([0.0], {'size': 2}), ([-1.0], {})],
    'power': [([5.0], {'size': 5}), ([0.0], {})],
    'laplace': [([], {'size': 5}), ([1.0, 2.0], {'size': 3}), ([0.0, -1.0], {})],
    'gumbel': [([], {'size': 5}), ([1.0, 2.0], {'size': 3})],
    'logistic': [([], {'size': 5}), ([1.0, 2.0], {'size': 3})],
    'lognormal': [([], {'size': 5}), ([1.0, 0.5], {'size': 3}), ([0.0, -1.0], {})],
    'rayleigh': [([], {'size': 5}), ([3.0], {'size': 3}), ([-1.0], {})],
    'wald': [([1.0, 3.0], {'size': 5}), ([np.array([0.5, 2.0]), 1.0], {}), ([0.0, 1.0], {})],
    'triangular': [([0.0, 0.5, 1.0], {'size': 5}), ([-1.0, -1.0, 2.0], {'size': 3}), ([np.array([0.0, 1.0]), 1.0, 3.0], {}),
                   ([1.0, 0.0, 2.0], {}), ([0.0, 0.0, 0.0], {})],
    'binomial': [([10, 0.3], {'size': 5}), ([100, 0.7], {'size': 3}), ([1000, 0.2], {'size': 4}), ([np.array([5, 50]), 0.5], {}),
                 ([0, 0.5], {'size': 2}), ([10, 1.5], {})],
    'negative_binomial': [([5.0, 0.5], {'size': 5}), ([0.5, 0.2], {'size': 3}), ([10.0, 0.0], {})],
    'poisson': [([], {'size': 5}), ([3.5], {'size': 4}), ([100.0], {'size': 3}), ([np.array([0.0, 20.0])], {}), ([-1.0], {})],
    'zipf': [([2.0], {'size': 5}), ([1.0], {})],
    'geometric': [([0.3], {'size': 5}), ([1.0], {'size': 2}), ([0.0], {})],
    'hypergeometric': [([10, 5, 7], {'size': 5}), ([100, 200, 50], {'size': 3}), ([5, 5, 11], {})],
    'logseries': [([0.6], {'size': 5}), ([0.99], {'size': 3}), ([1.0], {})],
    # methods older than the registry (hand-written on each backend), verified here against NumPy as well
    'integers': [([10], {}), ([0, 10], {'size': 5}), ([5], {'size': 3}), ([-3, 3], {'size': (2, 2)}),
                 ([0, 2 ** 40], {'size': 3}), ([7, 8], {'size': 3}), ([0, 1000003], {'size': 4})],
    'permutation': [([10], {}), ([np.array([1.5, 2.5, 3.5, 4.5, 5.5])], {}), ([1], {})],
}


def call(g, fn, args, kwargs):
    with np.errstate(all='ignore'):
        try:
            r = getattr(g, fn)(*args, **kwargs)
            if isinstance(r, np.ndarray) and r.dtype == np.float32:
                r = r.astype(np.float64)                  # float32 values are exact in float64
            expect = F.enc_result(r, [])
        except Exception as e:  # noqa: BLE001
            expect = {'error': type(e).__name__}
    kw = {k: ('float32' if v is np.float32 else v) for k, v in kwargs.items()}
    kw = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}
    return {'args': [F.enc(a) for a in args], 'kwargs': {k: F.enc(v) for k, v in kw.items()}, 'expect': expect}


def main():
    blocks = []
    for fn, seq in CASES.items():
        cases = []
        for seed in (0, 42, 20260928):
            g = np.random.default_rng(seed)
            cases.append({'seed': seed, 'calls': [call(g, fn, a, k) for a, k in seq]})
        blocks.append({'fn': fn, 'cases': cases})
    data = {'module': 'random', 'numpy': np.__version__, 'env': fixture_env.env(), 'calls': blocks}
    fixture_env.require_pinned()
    with open(OUT, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f'random: {len(blocks)} methods, {sum(len(b["cases"]) * len(CASES[b["fn"]]) for b in blocks)} calls')


if __name__ == '__main__':
    main()
