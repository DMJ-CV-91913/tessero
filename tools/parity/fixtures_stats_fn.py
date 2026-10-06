#!/usr/bin/env python3
"""Shared helpers for the scipy.stats function fixtures (tools/parity/fixtures_stats_<group>.py).

Each group module builds a list of calls with `call()` and writes them with `write()` into
tests/fixtures/parity/stats_<group>.json (module "stats", kind "functions"); tools/parity/run-tests.php
replays every call against Tessero\\Stats (FFI) or Tessero\\Ext\\Stats with the keyword arguments as PHP named
arguments (snake_case -> camelCase) and compares kind, shape and values (default rtol 1e-12).

Result objects (TtestResult, SignificanceResult, DescribeResult, ...) are recorded as {"dict": {field: value}}
with the field names given to call(); Tessero returns an associative array with the same keys.
"""
import json, os, time, warnings

import numpy as np
import fixture_env
import scipy
import scipy.stats as st

import fixtures_np as F

OUTDIR = os.path.join(F.ROOT, 'tests', 'fixtures', 'parity')


def _enc_obj(r, names):
    if names:
        vals = []
        for n in names:
            v = getattr(r, n) if not isinstance(r, tuple) or hasattr(r, n) else r[names.index(n)]
            vals.append(v)
        return {'dict': {n: _enc_any(v) for n, v in zip(names, vals)}}
    if isinstance(r, tuple):
        return {'list': [_enc_any(v) for v in r]}
    return _enc_any(r)


def _enc_any(v):
    if isinstance(v, tuple):
        return {'list': [_enc_any(x) for x in v]}
    return F.enc_result_one(v)


def call(fn, args, kwargs=None, names=None, compare='tol', tol=None):
    """Record scipy.stats.<fn>(*args, **kwargs). `names`: the result fields Tessero returns (in order)."""
    kwargs = kwargs or {}
    f = getattr(st, fn)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            expect = _enc_obj(f(*args, **kwargs), names)
        except Exception as e:  # noqa: BLE001 - the exception class is the expectation
            expect = {'error': type(e).__name__}
    rec = {'args': [F.enc(a) for a in args], 'kwargs': {k: F.enc(list(v) if isinstance(v, tuple) else v) for k, v in kwargs.items()},
           'expect': expect, 'compare': compare}
    if tol:
        rec['tol'] = tol
    return rec


def write(group, blocks):
    """blocks: {fn: [call(...), ...]}"""
    out = {'module': 'stats', 'kind': 'functions', 'numpy': np.__version__, 'scipy': scipy.__version__, 'env': fixture_env.env(),
           'calls': [{'fn': k, 'cases': v} for k, v in blocks.items()]}
    path = os.environ.get('FIXTURE_OUT') or os.path.join(OUTDIR, f'stats_{group}.json')
    fixture_env.require_pinned()
    with open(path, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'stats_{group}: {len(blocks)} functions, {sum(len(v) for v in blocks.values())} calls -> {path}')


rng = np.random.default_rng(20260928)


def data(shape, lo=-10.0, hi=10.0, nan=0.0, ties=False, ints=False):
    """Seeded test data (the same helper style as fixtures_np.py)."""
    if ints:
        return rng.integers(-5, 6, size=shape).astype(np.int64)
    a = (rng.integers(0, 8, size=shape) / 2.0) if ties else rng.uniform(lo, hi, size=shape)
    if nan:
        a = np.where(rng.uniform(size=shape) < nan, np.nan, a)
    return a
