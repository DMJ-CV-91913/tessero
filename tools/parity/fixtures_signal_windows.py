#!/usr/bin/env python3
"""scipy.signal.windows fixtures (module 'windows' -> Tessero\\SignalWindows / Tessero\\Ext\\SignalWindows).
Closed-form windows (hann, hamming, blackman, ..., parzen), length M in, 1-D array out, compared to
scipy.signal.windows within tolerance. The values are trig/polynomial over libm, so they record the pinned
environment like the other NumPy/SciPy generators."""
import json, os
import numpy as np
import scipy.signal.windows as w
import fixtures_np as F  # enc / enc_result / fixture_env

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'windows.json')

MS = [1, 2, 7, 8, 11]


def case(fn, M, kwargs=None, extra=None):
    kwargs = kwargs or {}
    pos = [M] + list(extra or [])
    r = getattr(w, fn)(*pos, **kwargs)
    return {'args': [F.enc(a) for a in pos], 'kwargs': {k: F.enc(v) for k, v in kwargs.items()},
            'expect': F.enc_result(r, []), 'compare': 'tol'}


SIMPLE = ['hann', 'hamming', 'blackman', 'blackmanharris', 'nuttall', 'flattop', 'boxcar', 'triang',
          'bartlett', 'cosine', 'lanczos', 'bohman', 'barthann', 'parzen']

blocks = {}
for fn in SIMPLE:
    cs = []
    for M in MS:
        cs.append(case(fn, M))
        cs.append(case(fn, M, {'sym': False}))
    blocks[fn] = cs

blocks['general_cosine'] = (
    [case('general_cosine', M, extra=[[0.5, 0.5]]) for M in MS]
    + [case('general_cosine', 8, {'sym': False}, extra=[[0.42, 0.5, 0.08]]),
       case('general_cosine', 9, extra=[[0.35875, 0.48829, 0.14128, 0.01168]])])
blocks['general_hamming'] = (
    [case('general_hamming', M, extra=[0.6]) for M in MS]
    + [case('general_hamming', 9, {'sym': False}, extra=[0.75])])
blocks['kaiser'] = (
    [case('kaiser', M, extra=[8.0]) for M in MS]
    + [case('kaiser', 9, {'sym': False}, extra=[14.0]), case('kaiser', 8, extra=[0.5])])
blocks['gaussian'] = (
    [case('gaussian', M, extra=[2.0]) for M in MS]
    + [case('gaussian', 9, {'sym': False}, extra=[1.5])])
blocks['general_gaussian'] = (
    [case('general_gaussian', M, extra=[1.5, 2.0]) for M in MS]
    + [case('general_gaussian', 8, {'sym': False}, extra=[2.0, 1.5])])
blocks['exponential'] = (
    [case('exponential', M) for M in MS]
    + [case('exponential', 8, {'tau': 2.0}), case('exponential', 9, {'center': 4.0, 'tau': 1.5, 'sym': False})])
blocks['tukey'] = (
    [case('tukey', M, extra=[0.5]) for M in MS]
    + [case('tukey', 8, extra=[0.3]), case('tukey', 8, extra=[0.0]), case('tukey', 8, extra=[1.0]),
       case('tukey', 9, {'sym': False}, extra=[0.5])])

out = {'module': 'windows', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(),
       'calls': [{'fn': k, 'cases': v, 'tol': {'rtol': 1e-9, 'atol': 1e-11}} for k, v in blocks.items()]}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'windows: {len(blocks)} functions, {sum(len(v) for v in blocks.values())} cases')
