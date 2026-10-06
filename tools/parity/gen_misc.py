#!/usr/bin/env python3
"""Splice the completeness blocks (spacing, modf, bitwise_*, fromiter, can_cast) into np_numeric.json.

These routines have no BLAS or transcendental dependence, so their reference values are host-independent and the
whole np_numeric.json need not be regenerated under the ADR-0012 OpenBLAS pin (which cannot import numpy on a CPU
without AVX512). This reads the committed np_numeric.json, replaces exactly those blocks with freshly computed
NumPy values, and writes it back, leaving every other block (and the recorded "env") untouched.

    python3 tools/parity/gen_misc.py
"""
import json
import os
import warnings

import numpy as np

from fixtures_np_shape import enc_arg, enc_out
import fixtures_misc_blocks

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'fixtures', 'parity', 'np_numeric.json')


def call(fn, args, kwargs=None, compare='tol', f=None, tol=None):
    kwargs = kwargs or {}
    f = f or getattr(np, fn)
    enc_args = [enc_arg(a) for a in args]
    enc_kwargs = {k: enc_arg(v) for k, v in kwargs.items()}
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            expect = enc_out(f(*args, **kwargs))
        except Exception as e:  # noqa: BLE001 - the exception class is the expectation
            expect = {'error': type(e).__name__}
    c = {'args': enc_args, 'kwargs': enc_kwargs, 'expect': expect, 'compare': compare}
    if tol is not None:
        c['tol'] = tol
    return c


def main():
    blocks = fixtures_misc_blocks.build(call)
    with open(OUT) as f:
        data = json.load(f)
    by_fn = {c['fn']: c for c in data['calls']}
    for k, v in blocks.items():
        by_fn[k] = {'fn': k, 'cases': v}
    data['calls'] = list(by_fn.values())
    with open(OUT, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f'gen_misc: spliced {len(blocks)} blocks, {sum(len(v) for v in blocks.values())} cases into np_numeric.json')


if __name__ == '__main__':
    main()
