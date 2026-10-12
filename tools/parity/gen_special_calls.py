#!/usr/bin/env python3
"""Splice scipy.special reduction calls (softmax, log_softmax) into special.json's "calls".

These functions reduce over an axis (max -> exp -> sum), so they are not element-wise ufuncs and do not live in
spec/special.yaml. Under the ADR-0012 pin NumPy's exp is the C library's (AVX512 dispatch disabled), so their
reference values are host-independent across AVX2+. This reads the committed special.json, splices exactly these
call blocks (keyed by function name), and writes it back, leaving the element-wise "blocks", the recorded "env",
and any other "calls" (e.g. logsumexp) untouched. Companion to gen_misc.py (which does the same for np_numeric).

    tools/parity/gen-fixtures.sh special_calls      # (needs the pinned env; see gen-fixtures.sh)
    python3 tools/parity/gen_special_calls.py       # directly, with the pinned env exported
"""
import json
import os
import warnings

import numpy as np
import fixture_env
import scipy.special as sc
from fixtures_np_shape import enc_arg, enc_out

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'fixtures', 'parity', 'special.json')
rng = np.random.default_rng(20260930)


def F(*shape, lo=-10.0, hi=10.0):
    return rng.uniform(lo, hi, size=shape)


def call(f, args, kwargs=None):
    kwargs = kwargs or {}
    enc_args = [enc_arg(a) for a in args]
    enc_kwargs = {k: enc_arg(v) for k, v in kwargs.items()}
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            expect = enc_out(f(*args, **kwargs))
        except Exception as e:  # noqa: BLE001 - the exception class is the expectation
            expect = {'error': type(e).__name__}
    return {'args': enc_args, 'kwargs': enc_kwargs, 'expect': expect, 'compare': 'tol'}


def build():
    inputs = [F(6), F(3, 4), F(2, 3, 4), np.array([1.0, 2.0, 3.0]),
              rng.integers(-5, 6, size=6).astype(np.int64), F(2, 3, lo=-40, hi=40)]
    blocks = {}
    for name, f in (('softmax', sc.softmax), ('log_softmax', sc.log_softmax)):
        cases = [call(f, [x]) for x in inputs]                        # axis=None (flatten)
        for x in inputs:
            for ax in range(-x.ndim, x.ndim):
                cases.append(call(f, [x], {'axis': ax}))
        blocks[name] = cases
    roots_fns = {
        'roots_chebyt': sc.roots_chebyt, 't_roots': sc.t_roots,
        'roots_chebyu': sc.roots_chebyu, 'u_roots': sc.u_roots,
        'roots_chebyc': sc.roots_chebyc, 'c_roots': sc.c_roots,
        'roots_chebys': sc.roots_chebys, 's_roots': sc.s_roots,
        'roots_sh_chebyt': sc.roots_sh_chebyt, 'ts_roots': sc.ts_roots,
        'roots_sh_chebyu': sc.roots_sh_chebyu, 'us_roots': sc.us_roots,
    }
    for nm, rf in roots_fns.items():
        blocks[nm] = [call(rf, [k]) for k in (1, 2, 3, 5, 8)]
    gw_fns = {
        'roots_legendre': (sc.roots_legendre, []), 'p_roots': (sc.p_roots, []),
        'roots_hermite': (sc.roots_hermite, []), 'h_roots': (sc.h_roots, []),
        'roots_laguerre': (sc.roots_laguerre, []), 'l_roots': (sc.l_roots, []),
        'roots_hermitenorm': (sc.roots_hermitenorm, []), 'he_roots': (sc.he_roots, []),
        'roots_jacobi': (sc.roots_jacobi, [1.5, 0.7]), 'j_roots': (sc.j_roots, [1.5, 0.7]),
        'roots_gegenbauer': (sc.roots_gegenbauer, [1.3]), 'cg_roots': (sc.cg_roots, [1.3]),
        'roots_genlaguerre': (sc.roots_genlaguerre, [1.5]), 'la_roots': (sc.la_roots, [1.5]),
        'roots_sh_legendre': (sc.roots_sh_legendre, []), 'ps_roots': (sc.ps_roots, []),
        'roots_sh_jacobi': (sc.roots_sh_jacobi, [2.0, 1.0]), 'js_roots': (sc.js_roots, [2.0, 1.0]),
    }
    for nm, (rf, extra) in gw_fns.items():
        blocks[nm] = [call(rf, [k] + extra) for k in (2, 3, 5, 8)]
    return blocks


def main():
    blocks = build()
    with open(OUT) as fh:
        data = json.load(fh)
    by_fn = {c['fn']: c for c in data['calls']}
    GW_TOL = {'roots_legendre', 'p_roots', 'roots_hermite', 'h_roots', 'roots_laguerre', 'l_roots',
              'roots_hermitenorm', 'he_roots', 'roots_jacobi', 'j_roots', 'roots_gegenbauer', 'cg_roots',
              'roots_genlaguerre', 'la_roots', 'roots_sh_legendre', 'ps_roots', 'roots_sh_jacobi', 'js_roots'}
    for k, v in blocks.items():
        by_fn[k] = {'fn': k, 'cases': v}
        if k in GW_TOL:
            by_fn[k]['tol'] = {'rtol': 1e-9, 'atol': 1e-11}
    data['calls'] = list(by_fn.values())
    fixture_env.require_pinned()
    with open(OUT, 'w') as fh:
        json.dump(data, fh, separators=(',', ':'))
    print(f"gen_special_calls: spliced {len(blocks)} fns, {sum(len(v) for v in blocks.values())} cases into special.json")


if __name__ == '__main__':
    main()
