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
    bz_fns = {'jn_zeros': sc.jn_zeros, 'yn_zeros': sc.yn_zeros, 'jnp_zeros': sc.jnp_zeros, 'ynp_zeros': sc.ynp_zeros}
    for nm, rf in bz_fns.items():
        blocks[nm] = [call(rf, [nn, 6]) for nn in (0, 1, 2, 5)]
    kz_fns = {'ber_zeros': sc.ber_zeros, 'bei_zeros': sc.bei_zeros, 'ker_zeros': sc.ker_zeros, 'kei_zeros': sc.kei_zeros,
              'berp_zeros': sc.berp_zeros, 'beip_zeros': sc.beip_zeros, 'kerp_zeros': sc.kerp_zeros, 'keip_zeros': sc.keip_zeros}
    for nm, rf in kz_fns.items():
        blocks[nm] = [call(rf, [nt]) for nt in (3, 6)]
    blocks['kelvin_zeros'] = [call(sc.kelvin_zeros, [nt]) for nt in (3, 5)]
    blocks['jnyn_zeros'] = [call(sc.jnyn_zeros, [nn, 6]) for nn in (0, 1, 2, 5)]
    blocks['ai_zeros'] = [call(sc.ai_zeros, [nt]) for nt in (3, 6)]
    blocks['bi_zeros'] = [call(sc.bi_zeros, [nt]) for nt in (3, 6)]
    blocks['lmbda'] = [call(sc.lmbda, [v, x]) for v in (0, 1, 2, 3, 2.5) for x in (1.2, 3.0)]
    blocks['pbdv_seq'] = [call(sc.pbdv_seq, [v, x]) for v in (0, 1, 2, 3, 2.5) for x in (0.8, 1.5)]
    blocks['pbvv_seq'] = [call(sc.pbvv_seq, [v, x]) for v in (0, 1, 2, 3, 2.5) for x in (0.8, 1.5)]
    blocks['obl_cv_seq'] = [call(sc.obl_cv_seq, [m, m + 3, cc]) for m in (0, 1) for cc in (0.5, 2.5)]
    blocks['pro_cv_seq'] = [call(sc.pro_cv_seq, [m, m + 3, cc]) for m in (0, 1) for cc in (0.5, 2.5)]
    blocks['y0_zeros'] = [call(sc.y0_zeros, [nt]) for nt in (3, 5)]
    blocks['y1_zeros'] = [call(sc.y1_zeros, [nt]) for nt in (3, 5)]
    blocks['y1p_zeros'] = [call(sc.y1p_zeros, [nt]) for nt in (3, 5)]
    blocks['modfresnelp'] = [call(sc.modfresnelp, [F(6, lo=-3.0, hi=3.0)])]
    blocks['modfresnelm'] = [call(sc.modfresnelm, [F(6, lo=-3.0, hi=3.0)])]
    blocks['mathieu_even_coef'] = [call(sc.mathieu_even_coef, [m, q]) for m in (0, 1, 2, 3, 4) for q in (2.0, 5.0)]
    blocks['mathieu_odd_coef'] = [call(sc.mathieu_odd_coef, [m, q]) for m in (1, 2, 3, 4) for q in (2.0, 5.0)]
    blocks['erf_zeros'] = [call(sc.erf_zeros, [nt]) for nt in (3, 5)]
    blocks['fresnel_zeros'] = [call(sc.fresnel_zeros, [nt]) for nt in (3, 5)]
    blocks['fresnelc_zeros'] = [call(sc.fresnelc_zeros, [nt]) for nt in (3, 5)]
    blocks['fresnels_zeros'] = [call(sc.fresnels_zeros, [nt]) for nt in (3, 5)]
    blocks['pbdn_seq'] = [call(sc.pbdn_seq, [n, z]) for n in (0, 1, 2, 3, 4) for z in (0.8, 1.5, 2.5)]
    blocks['jnjnp_zeros'] = [call(sc.jnjnp_zeros, [nt]) for nt in (5, 10, 20)]
    _eh = [(5.0, 8.0), (2.0, 9.0), (1.0, 3.0)]
    blocks['ellip_harm'] = [call(sc.ellip_harm, [h2, k2, n, p, s]) for (h2, k2) in _eh for n in range(0, 5) for p in range(1, 2 * n + 2) for s in (3.5, 6.0)]
    blocks['ellip_harm_2'] = [call(sc.ellip_harm_2, [h2, k2, n, p, s]) for (h2, k2) in _eh for n in range(0, 4) for p in range(1, 2 * n + 2) for s in (3.5, 6.0)]
    blocks['ellip_normal'] = [call(sc.ellip_normal, [h2, k2, n, p]) for (h2, k2) in _eh for n in range(0, 4) for p in range(1, 2 * n + 2)]
    blocks['multigammaln'] = [call(sc.multigammaln, [F(5, lo=2.0, hi=9.0), d]) for d in (1, 2, 3)] + [call(sc.multigammaln, [3.5, 2])]
    import numpy as _np
    blocks['factorial2'] = [call(sc.factorial2, [_np.arange(0, 12)])]
    blocks['factorialk'] = [call(sc.factorialk, [_np.arange(0, 13), kk]) for kk in (1, 2, 3, 4, 5)]
    blocks['bernoulli'] = [call(sc.bernoulli, [m]) for m in (6, 10, 14)]
    blocks['euler'] = [call(sc.euler, [m]) for m in (6, 10, 14)]
    return blocks


def main():
    blocks = build()
    with open(OUT) as fh:
        data = json.load(fh)
    by_fn = {c['fn']: c for c in data['calls']}
    GW_TOL = {'roots_legendre', 'p_roots', 'roots_hermite', 'h_roots', 'roots_laguerre', 'l_roots',
              'roots_hermitenorm', 'he_roots', 'roots_jacobi', 'j_roots', 'roots_gegenbauer', 'cg_roots',
              'roots_genlaguerre', 'la_roots', 'roots_sh_legendre', 'ps_roots', 'roots_sh_jacobi', 'js_roots'}
    GW_TOL = GW_TOL | {'jn_zeros', 'yn_zeros', 'jnp_zeros', 'ynp_zeros'}
    GW_TOL = GW_TOL | {'ber_zeros', 'bei_zeros', 'ker_zeros', 'kei_zeros', 'berp_zeros', 'beip_zeros', 'kerp_zeros', 'keip_zeros'}
    GW_TOL = GW_TOL | {'kelvin_zeros'}
    GW_TOL = GW_TOL | {'jnyn_zeros', 'ai_zeros', 'bi_zeros'}
    GW_TOL = GW_TOL | {'lmbda', 'pbdv_seq', 'pbvv_seq', 'obl_cv_seq', 'pro_cv_seq'}
    GW_TOL = GW_TOL | {'y0_zeros', 'y1_zeros', 'y1p_zeros'}
    GW_TOL = GW_TOL | {'modfresnelp', 'modfresnelm'}
    GW_TOL = GW_TOL | {'mathieu_even_coef', 'mathieu_odd_coef'}
    GW_TOL = GW_TOL | {'erf_zeros', 'fresnel_zeros', 'fresnelc_zeros', 'fresnels_zeros'}
    GW_TOL = GW_TOL | {'pbdn_seq', 'jnjnp_zeros'}
    GW_TOL = GW_TOL | {'ellip_harm', 'ellip_harm_2', 'ellip_normal'}
    GW_TOL = GW_TOL | {'factorial2', 'factorialk', 'bernoulli', 'euler'}
    for k, v in blocks.items():
        by_fn[k] = {'fn': k, 'cases': v}
        if k in GW_TOL:
            kv = k in {'ber_zeros', 'bei_zeros', 'ker_zeros', 'kei_zeros', 'berp_zeros', 'beip_zeros', 'kerp_zeros', 'keip_zeros', 'kelvin_zeros', 'obl_cv_seq', 'pro_cv_seq', 'y0_zeros', 'y1_zeros', 'y1p_zeros', 'ellip_harm_2', 'ellip_normal'}
            by_fn[k]['tol'] = {'rtol': 1e-7, 'atol': 1e-8} if kv else {'rtol': 1e-9, 'atol': 1e-11}
    data['calls'] = list(by_fn.values())
    fixture_env.require_pinned()
    with open(OUT, 'w') as fh:
        json.dump(data, fh, separators=(',', ':'))
    print(f"gen_special_calls: spliced {len(blocks)} fns, {sum(len(v) for v in blocks.values())} cases into special.json")


if __name__ == '__main__':
    main()
