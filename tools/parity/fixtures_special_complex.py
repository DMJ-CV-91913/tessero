#!/usr/bin/env python3
"""SciPy reference fixtures for the complex-valued scipy.special routines (special.* -> Tessero\\Special and
Tessero\\Ext\\Special): wofz, hankel1, hankel2, lambertw.

    python3 tools/parity/fixtures_special_complex.py

Writes tests/fixtures/parity/special_complex.json. Tessero and SciPy both compute these through the vendored
xsf library (ADR 0011), so the results match tightly.
"""
import json
import math
import os

import numpy as np
import scipy.special as sc
import fixture_env

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'fixtures', 'parity', 'special_complex.json')


def enc_scalar(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    if math.isnan(v):
        return 'nan'
    if math.isinf(v):
        return 'inf' if v > 0 else '-inf'
    return v


def enc(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        if np.iscomplexobj(v):
            data = []
            for x in v.ravel():
                data += [enc_scalar(x.real), enc_scalar(x.imag)]
            return {'shape': list(v.shape), 'dtype': 'complex128', 'data': data}
        return {'shape': list(v.shape), 'dtype': 'float64', 'data': [enc_scalar(x) for x in v.ravel()]}
    if isinstance(v, complex):
        return {'shape': [1], 'dtype': 'complex128', 'data': [enc_scalar(v.real), enc_scalar(v.imag)]}
    return enc_scalar(v)


def enc_one(r):
    r = np.asarray(r)
    if r.ndim == 0:
        r = r.reshape(1)
    return enc(r)


def enc_result(r, names):
    if isinstance(r, tuple) or (hasattr(r, '_fields')):
        return {'dict': {n: enc_one(x) for n, x in zip(names, tuple(r))}}
    return enc_one(r)


def call(fn, args, kwargs=None, names=None):
    kwargs = kwargs or {}
    f = getattr(sc, fn)
    try:
        expect = enc_result(f(*args, **kwargs), names or [])
    except Exception as e:  # noqa: BLE001
        expect = {'error': type(e).__name__}
    return {'args': [enc(a) for a in args], 'kwargs': {k: enc(v) for k, v in kwargs.items()}, 'expect': expect}


def main():
    blocks = {}
    z1 = np.array([1 + 2j, -1 + 0.5j, 0.5 - 1j, 2 + 0j, 0 + 1j, 3 + 2j, -2 - 1.5j])
    z2 = np.array([0.3 + 0.4j, 1.5 + 0j, -0.5 + 2j, 4 - 1j])
    blocks['wofz'] = [call('wofz', [z1]), call('wofz', [z2]),
                      call('wofz', [np.array([0 + 0j, 5 + 5j, -3 + 0.1j])])]
    zh = np.array([1 + 1j, 2 + 0.5j, 0.5 + 2j, 3 - 1j, 1.5 + 0j])
    blocks['hankel1'] = [call('hankel1', [1.0, zh]), call('hankel1', [0.0, zh]), call('hankel1', [2.5, zh]),
                         call('hankel1', [np.array([0.0, 1.0, 2.0, 0.5, 1.5]), zh])]
    blocks['hankel2'] = [call('hankel2', [1.0, zh]), call('hankel2', [0.0, zh]), call('hankel2', [2.5, zh]),
                         call('hankel2', [np.array([0.0, 1.0, 2.0, 0.5, 1.5]), zh])]
    zl = np.array([1 + 1j, 2 + 0j, 0.5 - 0.5j, 3 + 2j, 10 + 0j, 0.1 + 0.1j])
    blocks['lambertw'] = [call('lambertw', [zl]), call('lambertw', [zl], {'k': 1}),
                          call('lambertw', [zl], {'k': -1}), call('lambertw', [np.array([1 + 0j, 2.7182818 + 0j])])]

    blocks['hankel1e'] = [call('hankel1e', [1.0, zh]), call('hankel1e', [0.0, zh]),
                          call('hankel1e', [np.array([0.0, 1.0, 2.0, 0.5, 1.5]), zh])]
    blocks['hankel2e'] = [call('hankel2e', [1.0, zh]), call('hankel2e', [0.0, zh]),
                          call('hankel2e', [np.array([0.0, 1.0, 2.0, 0.5, 1.5]), zh])]
    theta = np.array([0.3, 1.0, 1.57, 2.5, 3.0])
    phi = np.array([0.0, 0.8, 1.5, 3.0, 5.0])
    blocks['sph_harm_y'] = [call('sph_harm_y', [2, 1, theta, phi]), call('sph_harm_y', [3, -2, theta, phi]),
                            call('sph_harm_y', [0, 0, theta, phi]), call('sph_harm_y', [4, 4, theta, phi]),
                            call('sph_harm_y', [5, 0, theta, phi])]
    kx = np.array([0.5, 1.0, 2.0, 3.5, 5.0, 8.0])
    blocks['kelvin'] = [call('kelvin', [kx], names=['Be', 'Ke', 'Bep', 'Kep']),
                        call('kelvin', [np.array([0.1, 0.75, 1.5])], names=['Be', 'Ke', 'Bep', 'Kep'])]

    # complex-argument Bessel (the real ufunc jv/yv/iv/kv now has a complex128 loop); z has positive real part
    zj = np.array([1 + 1j, 2 + 0.5j, 0.5 + 2j, 3 - 1j, 1.5 + 0j])
    for fn in ('jv', 'yv', 'iv', 'kv'):
        blocks[fn] = [call(fn, [1.0, zj]), call(fn, [0.0, zj]), call(fn, [2.5, zj]),
                      call(fn, [np.array([0.0, 1.0, 2.0, 0.5, 1.5]), zj])]

    tol = {'rtol': 1e-11, 'atol': 1e-13}
    out = {'module': 'special', 'scipy': __import__('scipy').__version__, 'env': fixture_env.env(),
           'calls': [{'fn': k, 'cases': v, 'tol': tol} for k, v in blocks.items()]}
    fixture_env.require_pinned()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'special_complex: {len(blocks)} functions, {sum(len(v) for v in blocks.values())} calls')


if __name__ == '__main__':
    main()
