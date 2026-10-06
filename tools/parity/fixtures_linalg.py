#!/usr/bin/env python3
"""NumPy reference fixtures for the kernel linalg module (linalg.* -> Tessero\\Linalg and Tessero\\Ext\\Linalg).

    python3 tools/parity/fixtures_linalg.py

Writes tests/fixtures/parity/linalg.json. linalg is LAPACK/BLAS-backed, so results depend on the BLAS
summation order; the reference host cannot be reproduced bit-for-bit on every machine (ADR 0012), so each
block carries a tolerance (rtol/atol) wide enough for a different but conforming LAPACK. Singular / non-positive
-definite inputs raise (numpy.linalg.LinAlgError); the runner accepts any raise where NumPy raises.
"""
import json
import math
import os

import numpy as np
import fixture_env

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'fixtures', 'parity', 'linalg.json')
rng = np.random.default_rng(20260930)


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
    if isinstance(v, str):
        return {'s': v}
    if isinstance(v, np.ndarray):
        if np.iscomplexobj(v):                    # complex128: interleaved [re, im, ...] (matches NDArray toList)
            data = []
            for x in v.ravel():
                data += [enc_scalar(x.real), enc_scalar(x.imag)]
            return {'shape': list(v.shape), 'dtype': 'complex128', 'data': data}
        return {'shape': list(v.shape), 'dtype': 'float64', 'data': [enc_scalar(x) for x in v.ravel()]}
    return enc_scalar(v)


def enc_one(x):
    x = np.asarray(x)
    return enc_scalar(x[()]) if x.ndim == 0 else enc(x)


def enc_result(r, names):
    if hasattr(r, '_fields') or isinstance(r, tuple):
        return {'dict': {n: enc_one(x) for n, x in zip(names, tuple(r))}}
    return enc_one(r)


def call(fn, args, kwargs=None, names=None):
    kwargs = kwargs or {}
    f = getattr(np.linalg, fn)
    try:
        expect = enc_result(f(*args, **kwargs), names or [])
    except Exception as e:  # noqa: BLE001 - the exception class is the expectation
        expect = {'error': type(e).__name__}
    return {'args': [enc(a) for a in args], 'kwargs': {k: enc(v) for k, v in kwargs.items()}, 'expect': expect}


# Decomposition factors are unique only up to the sign of each singular/eigen vector; we canonicalise BOTH the
# reference (here) and Tessero's kernel output (csrc/src/np_linalg.c sign_canon / positive R diagonal) the same
# way so exact comparison is valid. Fixtures use non-degenerate inputs (distinct singular/eigen values).
def _flip_cols(v):
    v = np.array(v, dtype=float)
    flat = v.reshape(-1, v.shape[-2], v.shape[-1])
    for b in range(flat.shape[0]):
        M = flat[b]
        for j in range(M.shape[1]):
            if M[int(np.argmax(np.abs(M[:, j]))), j] < 0:
                M[:, j] *= -1.0
    return flat.reshape(v.shape)


def eigh_canon(r):
    w, v = r
    return (w, _flip_cols(v))


def svd_canon(r):
    u, s, vh = r
    u = np.array(u, float)
    vh = np.array(vh, float)
    for j in range(min(len(s), u.shape[1])):
        if u[int(np.argmax(np.abs(u[:, j]))), j] < 0:
            u[:, j] *= -1.0
            if j < vh.shape[0]:
                vh[j, :] *= -1.0
    return (u, s, vh)


def qr_canon(r):
    q, rr = r
    q = np.array(q, float)
    rr = np.array(rr, float)
    for i in range(min(rr.shape[0], rr.shape[1], q.shape[1])):
        if rr[i, i] < 0:
            rr[i, :] *= -1.0
            q[:, i] *= -1.0
    return (q, rr)


def eigvals_canon(w):
    w = np.asarray(w)
    order = np.argsort(w) if not np.iscomplexobj(w) else np.lexsort((w.imag, w.real))
    return w[order]


def eig_canon(r):
    w, v = r
    w = np.asarray(w)
    v = np.asarray(v)
    real = not np.iscomplexobj(w)
    order = np.argsort(w) if real else np.lexsort((w.imag, w.real))
    w = w[order]
    v = v[:, order]
    if real:
        for j in range(v.shape[1]):
            if v[int(np.argmax(np.abs(v[:, j]))), j] < 0:
                v[:, j] = -v[:, j]
    else:
        v = v.astype(complex)
        for j in range(v.shape[1]):
            vk = v[int(np.argmax(np.abs(v[:, j]))), j]
            if abs(vk) > 0:
                v[:, j] = v[:, j] * (np.conj(vk) / abs(vk))
    return (w, v)


def callc(fn, args, kwargs, names, canon):
    kwargs = kwargs or {}
    f = getattr(np.linalg, fn)
    try:
        expect = enc_result(canon(f(*args, **kwargs)), names)
    except Exception as e:  # noqa: BLE001
        expect = {'error': type(e).__name__}
    return {'args': [enc(a) for a in args], 'kwargs': {k: enc(v) for k, v in kwargs.items()}, 'expect': expect}


def spd(n):                                  # symmetric positive definite (for cholesky)
    a = rng.uniform(-1.0, 1.0, size=(n, n))
    return a @ a.T + n * np.eye(n)


def gen(n):                                  # general, well conditioned (diagonally dominant)
    return rng.uniform(-5.0, 5.0, size=(n, n)) + 2 * n * np.eye(n)


def main():
    blocks = {}

    sc = []
    for n in (1, 2, 3, 5, 8):
        a = gen(n)
        sc += [call('solve', [a, rng.uniform(-5, 5, size=n)]), call('solve', [a, rng.uniform(-5, 5, size=(n, 3))])]
    a3 = np.stack([gen(4) for _ in range(3)])
    sc += [call('solve', [a3, rng.uniform(-5, 5, size=(3, 4))]),
           call('solve', [a3, rng.uniform(-5, 5, size=(3, 4, 2))]),
           call('solve', [a3, rng.uniform(-5, 5, size=4)]),                       # shared vector
           call('solve', [np.array([[1.0, 1.0], [1.0, 1.0]]), np.array([1.0, 2.0])])]  # singular -> raises
    blocks['solve'] = sc

    ic = [call('inv', [gen(n)]) for n in (1, 2, 3, 5, 8)]
    ic += [call('inv', [np.stack([gen(3) for _ in range(4)])]),
           call('inv', [np.array([[1.0, 2.0], [2.0, 4.0]])])]                     # singular -> raises
    blocks['inv'] = ic

    blocks['det'] = [call('det', [gen(n)]) for n in (1, 2, 3, 5)] + \
        [call('det', [np.stack([gen(3) for _ in range(4)])]),
         call('det', [np.array([[1.0, 2.0], [2.0, 4.0]])])]                        # singular -> 0

    blocks['slogdet'] = [call('slogdet', [gen(n)], names=['sign', 'logabsdet']) for n in (1, 2, 3, 5)] + \
        [call('slogdet', [np.stack([gen(3) for _ in range(3)])], names=['sign', 'logabsdet']),
         call('slogdet', [np.array([[1.0, 2.0], [2.0, 4.0]])], names=['sign', 'logabsdet'])]  # singular -> (0,-inf)

    cc = [call('cholesky', [spd(n)]) for n in (1, 2, 3, 5, 8)]
    cc += [call('cholesky', [np.stack([spd(3) for _ in range(3)])]),
           call('cholesky', [spd(4)], {'upper': True}),
           call('cholesky', [np.array([[1.0, 2.0], [2.0, 1.0]])])]                 # not positive definite -> raises
    blocks['cholesky'] = cc

    # PAR-4.1d: unique-output SVD / symmetric-eigen family (values, pinv, rank, cond, lstsq, norm)
    def rankdef(m, n, r):
        return rng.uniform(-2, 2, size=(m, r)) @ rng.uniform(-2, 2, size=(r, n))

    def sym(n):
        a = rng.uniform(-2, 2, size=(n, n))
        return a + a.T

    blocks['svdvals'] = [call('svdvals', [gen(n)]) for n in (2, 3, 5)] + \
        [call('svdvals', [rng.uniform(-3, 3, size=(4, 6))]), call('svdvals', [rng.uniform(-3, 3, size=(6, 4))]),
         call('svdvals', [rankdef(5, 5, 3)])]
    blocks['matrix_rank'] = [call('matrix_rank', [gen(n)]) for n in (2, 4)] + \
        [call('matrix_rank', [rankdef(5, 5, 3)]), call('matrix_rank', [rankdef(6, 4, 2)]),
         call('matrix_rank', [np.zeros((3, 3))]), call('matrix_rank', [rng.uniform(-2, 2, size=(4, 6))])]
    blocks['cond'] = [call('cond', [gen(n)]) for n in (2, 3, 5)] + [call('cond', [rng.uniform(-3, 3, size=(4, 4))])]
    blocks['pinv'] = [call('pinv', [gen(n)]) for n in (2, 3)] + \
        [call('pinv', [rng.uniform(-3, 3, size=(4, 6))]), call('pinv', [rng.uniform(-3, 3, size=(6, 4))]),
         call('pinv', [rankdef(5, 5, 3)])]
    blocks['eigvalsh'] = [call('eigvalsh', [sym(n)]) for n in (1, 2, 3, 5)] + \
        [call('eigvalsh', [np.stack([sym(3) for _ in range(3)])])]
    lc = []
    for (m, n) in ((5, 3), (3, 5), (4, 4), (6, 2)):
        a = rng.uniform(-3, 3, size=(m, n))
        lc += [call('lstsq', [a, rng.uniform(-3, 3, size=m)], names=['x', 'residuals', 'rank', 'singular_values']),
               call('lstsq', [a, rng.uniform(-3, 3, size=(m, 2))], names=['x', 'residuals', 'rank', 'singular_values'])]
    blocks['lstsq'] = lc
    nv = rng.uniform(-4, 4, size=8)
    nm = rng.uniform(-4, 4, size=(4, 5))
    nm2 = rng.uniform(-4, 4, size=(3, 5))
    nc = [call('norm', [nv, o]) for o in (None, 1, 2, 3, -1, float('inf'), float('-inf'), 0)]
    nc += [call('norm', [nm, o]) for o in (None, 'fro', 'nuc', 1, -1, 2, -2, float('inf'), float('-inf'))]
    for ax in (0, 1):
        nc += [call('norm', [nm2, o, ax]) for o in (None, 1, 2, float('inf'))]
        nc.append(call('norm', [nm2, 2, ax], {'keepdims': True}))
    blocks['norm'] = nc

    # PAR-4.1e: factor decompositions (sign-canonicalised, non-degenerate inputs)
    def symd(n):                             # symmetric with distinct, separated eigenvalues
        a = rng.uniform(-1, 1, size=(n, n))
        return a + a.T + np.diag(np.arange(1.0, n + 1.0) * 5.0)
    eh = [callc('eigh', [symd(n)], None, ['eigenvalues', 'eigenvectors'], eigh_canon) for n in (2, 3, 4, 5)]
    eh.append(callc('eigh', [np.stack([symd(3) for _ in range(3)])], None, ['eigenvalues', 'eigenvectors'], eigh_canon))
    blocks['eigh'] = eh
    sv = [callc('svd', [gen(n)], None, ['U', 'S', 'Vh'], svd_canon) for n in (2, 3, 4)]           # square, full
    sv += [callc('svd', [rng.uniform(-3, 3, size=(4, 6))], {'full_matrices': False}, ['U', 'S', 'Vh'], svd_canon),
           callc('svd', [rng.uniform(-3, 3, size=(6, 4))], {'full_matrices': False}, ['U', 'S', 'Vh'], svd_canon)]
    blocks['svd'] = sv
    qr = []
    for (m, n) in ((4, 4), (5, 3), (3, 5)):
        a = rng.uniform(-3, 3, size=(m, n))
        qr.append(callc('qr', [a], None, ['Q', 'R'], qr_canon))
        qr.append(callc('qr', [a], {'mode': 'complete'}, ['Q', 'R'], qr_canon))
    blocks['qr'] = qr

    # PAR-4.1f: general eigenproblem (complex results): eigvals (values), eig (values + vectors), canonicalised
    def sym(nn):
        a = rng.uniform(-2, 2, size=(nn, nn))
        return a + a.T
    emats = [rng.uniform(-3, 3, size=(3, 3)), rng.uniform(-3, 3, size=(4, 4)), rng.uniform(-3, 3, size=(5, 5)),
             sym(3), np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([[2.0, 0.0], [0.0, 3.0]])]
    blocks['eigvals'] = [callc('eigvals', [m], None, [], eigvals_canon) for m in emats]
    blocks['eig'] = [callc('eig', [m], None, ['eigenvalues', 'eigenvectors'], eig_canon) for m in emats]

    # multi_dot: NumPy takes a list; the Tessero facade is variadic, so the arrays are the (spread) args
    def md(mats):
        return {'args': [enc(m) for m in mats], 'kwargs': {}, 'expect': enc_one(np.linalg.multi_dot(mats))}
    blocks['multi_dot'] = [
        md([gen(3), gen(3), gen(3)]),
        md([rng.uniform(-3, 3, size=(2, 4)), rng.uniform(-3, 3, size=(4, 3)), rng.uniform(-3, 3, size=(3, 5))]),
        md([rng.uniform(-3, 3, size=(3, 5)), rng.uniform(-3, 3, size=(5, 2))]),
        md([rng.uniform(-3, 3, size=(2, 6)), rng.uniform(-3, 3, size=(6, 4)), rng.uniform(-3, 3, size=(4, 4)), rng.uniform(-3, 3, size=(4, 3))]),
        md([rng.uniform(-3, 3, size=4), rng.uniform(-3, 3, size=(4, 5)), rng.uniform(-3, 3, size=5)]),   # 1-D ends -> scalar
        md([rng.uniform(-3, 3, size=3), rng.uniform(-3, 3, size=(3, 4))]),                               # 1-D first -> vector
        md([rng.uniform(-3, 3, size=(4, 3)), rng.uniform(-3, 3, size=3)]),                               # 1-D last  -> vector
    ]

    blocks['matrix_power'] = [call('matrix_power', [gen(n), p]) for (n, p) in [(2, 0), (3, 1), (3, 2), (4, 3), (2, 5), (3, -1), (4, -2)]]
    blocks['matrix_transpose'] = [call('matrix_transpose', [gen(3)]),
                                  call('matrix_transpose', [rng.uniform(-3, 3, size=(2, 5))]),
                                  call('matrix_transpose', [np.stack([gen(3) for _ in range(2)])])]
    blocks['tensorsolve'] = [call('tensorsolve', [gen(6), rng.uniform(-3, 3, size=6)]),
                             call('tensorsolve', [gen(4).reshape(2, 2, 4), rng.uniform(-3, 3, size=(2, 2))])]
    blocks['tensorinv'] = [call('tensorinv', [gen(6), 1]),
                           call('tensorinv', [gen(6).reshape(2, 3, 6), 2])]
    blocks['trace'] = [call('trace', [gen(4)]), call('trace', [rng.uniform(-3, 3, size=(3, 5))]),
                       call('trace', [gen(5)], {'offset': 1}), call('trace', [gen(5)], {'offset': -2})]
    blocks['diagonal'] = [call('diagonal', [gen(4)]), call('diagonal', [rng.uniform(-3, 3, size=(3, 5))]),
                          call('diagonal', [gen(5)], {'offset': 1}), call('diagonal', [gen(5)], {'offset': -2})]
    blocks['outer'] = [call('outer', [rng.uniform(-3, 3, size=4), rng.uniform(-3, 3, size=5)]),
                       call('outer', [rng.uniform(-3, 3, size=3), rng.uniform(-3, 3, size=3)])]
    blocks['cross'] = [call('cross', [rng.uniform(-3, 3, size=3), rng.uniform(-3, 3, size=3)]),
                       call('cross', [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])])]
    blocks['vecdot'] = [call('vecdot', [rng.uniform(-3, 3, size=5), rng.uniform(-3, 3, size=5)]),
                        call('vecdot', [rng.uniform(-3, 3, size=3), rng.uniform(-3, 3, size=3)])]
    _v = rng.uniform(-3, 3, size=8)
    blocks['vector_norm'] = [call('vector_norm', [_v]), call('vector_norm', [_v], {'ord': 1}),
                             call('vector_norm', [_v], {'ord': np.inf}), call('vector_norm', [_v], {'ord': -np.inf}),
                             call('vector_norm', [_v], {'ord': 3.0})]
    blocks['matmul'] = [call('matmul', [rng.uniform(-3, 3, size=5), rng.uniform(-3, 3, size=5)]),            # 1-D . 1-D
                        call('matmul', [rng.uniform(-3, 3, size=(3, 4)), rng.uniform(-3, 3, size=(4, 2))]),  # 2-D . 2-D
                        call('matmul', [rng.uniform(-3, 3, size=4), rng.uniform(-3, 3, size=(4, 3))]),        # 1-D . 2-D
                        call('matmul', [rng.uniform(-3, 3, size=(3, 4)), rng.uniform(-3, 3, size=4)]),        # 2-D . 1-D
                        call('matmul', [rng.uniform(-3, 3, size=(2, 3, 4)), rng.uniform(-3, 3, size=(2, 4, 5))])]  # batched
    _mn = gen(4)
    blocks['matrix_norm'] = [call('matrix_norm', [_mn]), call('matrix_norm', [_mn], {'ord': 1}),
                             call('matrix_norm', [_mn], {'ord': -1}), call('matrix_norm', [_mn], {'ord': np.inf}),
                             call('matrix_norm', [_mn], {'ord': -np.inf}), call('matrix_norm', [_mn], {'ord': 2}),
                             call('matrix_norm', [_mn], {'ord': 'nuc'})]

    tol = {'rtol': 1e-9, 'atol': 1e-11}
    out = {'module': 'linalg', 'numpy': np.__version__, 'env': fixture_env.env(),
           'calls': [{'fn': k, 'cases': v, 'tol': tol} for k, v in blocks.items()]}
    fixture_env.require_pinned()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'linalg: {len(blocks)} functions, {sum(len(v) for v in blocks.values())} calls')


if __name__ == '__main__':
    main()
