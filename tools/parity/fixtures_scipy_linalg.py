#!/usr/bin/env python3
"""NumPy/SciPy reference fixtures for the kernel scipy.linalg module (slinalg.* -> Tessero\\ScipyLinalg and
Tessero\\Ext\\ScipyLinalg).

    python3 tools/parity/fixtures_scipy_linalg.py

Writes tests/fixtures/parity/slinalg.json. scipy.linalg is LAPACK/BLAS-backed; each block carries a tolerance
wide enough for a different but conforming LAPACK (ADR 0012). This first module covers the functions whose
default behaviour matches numpy.linalg (unique outputs, no sign/phase freedom).
"""
import json
import math
import os

import numpy as np
import scipy.linalg as sla
import fixture_env

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'fixtures', 'parity', 'slinalg.json')
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
        if np.iscomplexobj(v):
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
    if isinstance(r, tuple):
        return {'dict': {n: enc_one(x) for n, x in zip(names, r)}}
    return enc_one(r)


def call(fn, args, kwargs=None, names=None):
    kwargs = kwargs or {}
    f = getattr(sla, fn)
    try:
        expect = enc_result(f(*args, **kwargs), names or [])
    except Exception as e:  # noqa: BLE001
        expect = {'error': type(e).__name__}
    return {'args': [enc(a) for a in args], 'kwargs': {k: enc(v) for k, v in kwargs.items()}, 'expect': expect}


# decomposition factors are unique only up to sign; canonicalise both sides (as in fixtures_linalg.py)
def _flip_cols(v):
    v = np.array(v, dtype=float)
    for j in range(v.shape[1]):
        if v[int(np.argmax(np.abs(v[:, j]))), j] < 0:
            v[:, j] *= -1.0
    return v


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


def eigh_canon(r):
    return (r[0], _flip_cols(r[1]))


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
    f = getattr(sla, fn)
    try:
        expect = enc_result(canon(f(*args, **kwargs)), names)
    except Exception as e:  # noqa: BLE001
        expect = {'error': type(e).__name__}
    return {'args': [enc(a) for a in args], 'kwargs': {k: enc(v) for k, v in kwargs.items()}, 'expect': expect}


def gen(n):
    return rng.uniform(-5.0, 5.0, size=(n, n)) + 2 * n * np.eye(n)


def sym(n):
    a = rng.uniform(-2.0, 2.0, size=(n, n))
    return a + a.T


def main():
    blocks = {}
    sc = []
    for n in (1, 2, 3, 5, 8):
        a = gen(n)
        sc += [call('solve', [a, rng.uniform(-5, 5, size=n)]), call('solve', [a, rng.uniform(-5, 5, size=(n, 3))])]
    blocks['solve'] = sc
    blocks['inv'] = [call('inv', [gen(n)]) for n in (1, 2, 3, 5, 8)]
    blocks['det'] = [call('det', [gen(n)]) for n in (1, 2, 3, 5)]
    blocks['svdvals'] = [call('svdvals', [gen(n)]) for n in (2, 3, 5)] + \
        [call('svdvals', [rng.uniform(-3, 3, size=(4, 6))]), call('svdvals', [rng.uniform(-3, 3, size=(6, 4))])]
    blocks['pinv'] = [call('pinv', [gen(n)]) for n in (2, 3)] + \
        [call('pinv', [rng.uniform(-3, 3, size=(4, 6))]), call('pinv', [rng.uniform(-3, 3, size=(6, 4))])]
    # matrix b only: scipy returns residues as a (K,) array (identical to numpy). For a 1-D b scipy returns a
    # scalar residue where numpy returns a (1,) array; that shape quirk is deferred to a scipy-specific lstsq.
    lc = []
    for (m, n) in ((5, 3), (3, 5), (4, 4), (6, 2)):
        a = rng.uniform(-3, 3, size=(m, n))
        lc.append(call('lstsq', [a, rng.uniform(-3, 3, size=(m, 2))], names=['x', 'residues', 'rank', 's']))
    blocks['lstsq'] = lc
    blocks['eigvalsh'] = [call('eigvalsh', [sym(n)]) for n in (1, 2, 3, 5)]
    nv = rng.uniform(-4, 4, size=8)
    nm = rng.uniform(-4, 4, size=(4, 5))
    nc = [call('norm', [nv, o]) for o in (None, 1, 2, float('inf'), float('-inf'))]
    nc += [call('norm', [nm, o]) for o in (None, 'fro', 1, -1, float('inf'), float('-inf'))]
    blocks['norm'] = nc

    # PAR-4.2b: factor decompositions (sign-canonicalised, non-degenerate inputs)
    def symd(n):
        a = rng.uniform(-1, 1, size=(n, n))
        return a + a.T + np.diag(np.arange(1.0, n + 1.0) * 5.0)
    sv = [callc('svd', [gen(n)], None, ['U', 's', 'Vh'], svd_canon) for n in (2, 3, 4)]
    sv += [callc('svd', [rng.uniform(-3, 3, size=(4, 6))], {'full_matrices': False}, ['U', 's', 'Vh'], svd_canon),
           callc('svd', [rng.uniform(-3, 3, size=(6, 4))], {'full_matrices': False}, ['U', 's', 'Vh'], svd_canon)]
    blocks['svd'] = sv
    blocks['eigh'] = [callc('eigh', [symd(n)], None, ['eigenvalues', 'eigenvectors'], eigh_canon) for n in (2, 3, 4, 5)]
    qr = []
    for (m, n) in ((4, 4), (5, 3), (3, 5)):
        a = rng.uniform(-3, 3, size=(m, n))
        qr.append(callc('qr', [a], None, ['Q', 'R'], qr_canon))
        qr.append(callc('qr', [a], {'mode': 'economic'}, ['Q', 'R'], qr_canon))
    blocks['qr'] = qr

    # PAR-4.2c: scipy-specific real routines (unique outputs, no sign freedom)
    def spd(n):
        a = rng.uniform(-1, 1, size=(n, n))
        return a @ a.T + n * np.eye(n)
    blocks['cholesky'] = [call('cholesky', [spd(n)]) for n in (2, 3, 5)] + [call('cholesky', [spd(4)], {'lower': True})]
    # cho_factor returns (c, lower); cho_solve takes c, lower, b unpacked (Tessero) vs ((c, lower), b) in SciPy
    blocks['cho_factor'] = [call('cho_factor', [spd(n)], names=['c', 'lower']) for n in (2, 3, 5)] + \
        [call('cho_factor', [spd(4)], {'lower': True}, names=['c', 'lower'])]
    chs = []
    for n, lower in ((2, False), (3, False), (5, False), (4, True)):
        c, low = sla.cho_factor(spd(n), lower=lower)
        for b in (rng.uniform(-3, 3, size=n), rng.uniform(-3, 3, size=(n, 2))):
            chs.append({'args': [enc(c), enc(bool(low)), enc(b)], 'kwargs': {}, 'expect': enc_one(sla.cho_solve((c, low), b))})
    blocks['cho_solve'] = chs
    blocks['pinvh'] = [call('pinvh', [spd(n)]) for n in (2, 3, 4, 5)]
    blocks['toeplitz'] = [call('toeplitz', [rng.uniform(-3, 3, size=n)]) for n in (3, 5)] + \
        [call('toeplitz', [rng.uniform(-3, 3, size=4), rng.uniform(-3, 3, size=5)]),
         call('toeplitz', [rng.uniform(-3, 3, size=5), rng.uniform(-3, 3, size=3)])]
    blocks['block_diag'] = [
        call('block_diag', [rng.uniform(-3, 3, size=(2, 2)), rng.uniform(-3, 3, size=(1, 3))]),
        call('block_diag', [rng.uniform(-3, 3, size=(2, 3)), rng.uniform(-3, 3, size=(3, 2)), rng.uniform(-3, 3, size=(1, 1))]),
        call('block_diag', [rng.uniform(-3, 3, size=(3, 3))]),
    ]
    blocks['circulant'] = [call('circulant', [rng.uniform(-3, 3, size=n)]) for n in (3, 4, 5)]
    blocks['companion'] = [call('companion', [np.array([1.0, -3.0, 2.0])]),
                           call('companion', [rng.uniform(1, 3, size=5)]),
                           call('companion', [np.array([2.0, 0.0, -1.0, 0.5])])]
    blocks['hadamard'] = [call('hadamard', [n]) for n in (1, 2, 4, 8)]
    blocks['hilbert'] = [call('hilbert', [n]) for n in (1, 3, 5)]
    blocks['hankel'] = [call('hankel', [rng.uniform(-3, 3, size=n)]) for n in (3, 5)] + \
        [call('hankel', [rng.uniform(-3, 3, size=4), rng.uniform(-3, 3, size=5)]),
         call('hankel', [rng.uniform(-3, 3, size=5), rng.uniform(-3, 3, size=3)])]
    blocks['fiedler'] = [call('fiedler', [rng.uniform(-3, 3, size=n)]) for n in (2, 4, 5)]
    blocks['leslie'] = [call('leslie', [rng.uniform(0, 3, size=n), rng.uniform(0, 1, size=n - 1)]) for n in (2, 4, 5)]
    blocks['pascal'] = [call('pascal', [n]) for n in (1, 4, 5)] + \
        [call('pascal', [5], {'kind': 'lower'}), call('pascal', [5], {'kind': 'upper'})]
    blocks['invpascal'] = [call('invpascal', [n]) for n in (1, 4, 5)] + \
        [call('invpascal', [5], {'kind': 'lower'}), call('invpascal', [5], {'kind': 'upper'})]
    blocks['convolution_matrix'] = [call('convolution_matrix', [rng.uniform(-3, 3, size=m), n]) for (m, n) in ((3, 5), (4, 4), (2, 6))] + \
        [call('convolution_matrix', [rng.uniform(-3, 3, size=4), 6], {'mode': 'same'}),
         call('convolution_matrix', [rng.uniform(-3, 3, size=4), 6], {'mode': 'valid'}),
         call('convolution_matrix', [rng.uniform(-3, 3, size=5), 3], {'mode': 'same'})]
    blocks['dft'] = [call('dft', [n]) for n in (1, 4, 5, 8)] + \
        [call('dft', [5], {'scale': 'sqrtn'}), call('dft', [4], {'scale': 'n'})]
    # Sylvester / Lyapunov (Bartels-Stewart): spectra chosen so eig(A) + eig(B) is never ~0 (solvable).
    _Asyl = np.array([[2.0, 0.5, 0.0], [0.0, 3.0, 0.3], [0.1, 0.0, 4.0]])
    _Bsyl = np.array([[-5.0, 0.2], [0.0, -6.0]])
    blocks['solve_sylvester'] = [call('solve_sylvester', [_Asyl, _Bsyl, rng.uniform(-2, 2, size=(3, 2))]),
                                 call('solve_sylvester', [_Asyl, _Asyl + 10 * np.eye(3), rng.uniform(-2, 2, size=(3, 3))])]
    _Aly = np.array([[-3.0, 1.0, 0.5], [0.0, -2.0, 0.3], [0.2, 0.0, -4.0]])
    _Qly = np.array([[2.0, 0.3, 0.1], [0.3, 3.0, 0.2], [0.1, 0.2, 1.5]])
    blocks['solve_continuous_lyapunov'] = [call('solve_continuous_lyapunov', [_Aly, _Qly])]
    blocks['eigvalsh_tridiagonal'] = [call('eigvalsh_tridiagonal', [rng.uniform(-2, 2, size=n), rng.uniform(-2, 2, size=n - 1)]) for n in (3, 5, 6)]
    _Ssym = rng.uniform(-2, 2, size=(4, 4)); _Ssym = _Ssym + _Ssym.T
    blocks['issymmetric'] = [call('issymmetric', [_Ssym]), call('issymmetric', [rng.uniform(-2, 2, size=(4, 4))]), call('issymmetric', [np.array([[1.0, 2.0], [2.0, 1.0]])])]
    blocks['ishermitian'] = [call('ishermitian', [_Ssym]), call('ishermitian', [rng.uniform(-2, 2, size=(3, 3))]), call('ishermitian', [np.eye(3)])]
    _ctoe = np.array([4.0, 1.0, 0.5, 0.2])      # diagonally dominant symmetric Toeplitz -> well-conditioned
    blocks['solve_toeplitz'] = [call('solve_toeplitz', [_ctoe, rng.uniform(-2, 2, size=4)]),
                                call('solve_toeplitz', [_ctoe, rng.uniform(-2, 2, size=(4, 2))])]
    _ccirc = np.array([5.0, 1.0, 0.5, 0.3, 0.2])   # dominant first element -> well-conditioned circulant
    blocks['solve_circulant'] = [call('solve_circulant', [_ccirc, rng.uniform(-2, 2, size=5)]),
                                 call('solve_circulant', [_ccirc, rng.uniform(-2, 2, size=(5, 2))])]
    blocks['matmul_toeplitz'] = [call('matmul_toeplitz', [_ctoe, rng.uniform(-2, 2, size=4)]),
                                 call('matmul_toeplitz', [_ctoe, rng.uniform(-2, 2, size=(4, 3))])]
    blocks['invhilbert'] = [call('invhilbert', [n]) for n in (1, 3, 4, 5)]
    blocks['eigh_tridiagonal'] = [callc('eigh_tridiagonal', [rng.uniform(-2, 2, size=n), rng.uniform(-2, 2, size=n - 1)], None, ['eigenvalues', 'eigenvectors'], eigh_canon) for n in (3, 4, 5)]
    # banded: SPD matrices in upper band storage (kd superdiagonals; row kd = diagonal).
    _band1 = np.array([[0.0, 1.0, 1.0, 1.0], [4.0, 5.0, 6.0, 7.0]])                      # tridiagonal SPD, kd=1
    _band2 = np.array([[0.0, 0.0, 0.5, 0.5], [0.0, 1.0, 1.0, 1.0], [6.0, 6.0, 6.0, 6.0]])  # kd=2 SPD
    blocks['eigvals_banded'] = [call('eigvals_banded', [_band1]), call('eigvals_banded', [_band2])]
    blocks['cholesky_banded'] = [call('cholesky_banded', [_band1]), call('cholesky_banded', [_band2])]
    _Adl = np.array([[0.5, 0.1, 0.0], [0.0, 0.3, 0.1], [0.1, 0.0, 0.4]])   # spectral radius < 1
    _Qdl = np.array([[2.0, 0.3, 0.1], [0.3, 3.0, 0.2], [0.1, 0.2, 1.5]])   # symmetric
    blocks['solve_discrete_lyapunov'] = [call('solve_discrete_lyapunov', [_Adl, _Qdl])]
    blocks['helmert'] = [call('helmert', [n]) for n in (2, 4, 5)] + \
        [call('helmert', [4], {'full': True}), call('helmert', [5], {'full': True})]
    blocks['khatri_rao'] = [call('khatri_rao', [rng.uniform(-3, 3, size=(3, 2)), rng.uniform(-3, 3, size=(4, 2))]),
                            call('khatri_rao', [rng.uniform(-3, 3, size=(2, 3)), rng.uniform(-3, 3, size=(2, 3))])]
    blocks['diagsvd'] = [call('diagsvd', [rng.uniform(0, 3, size=k), M, N]) for (k, M, N) in ((3, 3, 5), (3, 5, 3), (4, 4, 4), (2, 2, 4))]
    # orth / null_space bases are unique only up to the sign of each column; canonicalise both sides (the kernel
    # sign-fixes columns the same way). polar is sign-invariant, so a plain call suffices.
    blocks['orth'] = [callc('orth', [gen(n)], None, [], _flip_cols) for n in (2, 3, 5)] + \
        [callc('orth', [rng.uniform(-3, 3, size=(5, 3))], None, [], _flip_cols),
         callc('orth', [rng.uniform(-3, 3, size=(3, 5))], None, [], _flip_cols)]
    blocks['null_space'] = [callc('null_space', [rng.uniform(-3, 3, size=(2, 4))], None, [], _flip_cols),
                            callc('null_space', [rng.uniform(-3, 3, size=(3, 5))], None, [], _flip_cols),
                            callc('null_space', [rng.uniform(-3, 3, size=(2, 5))], None, [], _flip_cols)]
    blocks['polar'] = [call('polar', [gen(n)], names=['u', 'p']) for n in (2, 3, 5)] + \
        [call('polar', [rng.uniform(-3, 3, size=(5, 3))], names=['u', 'p']),
         call('polar', [rng.uniform(-3, 3, size=(4, 2))], {'side': 'left'}, names=['u', 'p'])]
    blocks['lu_factor'] = [call('lu_factor', [gen(n)], names=['lu', 'piv']) for n in (2, 3, 5)]
    lus = []
    for n in (2, 3, 5):
        amat = gen(n)
        lu_, piv_ = sla.lu_factor(amat)
        for b in (rng.uniform(-3, 3, size=n), rng.uniform(-3, 3, size=(n, 2))):
            lus.append({'args': [enc(lu_), enc(piv_), enc(b)], 'kwargs': {}, 'expect': enc_one(sla.lu_solve((lu_, piv_), b))})
    blocks['lu_solve'] = lus
    blocks['hessenberg'] = [call('hessenberg', [gen(n)]) for n in (2, 3, 4, 5)]
    blocks['schur'] = [call('schur', [sym(n)], names=['T', 'Z']) for n in (2, 3, 4)]

    def _ab(A, lo, up):
        m = A.shape[1]
        out = np.zeros((lo + up + 1, m))
        for ii in range(m):
            for jj in range(m):
                if -up <= ii - jj <= lo:
                    out[up + ii - jj, jj] = A[ii, jj]
        return out
    sb = []
    for lo, up, A in ((1, 1, np.array([[2., 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1], [0, 0, 1, 2]])),
                      (2, 1, np.array([[4., 1, 0, 0], [1, 4, 1, 0], [1, 1, 4, 1], [0, 1, 1, 4]]))):
        ab = _ab(A, lo, up)
        bb = rng.uniform(-3, 3, size=A.shape[0])
        sb.append({'args': [enc(lo), enc(up), enc(ab), enc(bb)], 'kwargs': {}, 'expect': enc_one(sla.solve_banded((lo, up), ab, bb))})
    blocks['solve_banded'] = sb

    def _abh(A, kd):
        m = A.shape[1]
        out = np.zeros((kd + 1, m))
        for jj in range(m):
            for ii in range(max(0, jj - kd), jj + 1):
                out[kd + ii - jj, jj] = A[ii, jj]
        return out
    shb = []
    for kd, A in ((1, np.array([[4., 1, 0], [1, 4, 1], [0, 1, 4]])),
                  (2, np.array([[5., 1, 1, 0], [1, 5, 1, 1], [1, 1, 5, 1], [0, 1, 1, 5]]))):
        ab = _abh(A, kd)
        bb = rng.uniform(-3, 3, size=A.shape[0])
        shb.append({'args': [enc(ab), enc(bb)], 'kwargs': {}, 'expect': enc_one(sla.solveh_banded(ab, bb))})
    blocks['solveh_banded'] = shb
    eb = []
    for kd, A in ((1, np.array([[4., 1, 0], [1, 5, 1], [0, 1, 6]])),
                  (1, np.array([[3., 1, 0, 0], [1, 4, 1, 0], [0, 1, 5, 1], [0, 0, 1, 6]])),
                  (2, np.array([[6., 1, 1, 0], [1, 7, 1, 1], [1, 1, 8, 1], [0, 1, 1, 9]]))):
        eb.append(callc('eig_banded', [_abh(A, kd)], None, ['w', 'v'], eigh_canon))
    blocks['eig_banded'] = eb
    blocks['qz'] = [call('qz', [gen(n), gen(n)], names=['AA', 'BB', 'Q', 'Z']) for n in (2, 3, 4)]
    blocks['sqrtm'] = [call('sqrtm', [spd(n)]) for n in (2, 3, 4)] + \
        [call('sqrtm', [np.array([[2., 1., 0.5], [0., 3., 1.], [0., 0., 4.]])]),
         call('sqrtm', [np.array([[6., 2.], [1., 5.]])])]
    blocks['logm'] = [call('logm', [spd(n)]) for n in (2, 3, 4)] + \
        [call('logm', [np.array([[2., 1., 0.5], [0., 3., 1.], [0., 0., 4.]])]),
         call('logm', [np.array([[6., 2.], [1., 5.]])])]
    blocks['ldl'] = [call('ldl', [sym(n)], names=['lu', 'd', 'perm']) for n in (2, 3, 4, 5)] + \
        [call('ldl', [spd(3)], names=['lu', 'd', 'perm'])]
    for _mf in ('sinm', 'cosm', 'sinhm', 'coshm', 'tanhm', 'expm'):
        blocks[_mf] = [call(_mf, [sym(n)]) for n in (2, 3, 4)]
    blocks['tanm'] = [call('tanm', [0.3 * sym(n)]) for n in (2, 3, 4)]   # small eigenvalues, away from pi/2
    blocks['fractional_matrix_power'] = [call('fractional_matrix_power', [spd(n), t]) for n in (2, 3) for t in (0.5, 0.3, 2.0)]
    st = []
    for n in (2, 3, 5):
        au = np.triu(rng.uniform(-3, 3, size=(n, n))) + n * np.eye(n)
        al = np.tril(rng.uniform(-3, 3, size=(n, n))) + n * np.eye(n)
        st += [call('solve_triangular', [au, rng.uniform(-3, 3, size=n)]),
               call('solve_triangular', [al, rng.uniform(-3, 3, size=(n, 2))], {'lower': True})]
    blocks['solve_triangular'] = st
    blocks['lu'] = [call('lu', [rng.uniform(-3, 3, size=(n, n))], names=['p', 'l', 'u']) for n in (2, 3, 5)] + \
        [call('lu', [rng.uniform(-3, 3, size=(4, 6))], names=['p', 'l', 'u']),
         call('lu', [rng.uniform(-3, 3, size=(6, 4))], names=['p', 'l', 'u'])]

    # PAR-4.2d: general eigenproblem (scipy returns complex always): eigvals, eig (canonicalised)
    emats = [rng.uniform(-3, 3, size=(3, 3)), rng.uniform(-3, 3, size=(4, 4)), rng.uniform(-3, 3, size=(5, 5)),
             sym(3), np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([[2.0, 0.0], [0.0, 3.0]])]
    blocks['eigvals'] = [callc('eigvals', [m], None, [], eigvals_canon) for m in emats]
    blocks['eig'] = [callc('eig', [m], None, ['eigenvalues', 'eigenvectors'], eig_canon) for m in emats]

    tol = {'rtol': 1e-9, 'atol': 1e-11}
    out = {'module': 'slinalg', 'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
           'env': fixture_env.env(), 'calls': [{'fn': k, 'cases': v, 'tol': tol} for k, v in blocks.items()]}
    fixture_env.require_pinned()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'slinalg: {len(blocks)} functions, {sum(len(v) for v in blocks.values())} calls')


if __name__ == '__main__':
    main()
