#!/usr/bin/env python3
"""scipy.signal fixtures: 1-D convolve (full/same/valid) and lfilter. Direct float64 arithmetic (no BLAS), so the
values are host-independent and this generator does not require the pinned environment."""
import json, os
import numpy as np
import scipy.signal as sg
import fixtures_np as F  # enc / enc_result

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'signal.json')
rng = np.random.default_rng(20261001)


def conv_case(a, v, mode):
    kw, pyk = {}, {}
    if mode is not None:
        kw['mode'] = F.enc(mode); pyk['mode'] = mode
    r = sg.convolve(a, v, **pyk)
    return {'args': [F.enc(a), F.enc(v)], 'kwargs': kw, 'expect': F.enc_result(r, []), 'compare': 'tol'}


def lf_case(b, a, x):
    return {'args': [F.enc(b), F.enc(a), F.enc(x)], 'kwargs': {}, 'expect': F.enc_result(sg.lfilter(b, a, x), []), 'compare': 'tol'}


def sig_case(fnname, a, v, mode):
    fn = getattr(sg, fnname)
    kw, pyk = {}, {}
    if mode is not None:
        kw['mode'] = F.enc(mode); pyk['mode'] = mode
    r = fn(a, v, **pyk)
    return {'args': [F.enc(a), F.enc(v)], 'kwargs': kw, 'expect': F.enc_result(r, []), 'compare': 'tol'}


def conv_block(fnname):
    """full/same/valid cases for an N-D convolution-family function; 'valid' kept to in1 >= in2 per axis."""
    cases = []
    for (m, n) in [(5, 3), (8, 4), (3, 7), (10, 1), (6, 6), (1, 5)]:     # 1-D
        a, v = rng.uniform(-5, 5, m), rng.uniform(-5, 5, n)
        for mode in (None, 'full', 'same'):
            cases.append(sig_case(fnname, a, v, mode))
        if m >= n:
            cases.append(sig_case(fnname, a, v, 'valid'))
    for (sa, sv) in [((5, 4), (3, 2)), ((6, 6), (3, 3)), ((4, 7), (2, 3)), ((5, 5), (1, 1))]:   # 2-D
        a, v = rng.uniform(-5, 5, sa), rng.uniform(-5, 5, sv)
        for mode in ('full', 'same', 'valid'):
            cases.append(sig_case(fnname, a, v, mode))
    return cases


convs = []
for (m, n) in [(5, 3), (8, 4), (3, 7), (10, 1), (6, 6), (1, 5)]:
    a, v = rng.uniform(-5, 5, m), rng.uniform(-5, 5, n)
    for mode in (None, 'full', 'same', 'valid'):
        convs.append(conv_case(a, v, mode))

lfs = [
    lf_case(np.array([0.25, 0.25, 0.25, 0.25]), np.array([1.0]), rng.uniform(-5, 5, 10)),        # FIR moving average
    lf_case(np.array([0.2, 0.2]), np.array([1.0, -0.5]), rng.uniform(-5, 5, 12)),                # IIR 1st order
    lf_case(rng.uniform(-1, 1, 3), np.array([1.0, 0.3, -0.2]), rng.uniform(-5, 5, 15)),          # IIR 2nd order
    lf_case(np.array([1.0]), np.array([1.0, -0.9]), np.ones(8)),                                 # leaky integrator
]

fftc = conv_block('fftconvolve')
oac = conv_block('oaconvolve')
corr = conv_block('correlate')


def sg_coeffs_case(wl, po, deriv=0, delta=1.0):
    r = sg.savgol_coeffs(wl, po, deriv=deriv, delta=delta)
    return {'args': [F.enc(wl), F.enc(po), F.enc(deriv), F.enc(delta)], 'kwargs': {}, 'expect': F.enc_result(r, []), 'compare': 'tol'}


def sg_filter_case(x, wl, po, deriv=0, delta=1.0, mode='interp', cval=0.0):
    r = sg.savgol_filter(x, wl, po, deriv=deriv, delta=delta, mode=mode, cval=cval)
    return {'args': [F.enc(x), F.enc(wl), F.enc(po), F.enc(deriv), F.enc(delta), F.enc(-1), F.enc(mode), F.enc(cval)],
            'kwargs': {}, 'expect': F.enc_result(r, []), 'compare': 'tol'}


def detrend_case(data, axis=-1, typ='linear', bp=None):
    if bp is None:
        r = sg.detrend(data, axis=axis, type=typ)
        args = [F.enc(data), F.enc(axis), F.enc(typ)]
    else:
        r = sg.detrend(data, axis=axis, type=typ, bp=bp)
        args = [F.enc(data), F.enc(axis), F.enc(typ), F.enc(np.asarray(bp, dtype=np.int64))]
    return {'args': args, 'kwargs': {}, 'expect': F.enc_result(r, []), 'compare': 'tol'}


sgc = [sg_coeffs_case(5, 2), sg_coeffs_case(7, 3), sg_coeffs_case(9, 2, deriv=1),
       sg_coeffs_case(11, 4), sg_coeffs_case(7, 2, deriv=2), sg_coeffs_case(5, 2, delta=0.5),
       sg_coeffs_case(9, 4, deriv=1, delta=2.0)]

sgf = []
for npts in (20, 31):
    x = rng.uniform(-5, 5, npts)
    for (wl, po) in ((5, 2), (7, 3), (9, 2)):
        sgf.append(sg_filter_case(x, wl, po))                       # mode='interp' (default)
        for mode in ('mirror', 'nearest', 'wrap', 'constant'):
            sgf.append(sg_filter_case(x, wl, po, mode=mode))
    sgf.append(sg_filter_case(x, 7, 3, deriv=1, delta=0.5))
    sgf.append(sg_filter_case(x, 9, 4, deriv=2))

dtr = [
    detrend_case(rng.uniform(-5, 5, 20) + np.arange(20) * 0.7),            # 1-D linear
    detrend_case(rng.uniform(-5, 5, 20) + 3.0, typ='constant'),           # 1-D constant
    detrend_case(rng.uniform(-5, 5, (4, 10)) + np.arange(10) * 0.5, axis=1),   # 2-D along axis 1
    detrend_case(rng.uniform(-5, 5, (10, 4)) + np.arange(10)[:, None] * 0.5, axis=0),  # 2-D along axis 0
    detrend_case(rng.uniform(-5, 5, 24) + np.arange(24) * 0.3, bp=[8, 16]),  # piecewise linear
]

# ---- IIR filtering: lfilter_zi, filtfilt, sosfilt (coefficients from scipy.signal.butter) ----------
from scipy.signal import butter, lfilter_zi as _lfz, filtfilt as _ff, sosfilt as _sf
from scipy.signal import sosfilt_zi as _sfz, sosfiltfilt as _sff, freqz as _fqz, zpk2tf as _z2t, cheby1 as _cb1, cheby2 as _cb2


def lfz_case(b, a):
    return {'args': [F.enc(b), F.enc(a)], 'kwargs': {}, 'expect': F.enc_result(_lfz(b, a), []), 'compare': 'tol'}


def ff_case(b, a, x):
    return {'args': [F.enc(b), F.enc(a), F.enc(x)], 'kwargs': {}, 'expect': F.enc_result(_ff(b, a, x), []), 'compare': 'tol'}


def sf_case(sos, x):
    return {'args': [F.enc(sos), F.enc(x)], 'kwargs': {}, 'expect': F.enc_result(_sf(sos, x), []), 'compare': 'tol'}


lfz, ffc, sfc, sfz, sffc = [], [], [], [], []
_designs = [(2, 0.2, 'low'), (3, 0.3, 'low'), (4, 0.15, 'low'), (2, 0.5, 'high'), (3, [0.2, 0.5], 'band')]
for (N, Wn, bt) in _designs:
    b, a = butter(N, Wn, btype=bt)
    lfz.append(lfz_case(b, a))
    sos = butter(N, Wn, btype=bt, output='sos')
    sfz.append({'args': [F.enc(sos)], 'kwargs': {}, 'expect': F.enc_result(_sfz(sos), []), 'compare': 'tol'})
    for npts in (40, 61):
        x = rng.uniform(-5, 5, npts)
        ffc.append(ff_case(b, a, x))
        sfc.append(sf_case(sos, x))
        sffc.append({'args': [F.enc(sos), F.enc(x)], 'kwargs': {}, 'expect': F.enc_result(_sff(sos, x), []), 'compare': 'tol'})

fqz = []
for (N, Wn, bt) in _designs:
    b, a = butter(N, Wn, btype=bt)
    w, h = _fqz(b, a, worN=16)
    fqz.append({'args': [F.enc(b), F.enc(a), F.enc(16)], 'kwargs': {}, 'expect': F.enc_result((w, h), ['w', 'h']), 'compare': 'tol'})
    w, h = _fqz(b, a, worN=25, whole=True)
    fqz.append({'args': [F.enc(b), F.enc(a), F.enc(25), F.enc(True)], 'kwargs': {}, 'expect': F.enc_result((w, h), ['w', 'h']), 'compare': 'tol'})

butc, z2tc = [], []
for (N, Wn, bt) in [(2, 0.2, 'low'), (3, 0.3, 'low'), (4, 0.25, 'low'), (2, 0.5, 'high'), (5, 0.4, 'low'), (3, 0.6, 'high'), (1, 0.3, 'low')]:
    b, a = butter(N, Wn, btype=bt)
    butc.append({'args': [F.enc(N), F.enc(Wn), F.enc(bt)], 'kwargs': {}, 'expect': F.enc_result((b, a), ['b', 'a']), 'compare': 'tol'})
    z, p, k = butter(N, Wn, btype=bt, output='zpk')
    z2tc.append({'args': [F.enc(z), F.enc(p), F.enc(float(k))], 'kwargs': {}, 'expect': F.enc_result(_z2t(z, p, k), ['b', 'a']), 'compare': 'tol'})

cb1c, cb2c = [], []
for (N, rp, Wn, bt) in [(2, 1.0, 0.2, 'low'), (3, 0.5, 0.3, 'low'), (4, 1.0, 0.25, 'low'), (2, 2.0, 0.5, 'high'), (3, 1.0, 0.6, 'high'), (1, 1.0, 0.3, 'low')]:
    b, a = _cb1(N, rp, Wn, btype=bt)
    cb1c.append({'args': [F.enc(N), F.enc(rp), F.enc(Wn), F.enc(bt)], 'kwargs': {}, 'expect': F.enc_result((b, a), ['b', 'a']), 'compare': 'tol'})
for (N, rs, Wn, bt) in [(2, 20.0, 0.2, 'low'), (3, 30.0, 0.3, 'low'), (4, 40.0, 0.25, 'low'), (2, 25.0, 0.5, 'high'), (3, 30.0, 0.6, 'high'), (1, 20.0, 0.3, 'low')]:
    b, a = _cb2(N, rs, Wn, btype=bt)
    cb2c.append({'args': [F.enc(N), F.enc(rs), F.enc(Wn), F.enc(bt)], 'kwargs': {}, 'expect': F.enc_result((b, a), ['b', 'a']), 'compare': 'tol'})

# ellip: elliptic IIR filter design, digital lowpass/highpass, output 'ba'
elc = []
for (N, rp, rs, Wn, bt) in [(2, 1.0, 40.0, 0.2, 'low'), (3, 0.5, 60.0, 0.3, 'low'), (4, 1.0, 50.0, 0.25, 'low'),
                            (2, 2.0, 40.0, 0.5, 'high'), (3, 1.0, 50.0, 0.6, 'high'), (1, 1.0, 40.0, 0.3, 'low')]:
    b, a = sg.ellip(N, rp, rs, Wn, btype=bt)
    elc.append({'args': [F.enc(N), F.enc(rp), F.enc(rs), F.enc(Wn), F.enc(bt)], 'kwargs': {}, 'expect': F.enc_result((b, a), ['b', 'a']), 'compare': 'tol'})

# bessel: Bessel/Thomson IIR filter design (phase norm), digital lowpass/highpass, output 'ba'
bslc = []
for (N, Wn, bt) in [(2, 0.2, 'low'), (3, 0.3, 'low'), (4, 0.25, 'low'), (2, 0.5, 'high'), (3, 0.6, 'high'), (1, 0.3, 'low')]:
    b, a = sg.bessel(N, Wn, btype=bt)
    bslc.append({'args': [F.enc(N), F.enc(Wn), F.enc(bt)], 'kwargs': {}, 'expect': F.enc_result((b, a), ['b', 'a']), 'compare': 'tol'})

# lfiltic: initial lfilter state from output/input ICs
lfic = []
for (b, a, y, x) in [([1.0, -0.5], [1.0, 0.3], [0.7], [0.2]), ([0.5, 0.2, 0.1], [1.0, -0.4, 0.05], [1.0, 0.5], [0.3, 0.1]),
                     ([1.0], [1.0, 0.6, 0.2], [0.4, 0.1], None), ([0.2, 0.3], [2.0, 0.5], [0.9], None)]:
    args = [F.enc(np.asarray(b, float)), F.enc(np.asarray(a, float)), F.enc(np.asarray(y, float))]
    zi = sg.lfiltic(b, a, y) if x is None else sg.lfiltic(b, a, y, x)
    if x is not None:
        args.append(F.enc(np.asarray(x, float)))
    lfic.append({'args': args, 'kwargs': {}, 'expect': F.enc_result(np.asarray(zi, float), []), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# findfreqs: log-spaced analog frequency grid (kind 'ba' coeffs and 'zp' roots)
ffq = []
for (num, den, N) in [([1.0, 0.0], [1.0, 8.0, 25.0], 9), ([1.0], [1.0, 0.5, 2.0], 12), ([2.0, 1.0], [1.0, 3.0, 2.0], 7)]:
    ffq.append({'args': [F.enc(np.asarray(num, float)), F.enc(np.asarray(den, float)), F.enc(N)], 'kwargs': {},
                'expect': F.enc_result(np.asarray(sg.findfreqs(num, den, N), float), []), 'compare': 'tol', 'tol': {'atol': 1e-12}})
zp_cases = [(np.array([-1.0]), np.array([-0.5+1j, -0.5-1j]), 10), (np.array([]), np.array([-2.0, -5.0]), 8)]
for (zz, pp, N) in zp_cases:
    ffq.append({'args': [F.enc(np.asarray(zz, complex)), F.enc(np.asarray(pp, complex)), F.enc(N), F.enc('zp')], 'kwargs': {},
                'expect': F.enc_result(np.asarray(sg.findfreqs(zz, pp, N, kind='zp'), float), []), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# wiener: 1-D Wiener filter (default and explicit noise)
wnr = []
for n in (12, 20, 31):
    im = np.sin(np.linspace(0, 6, n)) + 0.3 * np.cos(np.linspace(0, 11, n))
    for m in (3, 5):
        wnr.append({'args': [F.enc(im), F.enc(m)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(sg.wiener(im, m), float), []), 'compare': 'tol', 'tol': {'atol': 1e-11}})
    wnr.append({'args': [F.enc(im), F.enc(5), F.enc(0.4)], 'kwargs': {}, 'expect': F.enc_result(np.asarray(sg.wiener(im, 5, 0.4), float), []), 'compare': 'tol', 'tol': {'atol': 1e-11}})

# sweep_poly: frequency-swept cosine with a polynomial instantaneous frequency
swp = []
for (poly, phi) in [([0.05, 1.0], 0.0), ([0.025, -0.36, 1.25, 10.0], 20.0), ([0.1, 0.0, 2.0], 90.0)]:
    t = np.linspace(0, 3, 40)
    swp.append({'args': [F.enc(t), F.enc(np.asarray(poly, float)), F.enc(float(phi))], 'kwargs': {},
                'expect': F.enc_result(np.asarray(sg.sweep_poly(t, poly, phi), float), []), 'compare': 'tol', 'tol': {'atol': 1e-11}})

# freqz_sos: SOS cascade digital response at explicit frequencies
fzs = []
for N in (2, 3, 4):
    sos = sg.butter(N, 0.3, output='sos')
    w = np.linspace(0.0, np.pi, 24, endpoint=False)
    w2, h2 = sg.freqz_sos(sos, worN=w)
    fzs.append({'args': [F.enc(np.asarray(sos, float)), F.enc(w)], 'kwargs': {},
                'expect': F.enc_result((np.asarray(w2, float), np.asarray(h2, complex)), ['w', 'h']), 'compare': 'tol', 'tol': {'atol': 1e-11}})

# max_len_seq: LFSR m-sequence and final state (default full period, and an explicit short length)
mls = []
for nb in (3, 4, 5, 6, 8):
    seq, st = sg.max_len_seq(nb)
    mls.append({'args': [F.enc(nb)], 'kwargs': {},
                'expect': F.enc_result((np.asarray(seq, dtype=np.int64), np.asarray(st, dtype=np.int64)), ['seq', 'state']), 'compare': 'tol'})
seq, st = sg.max_len_seq(5, length=12)
mls.append({'args': [F.enc(5), F.enc(None), F.enc(12)], 'kwargs': {},
            'expect': F.enc_result((np.asarray(seq, dtype=np.int64), np.asarray(st, dtype=np.int64)), ['seq', 'state']), 'compare': 'tol'})

# upfirdn: upsample / FIR / downsample
ufd = []
_r = np.random.default_rng(7)
for (nh, nx, up, down) in [(4, 10, 1, 1), (3, 12, 2, 1), (5, 15, 1, 3), (6, 20, 3, 2), (4, 9, 2, 2)]:
    h = _r.normal(size=nh); x = _r.normal(size=nx)
    ufd.append({'args': [F.enc(h), F.enc(x), F.enc(up), F.enc(down)], 'kwargs': {},
                'expect': F.enc_result(np.asarray(sg.upfirdn(h, x, up, down), float), []), 'compare': 'tol', 'tol': {'atol': 1e-11}})

# gammatone (FIR)
gmt = []
for (freq, order, numtaps, fs) in [(440, 4, 64, 16000), (1000, 4, 48, 8000), (250, 3, 32, 8000), (2000, 5, 100, 44100)]:
    b, a = sg.gammatone(freq, 'fir', order=order, numtaps=numtaps, fs=fs)
    gmt.append({'args': [F.enc(freq), F.enc('fir'), F.enc(order), F.enc(numtaps), F.enc(fs)], 'kwargs': {},
                'expect': F.enc_result((np.asarray(b, float), np.asarray(a, float)), ['b', 'a']), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# check_COLA / check_NOLA: window-overlap constraints (window passed as an explicit array)
cola, nola = [], []
for (wname, nperseg, noverlap) in [('hann', 16, 8), ('hann', 16, 4), ('hamming', 20, 10), ('boxcar', 12, 6),
                                   ('hann', 16, 3), ('bartlett', 18, 9), ('blackman', 24, 18)]:
    win = sg.get_window(wname, nperseg, fftbins=True)
    cola.append({'args': [F.enc(np.asarray(win, float)), F.enc(nperseg), F.enc(noverlap)], 'kwargs': {},
                 'expect': F.enc_result(bool(sg.check_COLA(win, nperseg, noverlap)), []), 'compare': 'tol'})
    nola.append({'args': [F.enc(np.asarray(win, float)), F.enc(nperseg), F.enc(noverlap)], 'kwargs': {},
                 'expect': F.enc_result(bool(sg.check_NOLA(win, nperseg, noverlap)), []), 'compare': 'tol'})

# band_stop_obj: analog band-stop order objective (butter type; finite, well-defined)
bso = []
for (wp, ind, pb, sb) in [(2.0, 1, [1, 3], [0.5, 4]), (1.5, 0, [1, 3], [0.4, 5]), (2.2, 1, [1.2, 3.1], [0.6, 4.5]), (2.0, 0, [1.5, 3.5], [0.5, 5.0])]:
    pb = np.asarray(pb, float); sb = np.asarray(sb, float)
    bso.append({'args': [F.enc(float(wp)), F.enc(ind), F.enc(pb), F.enc(sb), F.enc(3.0), F.enc(30.0), F.enc('butter')], 'kwargs': {},
                'expect': F.enc_result(float(sg.band_stop_obj(wp, ind, pb, sb, 3.0, 30.0, 'butter')), []), 'compare': 'tol', 'tol': {'atol': 1e-11}})

# freqresp: analog LTI response of a (b, a) system at explicit frequencies
frp = []
for N in (2, 3, 4):
    b, a = sg.butter(N, 1.0, analog=True)
    w = np.linspace(0.1, 10.0, 24)
    w2, h2 = sg.freqresp((b, a), w=w)
    frp.append({'args': [F.enc([np.asarray(b, float), np.asarray(a, float)]), F.enc(w)], 'kwargs': {},
                'expect': F.enc_result((np.asarray(w2, float), np.asarray(h2, complex)), ['w', 'H']), 'compare': 'tol', 'tol': {'atol': 1e-11}})

# waveform generators (pure, host-independent)
def _wave_case(fn, args, pyargs):
    return {'args': [F.enc(a) for a in args], 'kwargs': {}, 'expect': F.enc_result(getattr(sg, fn)(*pyargs), []), 'compare': 'tol'}


sqw, saw, chp, gps, uimp = [], [], [], [], []
_tt = np.linspace(0, 4 * np.pi, 60)
for duty in (0.5, 0.3, 0.75, 0.1):
    sqw.append(_wave_case('square', [_tt, duty], [_tt, duty]))
for width in (1.0, 0.0, 0.5, 0.3):
    saw.append(_wave_case('sawtooth', [_tt, width], [_tt, width]))
_ct = np.linspace(0, 1.0, 80)
for (f0, t1, f1, method) in [(6.0, 1.0, 1.0, 'linear'), (1.0, 1.0, 10.0, 'linear'),
                             (1.0, 1.0, 8.0, 'quadratic'), (1.0, 1.0, 6.0, 'logarithmic'),
                             (1.0, 1.0, 6.0, 'hyperbolic')]:
    chp.append(_wave_case('chirp', [_ct, f0, t1, f1, method, 0.0], [_ct, f0, t1, f1, method, 0.0]))
chp.append(_wave_case('chirp', [_ct, 2.0, 1.0, 5.0, 'linear', 90.0], [_ct, 2.0, 1.0, 5.0, 'linear', 90.0]))
_gt = np.linspace(-0.5, 0.5, 60)
for (fc, bw) in [(5.0, 0.5), (10.0, 0.3), (3.0, 0.8)]:
    gps.append(_wave_case('gausspulse', [_gt, fc, bw], [_gt, fc, bw]))
uimp.append(_wave_case('unit_impulse', [8], [8]))
uimp.append(_wave_case('unit_impulse', [8, 3], [8, 3]))
uimp.append(_wave_case('unit_impulse', [9, 'mid'], [9, 'mid']))
uimp.append(_wave_case('unit_impulse', [8, -1], [8, -1]))

# transfer-function / zpk / sos conversions
def _sortc(c):
    c = np.asarray(c, complex)
    return c[np.lexsort((c.imag, c.real))]


nrm, t2z, s2t = [], [], []
for (b, a) in [([1., 2., 3.], [2., 4., 6.]), ([0., 1., 2.], [1., 2., 1.]), ([1., 2.], [3., 4., 5.]),
               ([2., 0., -1.], [1., 0.5, 0.25])]:
    bn, an = sg.normalize(b, a)
    nrm.append({'args': [F.enc(np.asarray(b)), F.enc(np.asarray(a))], 'kwargs': {},
                'expect': F.enc_result((bn, an), ['b', 'a']), 'compare': 'tol'})
for (N, Wn, bt) in [(2, 0.2, 'low'), (3, 0.3, 'low'), (4, 0.25, 'high'), (3, 0.5, 'low')]:
    bb, aa = butter(N, Wn, btype=bt)
    z, p, k = sg.tf2zpk(bb, aa)
    t2z.append({'args': [F.enc(bb), F.enc(aa)], 'kwargs': {},
                'expect': F.enc_result((_sortc(z), _sortc(p), float(k)), ['z', 'p', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-12}})
    sos = butter(N, Wn, btype=bt, output='sos')
    b2, a2 = sg.sos2tf(sos)
    s2t.append({'args': [F.enc(np.asarray(sos))], 'kwargs': {},
                'expect': F.enc_result((b2, a2), ['b', 'a']), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# state-space <-> transfer function
t2ss, abn, s2tf = [], [], []
for (num, den) in [([1., 3., 3.], [1., 2., 1.]), ([1., 2.], [1., 3., 2.]),
                   ([2., 0., -1.], [1., 0.5, 0.25, 0.1]), ([1.], [1., 0.4, 0.05]),
                   ([1., 0., 0.], [1., -1.5, 0.7])]:
    A, B, C, D = sg.tf2ss(num, den)
    t2ss.append({'args': [F.enc(np.asarray(num)), F.enc(np.asarray(den))], 'kwargs': {},
                 'expect': F.enc_result((A, B, C, D), ['A', 'B', 'C', 'D']), 'compare': 'tol', 'tol': {'atol': 1e-11}})
    An, Bn, Cn, Dn = sg.abcd_normalize(A, B, C, D)
    abn.append({'args': [F.enc(A), F.enc(B), F.enc(C), F.enc(D)], 'kwargs': {},
                'expect': F.enc_result((An, Bn, Cn, Dn), ['A', 'B', 'C', 'D']), 'compare': 'tol', 'tol': {'atol': 1e-12}})
    nu, de = sg.ss2tf(A, B, C, D)
    s2tf.append({'args': [F.enc(A), F.enc(B), F.enc(C), F.enc(D)], 'kwargs': {},
                 'expect': F.enc_result((np.atleast_2d(nu), de), ['num', 'den']), 'compare': 'tol', 'tol': {'atol': 1e-10}})

# zpk2ss / ss2zpk
z2ss, s2zpk = [], []
for (N, Wn, bt) in [(2, 0.3, 'low'), (3, 0.25, 'low'), (2, 0.5, 'high')]:
    z, p, k = butter(N, Wn, btype=bt, output='zpk')
    A, B, C, D = sg.zpk2ss(z, p, k)
    z2ss.append({'args': [F.enc(np.asarray(z)), F.enc(np.asarray(p)), F.enc(float(k))], 'kwargs': {},
                 'expect': F.enc_result((A, B, C, D), ['A', 'B', 'C', 'D']), 'compare': 'tol', 'tol': {'atol': 1e-9}})
    bb, aa = butter(N, Wn, btype=bt)
    A2, B2, C2, D2 = sg.tf2ss(bb, aa)
    zz, pp, kk = sg.ss2zpk(A2, B2, C2, D2)
    s2zpk.append({'args': [F.enc(A2), F.enc(B2), F.enc(C2), F.enc(D2)], 'kwargs': {},
                  'expect': F.enc_result((_sortc(zz), _sortc(pp), float(kk)), ['z', 'p', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# cont2discrete (state-space form)
c2d = []
_c2d_sys = [(np.array([[-0.5, 0.0], [0.0, -1.0]]), np.array([[1.0], [0.5]]), np.array([[1.0, 1.0]]), np.array([[0.0]])),
            (np.array([[0.0, 1.0], [-2.0, -3.0]]), np.array([[0.0], [1.0]]), np.array([[1.0, 0.0]]), np.array([[0.0]])),
            (np.array([[-1.0]]), np.array([[2.0]]), np.array([[3.0]]), np.array([[0.5]]))]
for (A, B, C, D) in _c2d_sys:
    for dt in (0.1, 0.05):
        for method, alpha in [('zoh', None), ('bilinear', None), ('euler', None), ('backward_diff', None), ('gbt', 0.3)]:
            pyk = {'method': method} if alpha is None else {'method': method, 'alpha': alpha}
            Ad, Bd, Cd, Dd, dtret = sg.cont2discrete((A, B, C, D), dt, **pyk)
            args = [F.enc(A), F.enc(B), F.enc(C), F.enc(D), F.enc(float(dt)), F.enc(method)]
            if alpha is not None:
                args.append(F.enc(float(alpha)))
            c2d.append({'args': args, 'kwargs': {},
                        'expect': F.enc_result((Ad, Bd, Cd, Dd, float(dtret)), ['Ad', 'Bd', 'Cd', 'Dd', 'dt']),
                        'compare': 'tol', 'tol': {'atol': 1e-10}})

# partial-fraction expansion
uroo, ivr, ivz = [], [], []
for (pp, tol, rt) in [([1., 1., 2., 3., 3., 3.], 1e-3, 'min'),
                      ([1.0, 1.0005, 2.0, 2.0003], 1e-2, 'avg'),
                      ([1., 2., 3., 4.], 1e-3, 'max'),
                      ([1. + 1.j, 1. + 1.j, 2. + 0.j, 3. - 1.j], 1e-3, 'min')]:
    u, m = sg.unique_roots(pp, tol=tol, rtype=rt)
    uroo.append({'args': [F.enc(np.asarray(pp)), F.enc(float(tol)), F.enc(rt)], 'kwargs': {},
                 'expect': F.enc_result((u.astype(complex), m.astype(np.int64)), ['unique', 'multiplicity']),
                 'compare': 'tol', 'tol': {'atol': 1e-9}})
for (rr, pp, kk) in [([1., 2.], [1., 2.], []), ([2., -1., 3.], [0.5, 1., 2.], [1.]),
                     ([1., 1.], [-1., -2.], []), ([3., -2., 1.], [0.2, 0.5, 0.9], [2., 1.])]:
    b, a = sg.invres(rr, pp, kk)
    ivr.append({'args': [F.enc(np.asarray(rr)), F.enc(np.asarray(pp)), F.enc(np.asarray(kk, dtype=float))], 'kwargs': {},
                'expect': F.enc_result((np.asarray(b, dtype=float), np.asarray(a, dtype=float)), ['b', 'a']),
                'compare': 'tol', 'tol': {'atol': 1e-10}})
    bz, az = sg.invresz(rr, pp, kk)
    ivz.append({'args': [F.enc(np.asarray(rr)), F.enc(np.asarray(pp)), F.enc(np.asarray(kk, dtype=float))], 'kwargs': {},
                'expect': F.enc_result((np.asarray(bz, dtype=float), np.asarray(az, dtype=float)), ['b', 'a']),
                'compare': 'tol', 'tol': {'atol': 1e-10}})

# IIR order selection (lowpass/highpass; low/high are closed-form, no optimiser)
btord, c1ord, c2ord, elord = [], [], [], []
for (wp, ws, gp, gs, an) in [(0.2, 0.3, 1.0, 40.0, False), (0.3, 0.2, 1.0, 40.0, False),
                             (0.1, 0.4, 0.5, 60.0, False), (0.4, 0.15, 2.0, 50.0, False),
                             (20.0, 50.0, 3.0, 40.0, True), (50.0, 20.0, 3.0, 40.0, True)]:
    pyk = {'analog': an}
    args = [F.enc(float(wp)), F.enc(float(ws)), F.enc(float(gp)), F.enc(float(gs)), F.enc(bool(an))]
    for fn, acc in (('buttord', btord), ('cheb1ord', c1ord), ('cheb2ord', c2ord), ('ellipord', elord)):
        o, wn = getattr(sg, fn)(wp, ws, gp, gs, **pyk)
        acc.append({'args': args, 'kwargs': {}, 'expect': F.enc_result((int(o), float(wn)), ['ord', 'wn']),
                    'compare': 'tol', 'tol': {'atol': 1e-9}})

# argrelmax / argrelmin (1-D, mode='clip')
armax, armin = [], []
_rng_ar = np.random.default_rng(13)
for data, order in [(np.array([2., 1., 2., 3., 2., 0., 1., 0.]), 1),
                    (_rng_ar.standard_normal(20), 1), (_rng_ar.standard_normal(30), 2),
                    (np.sin(np.linspace(0, 6 * np.pi, 50)), 3)]:
    armax.append({'args': [F.enc(data), F.enc(0), F.enc(int(order))], 'kwargs': {},
                  'expect': F.enc_result(np.asarray(sg.argrelmax(data, order=order)[0], dtype=np.int64), []), 'compare': 'tol'})
    armin.append({'args': [F.enc(data), F.enc(0), F.enc(int(order))], 'kwargs': {},
                  'expect': F.enc_result(np.asarray(sg.argrelmin(data, order=order)[0], dtype=np.int64), []), 'compare': 'tol'})

# residue / residuez (real-root denominators, distinct magnitudes -> unambiguous pole order)
rsd, rsz = [], []
for (b, a) in [([1., 3., 3.], [1., 6., 11., 6.]), ([1., 0.], [1., 3., 2.]),
               ([1., 0., 0., 0.], [1., 3., 2.])]:
    r, p, k = sg.residue(b, a)
    rsd.append({'args': [F.enc(np.asarray(b)), F.enc(np.asarray(a))], 'kwargs': {},
                'expect': F.enc_result((r.astype(complex), p.astype(complex), np.asarray(k, dtype=float)), ['r', 'p', 'k']),
                'compare': 'tol', 'tol': {'atol': 1e-9}})
for (b, a) in [([1.], [1., -0.5]), ([1., -0.3], [1., -0.6, 0.08]),
               ([1., 0., 0.], [1., -0.6, 0.08])]:
    r, p, k = sg.residuez(b, a)
    rsz.append({'args': [F.enc(np.asarray(b)), F.enc(np.asarray(a))], 'kwargs': {},
                'expect': F.enc_result((r.astype(complex), p.astype(complex), np.asarray(k, dtype=float)), ['r', 'p', 'k']),
                'compare': 'tol', 'tol': {'atol': 1e-9}})

# rank / median filters (zero-padded)
mdf, mdf2, ordf = [], [], []
_rng_mf = np.random.default_rng(7)
for arr, ks in [(np.array([2., 80., 6., 3.]), 3), (_rng_mf.standard_normal(9), 5),
                (_rng_mf.standard_normal((4, 5)), 3), (_rng_mf.standard_normal((5, 6)), [3, 5]),
                (_rng_mf.standard_normal((3, 4, 3)), 3)]:
    mdf.append({'args': [F.enc(arr), F.enc(ks if np.isscalar(ks) else np.asarray(ks))], 'kwargs': {},
                'expect': F.enc_result(sg.medfilt(arr, ks), []), 'compare': 'tol', 'tol': {'atol': 1e-12}})
for arr, ks in [(_rng_mf.standard_normal((5, 6)), 3), (_rng_mf.standard_normal((6, 5)), [5, 3]),
                (_rng_mf.standard_normal((7, 7)), 5)]:
    mdf2.append({'args': [F.enc(arr), F.enc(ks if np.isscalar(ks) else np.asarray(ks))], 'kwargs': {},
                 'expect': F.enc_result(sg.medfilt2d(arr, ks), []), 'compare': 'tol', 'tol': {'atol': 1e-12}})
for arr, dom, rank in [(_rng_mf.standard_normal((5, 6)), np.ones((3, 3)), 4),
                       (_rng_mf.standard_normal((6, 6)), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), 0),
                       (_rng_mf.standard_normal((5, 5)), np.ones((3, 5)), 14),
                       (_rng_mf.standard_normal(10), np.ones(5), 2)]:
    ordf.append({'args': [F.enc(arr), F.enc(dom), F.enc(int(rank))], 'kwargs': {},
                 'expect': F.enc_result(sg.order_filter(arr, dom, rank), []), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# analog lowpass-prototype transforms
def _tf_case(args, bo, ao):
    return {'args': [F.enc(np.asarray(x) if isinstance(x, (list, np.ndarray)) else x) for x in args],
            'kwargs': {}, 'expect': F.enc_result((bo, ao), ['b', 'a']), 'compare': 'tol', 'tol': {'atol': 1e-12}}


lplp, lphp, lpbp, lpbs, blin = [], [], [], [], []
for N in (1, 2, 3):
    bp, ap = sg.butter(N, 1.0, analog=True, output='ba')
    for wo in (2.0, 0.5):
        b_, a_ = sg.lp2lp(bp, ap, wo); lplp.append(_tf_case([bp, ap, wo], b_, a_))
        b_, a_ = sg.lp2hp(bp, ap, wo); lphp.append(_tf_case([bp, ap, wo], b_, a_))
    for (wo, bw) in ((2.0, 1.0), (1.5, 0.5)):
        b_, a_ = sg.lp2bp(bp, ap, wo, bw); lpbp.append(_tf_case([bp, ap, wo, bw], b_, a_))
        b_, a_ = sg.lp2bs(bp, ap, wo, bw); lpbs.append(_tf_case([bp, ap, wo, bw], b_, a_))
    for fs in (2.0, 10.0):
        b_, a_ = sg.bilinear(bp, ap, fs); blin.append(_tf_case([bp, ap, fs], b_, a_))

# frequency response: analog freqs / freqs_zpk, digital freqz_zpk
def _wh_case(args, wh):
    return {'args': [F.enc(np.asarray(x) if isinstance(x, (list, np.ndarray)) else x) for x in args],
            'kwargs': {}, 'expect': F.enc_result(wh, ['w', 'h']), 'compare': 'tol', 'tol': {'atol': 1e-12}}


frqs, frzp, frzz = [], [], []
_wa = np.linspace(0.1, 20.0, 60)
for N in (1, 2, 3):
    ba_b, ba_a = sg.butter(N, 1.0, analog=True, output='ba')
    frqs.append(_wh_case([ba_b, ba_a, _wa], sg.freqs(ba_b, ba_a, _wa)))
    z, p, kk = sg.butter(N, 1.0, analog=True, output='zpk')
    frzp.append(_wh_case([z, p, float(kk), _wa], sg.freqs_zpk(z, p, kk, _wa)))
    zd, pd, kd = sg.butter(N, 0.3, btype='low', output='zpk')
    for (wn, whole) in ((16, False), (24, True)):
        frzz.append(_wh_case([zd, pd, float(kd), wn, whole], sg.freqz_zpk(zd, pd, kd, worN=wn, whole=whole)))

# bode / dbode: Bode magnitude (dB) and phase (deg) - continuous (explicit w) and discrete (n points on [0, pi))
bodc, dbodc = [], []
_wbd = np.linspace(0.1, 20.0, 60)
for N in (1, 2, 3):
    bb, ba = sg.butter(N, 1.0, analog=True, output='ba')
    w_, mag_, ph_ = sg.bode((bb, ba), w=_wbd)
    bodc.append({'args': [F.enc(bb), F.enc(ba), F.enc(_wbd)], 'kwargs': {},
                 'expect': F.enc_result((w_, mag_, ph_), ['w', 'mag', 'phase']), 'compare': 'tol', 'tol': {'atol': 1e-9}})
for (N, Wn) in ((2, 0.3), (3, 0.25), (4, 0.4)):
    bd, ad = sg.butter(N, Wn, btype='low')
    for n_ in (24, 48):
        w_, mag_, ph_ = sg.dbode((bd, ad, True), n=n_)
        dbodc.append({'args': [F.enc(bd), F.enc(ad), F.enc(n_)], 'kwargs': {},
                      'expect': F.enc_result((w_, mag_, ph_), ['w', 'mag', 'phase']), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# lsim / impulse / step: continuous state-space time responses (expm-based, equally-spaced T)
lsimc, impc, stpc = [], [], []
_Ts = np.linspace(0.0, 8.0, 40)
for (order, Wn) in ((2, 1.0), (3, 0.8)):
    bb, aa = sg.butter(order, Wn, analog=True)
    A, B, C, D = sg.tf2ss(bb, aa)
    A = np.asarray(A, float); B = np.asarray(B, float).reshape(-1, 1)
    C = np.asarray(C, float).reshape(1, -1); D = np.atleast_2d(np.asarray(D, float))
    U = np.sin(_Ts) + 0.5
    tt, yy, xx = sg.lsim((A, B, C, D), U, _Ts)
    lsimc.append({'args': [F.enc(A), F.enc(B), F.enc(C), F.enc(D), F.enc(U), F.enc(_Ts)], 'kwargs': {},
                  'expect': F.enc_result((tt, yy, xx), ['tout', 'yout', 'xout']), 'compare': 'tol', 'tol': {'atol': 1e-8}})
    ti, hi = sg.impulse((A, B, C, D), T=_Ts)
    impc.append({'args': [F.enc(A), F.enc(B), F.enc(C), F.enc(D), F.enc(_Ts)], 'kwargs': {},
                 'expect': F.enc_result((ti, hi), ['tout', 'yout']), 'compare': 'tol', 'tol': {'atol': 1e-8}})
    ts, ys = sg.step((A, B, C, D), T=_Ts)
    stpc.append({'args': [F.enc(A), F.enc(B), F.enc(C), F.enc(D), F.enc(_Ts)], 'kwargs': {},
                 'expect': F.enc_result((ts, ys), ['tout', 'yout']), 'compare': 'tol', 'tol': {'atol': 1e-8}})

# iirnotch / iirpeak: second-order notch/peak digital filter design (closed form)
notc, peakc = [], []
for (w0, Q) in ((0.25, 30.0), (0.5, 10.0), (0.1, 5.0)):
    bn, an = sg.iirnotch(w0, Q)
    notc.append({'args': [F.enc(w0), F.enc(Q)], 'kwargs': {}, 'expect': F.enc_result((bn, an), ['b', 'a']), 'compare': 'tol', 'tol': {'atol': 1e-12}})
    bp, ap = sg.iirpeak(w0, Q)
    peakc.append({'args': [F.enc(w0), F.enc(Q)], 'kwargs': {}, 'expect': F.enc_result((bp, ap), ['b', 'a']), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# iircomb: notching/peaking comb filter (w0 chosen so fs/w0 is integral)
combc = []
for (w0, Q, ftype, pz) in ((0.25, 30.0, 'notch', False), (0.2, 25.0, 'peak', False), (0.25, 30.0, 'peak', True), (0.4, 20.0, 'notch', True)):
    bb, aa = sg.iircomb(w0, Q, ftype=ftype, pass_zero=pz)
    combc.append({'args': [F.enc(w0), F.enc(Q), F.enc(ftype), F.enc(2.0), F.enc(pz)], 'kwargs': {},
                  'expect': F.enc_result((bb, aa), ['b', 'a']), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# dlsim / dimpulse / dstep: discrete state-space responses (no expm; Ad,Bd are the given discrete system)
dlsc, dimc, dstc = [], [], []
_dt = 0.1
for (order, Wn) in ((2, 0.3), (3, 0.25)):
    bb, aa = sg.butter(order, Wn)                       # digital filter -> discrete state space
    A, B, C, D = sg.tf2ss(bb, aa)
    A = np.asarray(A, float); B = np.asarray(B, float).reshape(-1, 1)
    C = np.asarray(C, float).reshape(1, -1); D = np.atleast_2d(np.asarray(D, float))
    U = np.sin(np.arange(30) * 0.3) + 0.5
    tt, yy, xx = sg.dlsim((A, B, C, D, _dt), U)
    dlsc.append({'args': [F.enc(A), F.enc(B), F.enc(C), F.enc(D), F.enc(_dt), F.enc(U)], 'kwargs': {},
                 'expect': F.enc_result((tt, yy, xx), ['tout', 'yout', 'xout']), 'compare': 'tol', 'tol': {'atol': 1e-9}})
    ti, yimp = sg.dimpulse((A, B, C, D, _dt), n=30)
    dimc.append({'args': [F.enc(A), F.enc(B), F.enc(C), F.enc(D), F.enc(_dt), F.enc(30)], 'kwargs': {},
                 'expect': F.enc_result((ti, yimp[0]), ['tout', 'yout']), 'compare': 'tol', 'tol': {'atol': 1e-9}})
    ts, ystp = sg.dstep((A, B, C, D, _dt), n=30)
    dstc.append({'args': [F.enc(A), F.enc(B), F.enc(C), F.enc(D), F.enc(_dt), F.enc(30)], 'kwargs': {},
                 'expect': F.enc_result((ts, ystp[0]), ['tout', 'yout']), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# sosfreqz (SOS cascade response) and group_delay
sfz2, gdl = [], []
for (N, Wn, bt) in [(2, 0.3, 'low'), (3, 0.25, 'low'), (2, 0.5, 'high'), (4, 0.4, 'low')]:
    sos = butter(N, Wn, btype=bt, output='sos')
    for (wn, whole) in ((16, False), (24, True)):
        w, h = sg.sosfreqz(sos, worN=wn, whole=whole)
        sfz2.append({'args': [F.enc(np.asarray(sos)), F.enc(wn), F.enc(whole)], 'kwargs': {}, 'expect': F.enc_result((w, h), ['w', 'h']), 'compare': 'tol', 'tol': {'atol': 1e-12}})
    bb, aa = butter(N, Wn, btype=bt)
    # whole=False only: the [0, pi) grid excludes the Nyquist point where a filter's group delay can be
    # singular (den -> 0, a huge cancellation-dominated value that no engine reproduces to tolerance).
    for wn in (32, 48):
        w, gd = sg.group_delay((bb, aa), w=wn, whole=False)
        gdl.append({'args': [F.enc(bb), F.enc(aa), F.enc(wn), F.enc(False)], 'kwargs': {}, 'expect': F.enc_result((w, gd), ['w', 'gd']), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# hilbert: analytic signal via FFT
hlb = []
for n in (8, 16, 17, 31, 64):
    xx = rng.uniform(-3, 3, n)
    hlb.append({'args': [F.enc(xx)], 'kwargs': {}, 'expect': F.enc_result(sg.hilbert(xx), []), 'compare': 'tol', 'tol': {'atol': 1e-10}})
# explicit N (zero-pad and truncate)
xx = rng.uniform(-3, 3, 20)
hlb.append({'args': [F.enc(xx), F.enc(32)], 'kwargs': {}, 'expect': F.enc_result(sg.hilbert(xx, 32), []), 'compare': 'tol', 'tol': {'atol': 1e-10}})
hlb.append({'args': [F.enc(xx), F.enc(12)], 'kwargs': {}, 'expect': F.enc_result(sg.hilbert(xx, 12), []), 'compare': 'tol', 'tol': {'atol': 1e-10}})

# hilbert2: 2-D analytic signal
hlb2 = []
for (r, c) in ((4, 4), (8, 6), (5, 7), (6, 8)):
    xx = rng.uniform(-3, 3, (r, c))
    hlb2.append({'args': [F.enc(xx)], 'kwargs': {}, 'expect': F.enc_result(sg.hilbert2(xx), []), 'compare': 'tol', 'tol': {'atol': 1e-10}})

# periodogram: single-segment PSD estimate
def _pgm_case(args, pyargs):
    return {'args': [F.enc(np.asarray(a) if isinstance(a, (list, np.ndarray)) else a) for a in args],
            'kwargs': {}, 'expect': F.enc_result(sg.periodogram(*pyargs), ['f', 'Pxx']), 'compare': 'tol', 'tol': {'atol': 1e-12}}


pgm = []
xp64 = rng.uniform(-3, 3, 64)
pgm.append(_pgm_case([xp64], [xp64]))                                              # default
pgm.append(_pgm_case([xp64, 100.0], [xp64, 100.0]))                                # fs
pgm.append(_pgm_case([xp64, 1.0, 'boxcar', 64, 'constant', True, 'spectrum'],
                     [xp64, 1.0, 'boxcar', 64, 'constant', True, 'spectrum']))      # spectrum scaling
pgm.append(_pgm_case([xp64, 1.0, 'boxcar', 64, 'constant', False, 'density'],
                     [xp64, 1.0, 'boxcar', 64, 'constant', False, 'density']))      # two-sided
pgm.append(_pgm_case([xp64, 2.0, 'boxcar', 64, 'linear'], [xp64, 2.0, 'boxcar', 64, 'linear']))  # linear detrend
_win = sg.windows.hann(64, sym=False)
pgm.append(_pgm_case([xp64, 1.0, _win], [xp64, 1.0, _win]))                         # explicit window array
xp51 = rng.uniform(-3, 3, 51)
pgm.append(_pgm_case([xp51], [xp51]))                                              # odd length

# welch: averaged-periodogram PSD
def _welch_case(args, pyargs):
    return {'args': [F.enc(np.asarray(a) if isinstance(a, (list, np.ndarray)) else a) for a in args],
            'kwargs': {}, 'expect': F.enc_result(sg.welch(*pyargs), ['f', 'Pxx']), 'compare': 'tol', 'tol': {'atol': 1e-12}}


wel = []
xw = rng.uniform(-3, 3, 200)
wel.append(_welch_case([xw, 1.0, 'hann', 64], [xw, 1.0, 'hann', 64]))                       # default noverlap, mean
wel.append(_welch_case([xw, 100.0, 'hann', 64], [xw, 100.0, 'hann', 64]))                   # fs
wel.append(_welch_case([xw, 1.0, 'boxcar', 50], [xw, 1.0, 'boxcar', 50]))                   # boxcar window
wel.append(_welch_case([xw, 1.0, 'hann', 64, 16], [xw, 1.0, 'hann', 64, 16]))               # noverlap=16
wel.append(_welch_case([xw, 1.0, 'hann', 64, None, None, 'constant', True, 'spectrum'],
                       [xw, 1.0, 'hann', 64, None, None, 'constant', True, 'spectrum']))     # spectrum
wel.append(_welch_case([xw, 1.0, 'hann', 64, None, None, 'constant', False],
                       [xw, 1.0, 'hann', 64, None, None, 'constant', False]))                # two-sided
wel.append(_welch_case([xw, 1.0, 'hann', 64, None, None, 'linear'],
                       [xw, 1.0, 'hann', 64, None, None, 'linear']))                         # linear detrend
wel.append(_welch_case([xw, 1.0, 'hann', 64, None, None, 'constant', True, 'density', -1, 'median'],
                       [xw, 1.0, 'hann', 64, None, None, 'constant', True, 'density', -1, 'median']))   # median average (axis=-1 then average)

# csd: cross power spectral density (Welch)
def _csd_case(args, pyargs):
    return {'args': [F.enc(np.asarray(a) if isinstance(a, (list, np.ndarray)) else a) for a in args],
            'kwargs': {}, 'expect': F.enc_result(sg.csd(*pyargs), ['f', 'Pxy']), 'compare': 'tol', 'tol': {'atol': 1e-12}}


csdc = []
xc = rng.uniform(-3, 3, 200)
yc = 0.5 * xc + rng.uniform(-1, 1, 200)
csdc.append(_csd_case([xc, yc, 1.0, 'hann', 64], [xc, yc, 1.0, 'hann', 64]))
csdc.append(_csd_case([xc, yc, 50.0, 'hann', 64], [xc, yc, 50.0, 'hann', 64]))
csdc.append(_csd_case([xc, yc, 1.0, 'boxcar', 50], [xc, yc, 1.0, 'boxcar', 50]))
csdc.append(_csd_case([xc, yc, 1.0, 'hann', 64, 16], [xc, yc, 1.0, 'hann', 64, 16]))
csdc.append(_csd_case([xc, yc, 1.0, 'hann', 64, None, None, 'constant', False],
                      [xc, yc, 1.0, 'hann', 64, None, None, 'constant', False]))   # two-sided

# coherence
coh = []
xh = rng.uniform(-3, 3, 200)
yh = 0.7 * xh + rng.uniform(-1, 1, 200)
coh.append({'args': [F.enc(xh), F.enc(yh), F.enc(1.0), F.enc('hann'), F.enc(64)], 'kwargs': {}, 'expect': F.enc_result(sg.coherence(xh, yh, 1.0, 'hann', 64), ['f', 'Cxy']), 'compare': 'tol', 'tol': {'atol': 1e-12}})
coh.append({'args': [F.enc(xh), F.enc(yh), F.enc(50.0), F.enc('hann'), F.enc(64), F.enc(16)], 'kwargs': {}, 'expect': F.enc_result(sg.coherence(xh, yh, 50.0, 'hann', 64, 16), ['f', 'Cxy']), 'compare': 'tol', 'tol': {'atol': 1e-12}})
# boxcar with detrend=False: the default constant detrend forces the boxcar DC bin to an exact 0, making the
# DC coherence a 0/0 that is pure fp noise in any engine; detrend=False keeps every bin well-defined.
coh.append({'args': [F.enc(xh), F.enc(yh), F.enc(1.0), F.enc('boxcar'), F.enc(50), None, None, F.enc(False)], 'kwargs': {}, 'expect': F.enc_result(sg.coherence(xh, yh, 1.0, 'boxcar', 50, None, None, False), ['f', 'Cxy']), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# spectrogram (mode='psd')
def _spg_case(args, pyargs):
    return {'args': [F.enc(np.asarray(a) if isinstance(a, (list, np.ndarray)) else a) for a in args],
            'kwargs': {}, 'expect': F.enc_result(sg.spectrogram(*pyargs), ['f', 't', 'Sxx']), 'compare': 'tol', 'tol': {'atol': 1e-12}}


spg = []
xs200 = rng.uniform(-3, 3, 200)
xs128 = rng.uniform(-3, 3, 128)
spg.append(_spg_case([xs128], [xs128]))                                   # default periodic-Tukey(0.25) window
spg.append(_spg_case([xs200, 1.0, 'boxcar', 50], [xs200, 1.0, 'boxcar', 50]))
spg.append(_spg_case([xs200, 1.0, 'hann', 64], [xs200, 1.0, 'hann', 64]))
spg.append(_spg_case([xs200, 100.0, 'hann', 64], [xs200, 100.0, 'hann', 64]))
_wt = sg.windows.tukey(64, 0.25, sym=False)
spg.append(_spg_case([xs200, 1.0, _wt], [xs200, 1.0, _wt]))               # explicit window array
spg.append(_spg_case([xs200, 1.0, 'hann', 64, 48], [xs200, 1.0, 'hann', 64, 48]))   # noverlap=48

# correlation_lags: integer lag indices
clg = []
for (a, b, m) in [(7, 4, 'full'), (7, 4, 'same'), (7, 4, 'valid'), (4, 7, 'valid'),
                  (10, 10, 'same'), (5, 8, 'full'), (6, 6, 'valid'), (9, 4, 'same'), (8, 5, 'valid')]:
    clg.append({'args': [F.enc(a), F.enc(b), F.enc(m)], 'kwargs': {}, 'expect': F.enc_result(sg.correlation_lags(a, b, m), []), 'compare': 'tol'})

# resample (FFT-based, window=None)
rsmp = []
for (n0, num) in [(64, 128), (128, 64), (100, 50), (50, 100), (64, 64), (81, 40), (40, 81), (100, 75)]:
    xr = rng.uniform(-3, 3, n0)
    rsmp.append({'args': [F.enc(xr), F.enc(num)], 'kwargs': {}, 'expect': F.enc_result(sg.resample(xr, num), []), 'compare': 'tol', 'tol': {'atol': 1e-10}})

# firwin: FIR design by the window method (window/pass_zero/scale/fs are keyword-only in scipy)
def _firwin_case(numtaps, cutoff, window='hamming', pass_zero=True, scale=True, fs=None):
    args = [numtaps, np.asarray(cutoff, dtype=float), window, pass_zero, scale]
    kw = dict(window=window, pass_zero=pass_zero, scale=scale)
    if fs is not None:
        args.append(fs); kw['fs'] = fs
    r = sg.firwin(numtaps, np.asarray(cutoff, dtype=float), **kw)
    return {'args': [F.enc(np.asarray(a) if isinstance(a, np.ndarray) else a) for a in args],
            'kwargs': {}, 'expect': F.enc_result(r, []), 'compare': 'tol', 'tol': {'atol': 1e-12}}


fwn = [
    _firwin_case(31, [0.3]),                               # lowpass, hamming, scaled
    _firwin_case(31, [0.3], window='hann'),
    _firwin_case(32, [0.3]),                               # even lowpass
    _firwin_case(31, [0.3], pass_zero=False),              # highpass
    _firwin_case(31, [0.2, 0.5], pass_zero=False),         # bandpass
    _firwin_case(31, [0.2, 0.5]),                          # bandstop
    _firwin_case(31, [0.3], scale=False),
    _firwin_case(31, [30.0], fs=100.0),                    # fs
    _firwin_case(31, [0.3], window='boxcar'),
    _firwin_case(41, [0.15, 0.35, 0.6], pass_zero=False),  # multi-band
]

# kaiser FIR helpers
katt, kbeta, kord = [], [], []
for (nt, w) in [(31, 0.1), (65, 0.05), (21, 0.2)]:
    katt.append({'args': [F.enc(nt), F.enc(w)], 'kwargs': {}, 'expect': F.enc_result(sg.kaiser_atten(nt, w), []), 'compare': 'tol'})
for a in (10.0, 25.0, 40.0, 60.0, 8.0):
    kbeta.append({'args': [F.enc(a)], 'kwargs': {}, 'expect': F.enc_result(sg.kaiser_beta(a), []), 'compare': 'tol'})
for (rip, w) in [(65.0, 0.1), (40.0, 0.05), (20.0, 0.2), (30.0, 0.15)]:
    nt, beta = sg.kaiserord(rip, w)
    kord.append({'args': [F.enc(rip), F.enc(w)], 'kwargs': {}, 'expect': F.enc_result((nt, beta), ['numtaps', 'beta']), 'compare': 'tol'})

# firwin2: FIR by frequency sampling (nfreqs/window/antisymmetric/fs are keyword-only in scipy)
def _firwin2_case(numtaps, freq, gain, window='hamming', fs=None):
    args = [numtaps, np.asarray(freq, dtype=float), np.asarray(gain, dtype=float), None, window]
    kw = dict(window=window)
    if fs is not None:
        args = [numtaps, np.asarray(freq, dtype=float), np.asarray(gain, dtype=float), None, window, False, fs]; kw['fs'] = fs
    r = sg.firwin2(numtaps, np.asarray(freq, dtype=float), np.asarray(gain, dtype=float), **kw)
    return {'args': [F.enc(np.asarray(a) if isinstance(a, np.ndarray) else a) for a in args],
            'kwargs': {}, 'expect': F.enc_result(r, []), 'compare': 'tol', 'tol': {'atol': 1e-12}}


fw2 = [
    _firwin2_case(31, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0]),            # lowpass ramp
    _firwin2_case(63, [0.0, 0.3, 0.6, 1.0], [0.0, 1.0, 1.0, 0.0]),  # bandpass
    _firwin2_case(32, [0.0, 0.5, 1.0], [1.0, 0.5, 0.0]),           # even length (zero at Nyquist)
    _firwin2_case(31, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0], window='hann'),
    _firwin2_case(45, [0.0, 30.0, 60.0, 100.0], [1.0, 1.0, 0.0, 0.0], fs=200.0),  # fs
]

# sos2zpk: zpk from an SOS cascade (compare z, p as (re,im)-sorted sets)
def _sortc2(c):
    c = np.asarray(c, complex)
    return c[np.lexsort((c.imag, c.real))]


s2z = []
for (N, Wn, bt) in [(4, 0.3, 'low'), (6, 0.25, 'low'), (4, 0.5, 'high'), (3, 0.4, 'low'), (8, 0.2, 'low')]:
    sos = butter(N, Wn, btype=bt, output='sos')
    z, p, kk = sg.sos2zpk(sos)
    s2z.append({'args': [F.enc(np.asarray(sos))], 'kwargs': {},
                'expect': F.enc_result((_sortc2(z), _sortc2(p), float(kk)), ['z', 'p', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-10}})

# deconvolve: polynomial division
dcv = []
for (sig, div) in [([1., 0., 0., 0.], [1., 1.]), ([0., 1., 2., 3., 4., 5.], [1., 2., 1.]),
                   ([2., 4., 6., 8.], [2.]), ([1., 2., 3.], [1., 0., 0., 0., 1.]),  # divisor longer -> empty quotient
                   ([5., 11., 14., 6.], [5., 1.])]:
    q, r = sg.deconvolve(np.asarray(sig), np.asarray(div))
    dcv.append({'args': [F.enc(np.asarray(sig)), F.enc(np.asarray(div))], 'kwargs': {},
                'expect': F.enc_result((q, r), ['quotient', 'remainder']), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# buttap: Butterworth analog prototype (poles compared as an (re,im)-sorted set)
bap = []
for N in (1, 2, 3, 4, 5, 8):
    z, p, kk = sg.buttap(N)
    bap.append({'args': [F.enc(N)], 'kwargs': {},
                'expect': F.enc_result((_sortc2(z), _sortc2(p), float(kk)), ['z', 'p', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# zpk-domain transforms (compare z, p as (re,im)-sorted sets)
def _zpk_case(fn, z, p, k, extra):
    zz, pp, kk = getattr(sg, fn)(np.asarray(z, complex), np.asarray(p, complex), k, *extra)
    return {'args': [F.enc(np.asarray(z, complex)), F.enc(np.asarray(p, complex)), F.enc(float(k))] + [F.enc(e) for e in extra],
            'kwargs': {}, 'expect': F.enc_result((_sortc2(zz), _sortc2(pp), float(kk)), ['z', 'p', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-10}}


lplpz, lphpz, blnz = [], [], []
for N in (2, 3, 4):
    z0, p0, k0 = sg.buttap(N)
    for wo in (2.0, 0.5):
        lplpz.append(_zpk_case('lp2lp_zpk', z0, p0, k0, (wo,)))
        lphpz.append(_zpk_case('lp2hp_zpk', z0, p0, k0, (wo,)))
    for fsv in (2.0, 10.0):
        # scale prototype first so the digital filter is sensible, then bilinear
        zl, pl, kl = sg.lp2lp_zpk(z0, p0, k0, 0.3)
        blnz.append(_zpk_case('bilinear_zpk', zl, pl, kl, (fsv,)))

# bandpass/bandstop zpk transforms
lpbpz, lpbsz = [], []
for N in (2, 3, 4):
    z0, p0, k0 = sg.buttap(N)
    for (wo, bw) in ((1.0, 0.5), (2.0, 1.0)):
        lpbpz.append(_zpk_case('lp2bp_zpk', z0, p0, k0, (wo, bw)))
        lpbsz.append(_zpk_case('lp2bs_zpk', z0, p0, k0, (wo, bw)))

# Chebyshev analog prototypes (poles/zeros compared as (re,im)-sorted sets)
c1ap, c2ap = [], []
for N in (1, 2, 3, 4, 5):
    for rp in (0.5, 1.0, 3.0):
        z, p, kk = sg.cheb1ap(N, rp)
        c1ap.append({'args': [F.enc(N), F.enc(rp)], 'kwargs': {}, 'expect': F.enc_result((_sortc2(z), _sortc2(p), float(kk)), ['z', 'p', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-11}})
    for rs in (20.0, 40.0, 60.0):
        z, p, kk = sg.cheb2ap(N, rs)
        c2ap.append({'args': [F.enc(N), F.enc(rs)], 'kwargs': {}, 'expect': F.enc_result((_sortc2(z), _sortc2(p), float(kk)), ['z', 'p', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-11}})

# gauss_spline + vectorstrength
gsp, vst = [], []
for n in (0, 1, 2, 3, 5):
    xg = np.linspace(-3, 3, 25)
    gsp.append({'args': [F.enc(xg), F.enc(n)], 'kwargs': {}, 'expect': F.enc_result(sg.gauss_spline(xg, n), []), 'compare': 'tol', 'tol': {'atol': 1e-12}})
for (ev, per) in [(rng.uniform(0, 10, 40), 2.0), (rng.uniform(0, 5, 30), 1.0), (np.array([0.0, 1.0, 2.0, 3.0]), 1.0), (rng.uniform(0, 20, 60), 3.5)]:
    s, ph = sg.vectorstrength(ev, per)
    vst.append({'args': [F.enc(np.asarray(ev)), F.enc(float(per))], 'kwargs': {}, 'expect': F.enc_result((float(s), float(ph)), ['strength', 'phase']), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# convolve2d / correlate2d (boundary='fill')
cv2, cr2 = [], []
for (m, n, km, kn) in [(6, 5, 3, 3), (8, 8, 2, 4), (5, 7, 4, 2), (4, 4, 1, 1), (7, 6, 3, 5)]:
    a2 = rng.uniform(-3, 3, (m, n)); k2 = rng.uniform(-3, 3, (km, kn))
    for mode in ('full', 'same', 'valid'):
        if mode == 'valid' and (km > m or kn > n):
            continue
        cv2.append({'args': [F.enc(a2), F.enc(k2), F.enc(mode)], 'kwargs': {}, 'expect': F.enc_result(sg.convolve2d(a2, k2, mode=mode), []), 'compare': 'tol', 'tol': {'atol': 1e-11}})
        cr2.append({'args': [F.enc(a2), F.enc(k2), F.enc(mode)], 'kwargs': {}, 'expect': F.enc_result(sg.correlate2d(a2, k2, mode=mode), []), 'compare': 'tol', 'tol': {'atol': 1e-11}})

# lombscargle: Lomb-Scargle periodogram of unevenly sampled data at given angular frequencies (default options).
lsc = []
for (nx, nf) in [(30, 40), (45, 25), (60, 50)]:
    xs = np.sort(rng.uniform(0.0, 10.0, nx))
    ys = np.sin(2.3 * xs) + 0.6 * np.cos(0.7 * xs) + 0.4 * rng.normal(size=nx)
    frq = np.linspace(0.1, 6.0, nf)
    lsc.append({'args': [F.enc(xs), F.enc(ys), F.enc(frq)], 'kwargs': {},
                'expect': F.enc_result(np.asarray(sg.lombscargle(xs, ys, frq)), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# get_window: dispatch a window name / (name, *params) spec / float to the windows.* routines, periodic
# (fftbins=True) or symmetric (fftbins=False).
gw = []
gw_specs = ['hann', 'hamming', 'blackman', 'bartlett', 'boxcar', 'triang', 'blackmanharris',
            'nuttall', 'flattop', 'cosine', 'bohman', 'barthann', 'parzen', 'lanczos', 'tukey',
            ('kaiser', 8.6), ('gaussian', 2.5), ('general_gaussian', 1.5, 2.0), ('general_hamming', 0.7),
            ('tukey', 0.3), 2.5, ('boxcar',), 'rect', 'rectangular']
for spec in gw_specs:
    for fftbins in (True, False):
        for Nx in (10, 11):
            gw.append({'args': [F.enc(spec), F.enc(Nx), F.enc(fftbins)], 'kwargs': {},
                       'expect': F.enc_result(np.asarray(sg.get_window(spec, Nx, fftbins=fftbins)), []), 'compare': 'tol', 'tol': {'atol': 1e-12}})

# besselap: analog Bessel prototype (z, p, k); poles compared as an (re,im)-lexsorted set, norm 'phase' / 'delay'.
bsl = []
for norm in ('phase', 'delay'):
    for N in (1, 2, 3, 4, 5, 6, 8):
        z, p, kk = sg.besselap(N, norm=norm)
        bsl.append({'args': [F.enc(N), F.enc(norm)], 'kwargs': {},
                    'expect': F.enc_result((_sortc2(z), _sortc2(p), float(kk)), ['z', 'p', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-9}})

# ellipap: analog elliptic (Cauer) prototype (z, p, k); zeros/poles compared as (re,im)-lexsorted sets.
elp = []
for (N, rp, rs) in [(2, 1.0, 40.0), (3, 0.5, 60.0), (4, 2.0, 50.0), (5, 1.0, 80.0), (6, 0.1, 50.0), (7, 1.5, 45.0), (8, 0.5, 70.0)]:
    z, p, kk = sg.ellipap(N, rp, rs)
    elp.append({'args': [F.enc(N), F.enc(rp), F.enc(rs)], 'kwargs': {},
                'expect': F.enc_result((_sortc2(z), _sortc2(p), float(kk)), ['z', 'p', 'k']), 'compare': 'tol', 'tol': {'atol': 1e-9}})

out = {'module': 'signal', 'scipy': __import__('scipy').__version__, 'env': F.fixture_env.env(),
       'calls': [{'fn': 'convolve', 'cases': convs}, {'fn': 'lfilter', 'cases': lfs},
                 {'fn': 'fftconvolve', 'cases': fftc}, {'fn': 'oaconvolve', 'cases': oac},
                 {'fn': 'correlate', 'cases': corr},
                 {'fn': 'savgol_coeffs', 'cases': sgc}, {'fn': 'savgol_filter', 'cases': sgf},
                 {'fn': 'detrend', 'cases': dtr},
                 {'fn': 'lfilter_zi', 'cases': lfz}, {'fn': 'filtfilt', 'cases': ffc}, {'fn': 'sosfilt', 'cases': sfc},
                 {'fn': 'sosfilt_zi', 'cases': sfz}, {'fn': 'sosfiltfilt', 'cases': sffc},
                 # atol for freqz: H(e^jw) of a real filter has a legitimately ~0 component at many frequencies,
                 # where a relative tolerance is meaningless (same rationale as polyfit/roots in run-tests.php).
                 {'fn': 'freqz', 'cases': fqz, 'tol': {'atol': 1e-12}},
                 {'fn': 'zpk2tf', 'cases': z2tc, 'tol': {'atol': 1e-12}}, {'fn': 'butter', 'cases': butc, 'tol': {'atol': 1e-12}},
                 {'fn': 'cheby1', 'cases': cb1c, 'tol': {'atol': 1e-12}},
                 {'fn': 'cheby2', 'cases': cb2c, 'tol': {'atol': 1e-12}},
                 {'fn': 'ellip', 'cases': elc, 'tol': {'atol': 1e-12}},
                 {'fn': 'bessel', 'cases': bslc, 'tol': {'atol': 1e-12}},
                 {'fn': 'lfiltic', 'cases': lfic, 'tol': {'atol': 1e-12}},
                 {'fn': 'findfreqs', 'cases': ffq, 'tol': {'atol': 1e-12}},
                 {'fn': 'wiener', 'cases': wnr, 'tol': {'atol': 1e-11}},
                 {'fn': 'sweep_poly', 'cases': swp, 'tol': {'atol': 1e-11}},
                 {'fn': 'freqz_sos', 'cases': fzs, 'tol': {'atol': 1e-11}},
                 {'fn': 'max_len_seq', 'cases': mls},
                 {'fn': 'upfirdn', 'cases': ufd, 'tol': {'atol': 1e-11}},
                 {'fn': 'gammatone', 'cases': gmt, 'tol': {'atol': 1e-12}},
                 {'fn': 'check_COLA', 'cases': cola}, {'fn': 'check_NOLA', 'cases': nola},
                 {'fn': 'band_stop_obj', 'cases': bso, 'tol': {'atol': 1e-11}},
                 {'fn': 'freqresp', 'cases': frp, 'tol': {'atol': 1e-11}},
                 {'fn': 'square', 'cases': sqw}, {'fn': 'sawtooth', 'cases': saw},
                 {'fn': 'chirp', 'cases': chp, 'tol': {'atol': 1e-12}},
                 {'fn': 'gausspulse', 'cases': gps, 'tol': {'atol': 1e-12}},
                 {'fn': 'unit_impulse', 'cases': uimp},
                 {'fn': 'normalize', 'cases': nrm, 'tol': {'atol': 1e-12}},
                 {'fn': 'tf2zpk', 'cases': t2z, 'tol': {'atol': 1e-12}},
                 {'fn': 'sos2tf', 'cases': s2t, 'tol': {'atol': 1e-12}},
                 {'fn': 'tf2ss', 'cases': t2ss, 'tol': {'atol': 1e-11}},
                 {'fn': 'abcd_normalize', 'cases': abn, 'tol': {'atol': 1e-12}},
                 {'fn': 'ss2tf', 'cases': s2tf, 'tol': {'atol': 1e-10}},
                 {'fn': 'cont2discrete', 'cases': c2d, 'tol': {'atol': 1e-10}},
                 {'fn': 'zpk2ss', 'cases': z2ss, 'tol': {'atol': 1e-9}},
                 {'fn': 'ss2zpk', 'cases': s2zpk, 'tol': {'atol': 1e-9}},
                 {'fn': 'medfilt', 'cases': mdf, 'tol': {'atol': 1e-12}},
                 {'fn': 'medfilt2d', 'cases': mdf2, 'tol': {'atol': 1e-12}},
                 {'fn': 'order_filter', 'cases': ordf, 'tol': {'atol': 1e-12}},
                 {'fn': 'unique_roots', 'cases': uroo, 'tol': {'atol': 1e-9}},
                 {'fn': 'invres', 'cases': ivr, 'tol': {'atol': 1e-10}},
                 {'fn': 'invresz', 'cases': ivz, 'tol': {'atol': 1e-10}},
                 {'fn': 'buttord', 'cases': btord, 'tol': {'atol': 1e-9}},
                 {'fn': 'cheb1ord', 'cases': c1ord, 'tol': {'atol': 1e-9}},
                 {'fn': 'cheb2ord', 'cases': c2ord, 'tol': {'atol': 1e-9}},
                 {'fn': 'ellipord', 'cases': elord, 'tol': {'atol': 1e-9}},
                 {'fn': 'argrelmax', 'cases': armax},
                 {'fn': 'argrelmin', 'cases': armin},
                 {'fn': 'residue', 'cases': rsd, 'tol': {'atol': 1e-9}},
                 {'fn': 'residuez', 'cases': rsz, 'tol': {'atol': 1e-9}},
                 {'fn': 'lp2lp', 'cases': lplp, 'tol': {'atol': 1e-12}},
                 {'fn': 'lp2hp', 'cases': lphp, 'tol': {'atol': 1e-12}},
                 {'fn': 'lp2bp', 'cases': lpbp, 'tol': {'atol': 1e-12}},
                 {'fn': 'lp2bs', 'cases': lpbs, 'tol': {'atol': 1e-12}},
                 {'fn': 'bilinear', 'cases': blin, 'tol': {'atol': 1e-12}},
                 {'fn': 'freqs', 'cases': frqs, 'tol': {'atol': 1e-12}},
                 {'fn': 'freqs_zpk', 'cases': frzp, 'tol': {'atol': 1e-12}},
                 {'fn': 'freqz_zpk', 'cases': frzz, 'tol': {'atol': 1e-12}},
                 {'fn': 'sosfreqz', 'cases': sfz2, 'tol': {'atol': 1e-12}},
                 {'fn': 'group_delay', 'cases': gdl, 'tol': {'atol': 1e-9}},
                 {'fn': 'bode', 'cases': bodc, 'tol': {'atol': 1e-9}},
                 {'fn': 'dbode', 'cases': dbodc, 'tol': {'atol': 1e-9}},
                 {'fn': 'lsim', 'cases': lsimc, 'tol': {'atol': 1e-8}},
                 {'fn': 'impulse', 'cases': impc, 'tol': {'atol': 1e-8}},
                 {'fn': 'step', 'cases': stpc, 'tol': {'atol': 1e-8}},
                 {'fn': 'dlsim', 'cases': dlsc, 'tol': {'atol': 1e-9}},
                 {'fn': 'dimpulse', 'cases': dimc, 'tol': {'atol': 1e-9}},
                 {'fn': 'dstep', 'cases': dstc, 'tol': {'atol': 1e-9}},
                 {'fn': 'iirnotch', 'cases': notc, 'tol': {'atol': 1e-12}},
                 {'fn': 'iirpeak', 'cases': peakc, 'tol': {'atol': 1e-12}},
                 {'fn': 'iircomb', 'cases': combc, 'tol': {'atol': 1e-12}},
                 {'fn': 'hilbert', 'cases': hlb, 'tol': {'atol': 1e-10}},
                 {'fn': 'hilbert2', 'cases': hlb2, 'tol': {'atol': 1e-10}},
                 {'fn': 'periodogram', 'cases': pgm, 'tol': {'atol': 1e-12}},
                 {'fn': 'welch', 'cases': wel, 'tol': {'atol': 1e-12}},
                 {'fn': 'csd', 'cases': csdc, 'tol': {'atol': 1e-12}},
                 {'fn': 'coherence', 'cases': coh, 'tol': {'atol': 1e-12}},
                 {'fn': 'spectrogram', 'cases': spg, 'tol': {'atol': 1e-12}},
                 {'fn': 'correlation_lags', 'cases': clg},
                 {'fn': 'resample', 'cases': rsmp, 'tol': {'atol': 1e-10}},
                 {'fn': 'firwin', 'cases': fwn, 'tol': {'atol': 1e-12}},
                 {'fn': 'kaiser_atten', 'cases': katt}, {'fn': 'kaiser_beta', 'cases': kbeta},
                 {'fn': 'kaiserord', 'cases': kord},
                 {'fn': 'firwin2', 'cases': fw2, 'tol': {'atol': 1e-12}},
                 {'fn': 'sos2zpk', 'cases': s2z, 'tol': {'atol': 1e-10}},
                 {'fn': 'deconvolve', 'cases': dcv, 'tol': {'atol': 1e-12}},
                 {'fn': 'buttap', 'cases': bap, 'tol': {'atol': 1e-12}},
                 {'fn': 'lp2lp_zpk', 'cases': lplpz, 'tol': {'atol': 1e-10}},
                 {'fn': 'lp2hp_zpk', 'cases': lphpz, 'tol': {'atol': 1e-10}},
                 {'fn': 'bilinear_zpk', 'cases': blnz, 'tol': {'atol': 1e-10}},
                 {'fn': 'lp2bp_zpk', 'cases': lpbpz, 'tol': {'atol': 1e-10}},
                 {'fn': 'lp2bs_zpk', 'cases': lpbsz, 'tol': {'atol': 1e-10}},
                 {'fn': 'cheb1ap', 'cases': c1ap, 'tol': {'atol': 1e-11}},
                 {'fn': 'cheb2ap', 'cases': c2ap, 'tol': {'atol': 1e-11}},
                 {'fn': 'gauss_spline', 'cases': gsp, 'tol': {'atol': 1e-12}},
                 {'fn': 'vectorstrength', 'cases': vst, 'tol': {'atol': 1e-12}},
                 {'fn': 'convolve2d', 'cases': cv2, 'tol': {'atol': 1e-11}},
                 {'fn': 'correlate2d', 'cases': cr2, 'tol': {'atol': 1e-11}},
                 {'fn': 'lombscargle', 'cases': lsc, 'tol': {'atol': 1e-9}},
                 {'fn': 'get_window', 'cases': gw, 'tol': {'atol': 1e-12}},
                 {'fn': 'besselap', 'cases': bsl, 'tol': {'atol': 1e-9}},
                 {'fn': 'ellipap', 'cases': elp, 'tol': {'atol': 1e-9}}]}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'signal: convolve {len(convs)}, lfilter {len(lfs)}, fftconvolve {len(fftc)}, oaconvolve {len(oac)}, '
      f'correlate {len(corr)}, savgol_coeffs {len(sgc)}, savgol_filter {len(sgf)}, detrend {len(dtr)} cases')
