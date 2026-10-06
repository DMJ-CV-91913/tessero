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
                 {'fn': 'square', 'cases': sqw}, {'fn': 'sawtooth', 'cases': saw},
                 {'fn': 'chirp', 'cases': chp, 'tol': {'atol': 1e-12}},
                 {'fn': 'gausspulse', 'cases': gps, 'tol': {'atol': 1e-12}},
                 {'fn': 'unit_impulse', 'cases': uimp},
                 {'fn': 'normalize', 'cases': nrm, 'tol': {'atol': 1e-12}},
                 {'fn': 'tf2zpk', 'cases': t2z, 'tol': {'atol': 1e-12}},
                 {'fn': 'sos2tf', 'cases': s2t, 'tol': {'atol': 1e-12}},
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
                 {'fn': 'hilbert', 'cases': hlb, 'tol': {'atol': 1e-10}},
                 {'fn': 'hilbert2', 'cases': hlb2, 'tol': {'atol': 1e-10}},
                 {'fn': 'periodogram', 'cases': pgm, 'tol': {'atol': 1e-12}},
                 {'fn': 'welch', 'cases': wel, 'tol': {'atol': 1e-12}},
                 {'fn': 'csd', 'cases': csdc, 'tol': {'atol': 1e-12}},
                 {'fn': 'coherence', 'cases': coh, 'tol': {'atol': 1e-12}},
                 {'fn': 'spectrogram', 'cases': spg, 'tol': {'atol': 1e-12}}]}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'signal: convolve {len(convs)}, lfilter {len(lfs)}, fftconvolve {len(fftc)}, oaconvolve {len(oac)}, '
      f'correlate {len(corr)}, savgol_coeffs {len(sgc)}, savgol_filter {len(sgf)}, detrend {len(dtr)} cases')
