#!/usr/bin/env python3
"""scipy.signal.CZT and scipy.signal.ZoomFFT callable transform objects.

The pure-PHP Tessero\\Signal\\Czt and Tessero\\Signal\\ZoomFft are thin deterministic containers over
the already-landed czt / zoom_fft primitives: CZT(n, m, w, a)(x) is defined by scipy as
czt(x, m, w, a), and ZoomFFT(n, fn, m, fs=fs, endpoint=endpoint)(x) as zoom_fft(x, fn, m, fs=fs,
endpoint=endpoint). Because the class simply stores the parameters and forwards to the free function,
the results are bit-identical; we record them here at rtol 1e-12 against scipy 1.17.1.

All inputs are embedded, so the recorded expectation is host-independent.

Writes tests/fixtures/parity/signal_classes.json with:
  czt_cases:  {n, m, w:[re,im]|null, a:[re,im], x:[[re,im],...], expect:[[re,im],...], rtol}
  zoomfft_cases: {n, fn:[...]|scalar, m, fs, endpoint, x:[...], expect:[[re,im],...], rtol}
"""
import json
import os

import numpy as np
from scipy.signal import CZT, ZoomFFT

OUT = os.environ.get('SIGNALCLASSES_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'signal_classes.json')

rng = np.random.default_rng(20240614)


def c2(z):
    z = complex(z)
    return [float(z.real), float(z.imag)]


def carr(a):
    return [c2(v) for v in np.asarray(a, complex).ravel()]


czt_cases = []


def add_czt(n, m=None, w=None, a=1, complex_input=False, rtol=1e-12):
    if complex_input:
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    else:
        x = rng.standard_normal(n)
    t = CZT(n, m, w, a)
    y = t(x)
    czt_cases.append({
        'n': int(n),
        'm': None if m is None else int(m),
        'w': None if w is None else c2(w),
        'a': c2(a),
        'x': carr(x),
        'expect': carr(y),
        'rtol': rtol,
    })


add_czt(8)                                                   # defaults: m=n, full unit circle
add_czt(8, m=5)                                              # fewer output points
add_czt(6, m=8)                                              # more output points than input
add_czt(10, m=10, w=np.exp(-1j * 0.1), a=0.9 * np.exp(1j * 0.05))  # spiral contour
add_czt(6, m=8, a=1.2)                                       # real start point != 1
add_czt(8, complex_input=True)                               # complex input signal
add_czt(12, m=16, w=np.exp(-2j * np.pi / 20), a=1.0)         # narrow-band zoom-like


zoomfft_cases = []


def add_zoom(n, fn, m=None, fs=2, endpoint=False, rtol=1e-12):
    x = rng.standard_normal(n)
    t = ZoomFFT(n, fn, m, fs=fs, endpoint=endpoint)
    y = t(x)
    zoomfft_cases.append({
        'n': int(n),
        'fn': fn,
        'm': None if m is None else int(m),
        'fs': float(fs),
        'endpoint': bool(endpoint),
        'x': [float(v) for v in x],
        'expect': carr(y),
        'rtol': rtol,
    })


add_zoom(16, [0.1, 0.4])                                     # band, default m, fs=2
add_zoom(16, [1.0, 3.0], m=20, fs=8.0)                       # band with real fs
add_zoom(12, 0.5, m=12)                                      # scalar fn -> [0, fn]
add_zoom(16, [0.2, 0.6], m=32, fs=2, endpoint=True)          # endpoint inclusive
add_zoom(20, [0.05, 0.25], m=25, fs=4.0)                     # finer band


os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump({'czt_cases': czt_cases, 'zoomfft_cases': zoomfft_cases}, f,
              allow_nan=False, indent=1)
print(f'wrote {len(czt_cases)} czt + {len(zoomfft_cases)} zoomfft cases to {OUT}')
