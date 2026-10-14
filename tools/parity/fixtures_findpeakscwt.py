#!/usr/bin/env python3
"""scipy.signal.find_peaks_cwt continuous-wavelet-transform peak detection.

The pure-PHP Tessero\\Signal\\FindPeaksCwt reproduces scipy.signal.find_peaks_cwt: it builds the CWT
matrix by convolving the signal with the (reversed) Ricker wavelet at each width (direct convolution,
mode='same'), identifies ridge lines across scales, filters them by length and SNR
(scoreatpercentile noise floor), and returns the sorted integer peak indices. The pipeline is fully
deterministic, so the recorded scipy expectation is host-independent -- every signal is serialized as
concrete float literals in the fixture (the seeds below only build those literals at generation time).

Writes tests/fixtures/parity/signal_findpeakscwt.json with a `findpeakscwt_cases` block:
  {fn:'find_peaks_cwt', vector:[...], widths:[...], min_snr, noise_perc, expect:[int indices]}
The comparison is exact integer-array equality (no rtol): find_peaks_cwt returns indices only.
"""
import json
import os

import numpy as np
import scipy
from scipy.signal import find_peaks_cwt

OUT = os.environ.get('FINDPEAKSCWT_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'parity', 'signal_findpeakscwt.json')

cases = []


def add(vector, widths, min_snr=1.0, noise_perc=10.0):
    v = np.asarray(vector, dtype=float)
    w = np.asarray(widths, dtype=float)
    res = find_peaks_cwt(v, w, min_snr=min_snr, noise_perc=noise_perc)
    cases.append({
        'fn': 'find_peaks_cwt',
        'vector': [float(x) for x in v],
        'widths': [float(x) for x in w],
        'min_snr': float(min_snr),
        'noise_perc': float(noise_perc),
        'expect': [int(i) for i in res],
    })


# Case 1: the scipy docstring example (a single sine half-wave -> one peak).
xs = np.arange(0, np.pi, 0.05)
add(np.sin(xs), np.arange(1, 10))

# Case 2: noisy multi-sine (fixed seed), broad width range -> ~10 peaks.
rng = np.random.default_rng(42)
t = np.linspace(0, 6 * np.pi, 240)
add(np.sin(t) + 0.5 * np.sin(3 * t) + 0.15 * rng.standard_normal(240), np.arange(1, 16))

# Case 3: three Gaussian bumps of different widths.
x = np.linspace(0, 100, 300)
g = sum(np.exp(-((x - c) ** 2) / (2 * wd ** 2)) for c, wd in [(20, 3), (50, 5), (80, 2)])
add(g, np.arange(1, 20))

# Case 4: sharp bumps in noise, custom min_snr / noise_perc.
rng = np.random.default_rng(7)
sig = np.zeros(200)
for c in (30, 70, 120, 160):
    sig += 2 * np.exp(-((np.arange(200) - c) ** 2) / (2 * 4 ** 2))
sig += 0.3 * rng.standard_normal(200)
add(sig, np.arange(1, 12), min_snr=1.5, noise_perc=20.0)

# Case 5: short signal, small widths (a dense comb).
add([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0, 4, 0, 1, 0], np.arange(1, 4))

# Case 6: flat (no peaks) and Case 7: monotone ramp (no interior maxima).
add(np.zeros(50) + 1.0, np.arange(1, 8))
add(np.arange(40.0), np.arange(1, 6))

# Case 8: regular comb with light noise, wide width range (ridge tie-breaking).
rng = np.random.default_rng(99)
comb = np.zeros(160)
for c in range(10, 160, 15):
    comb[c] = 1.0
add(comb + 0.05 * rng.standard_normal(160), np.arange(1, 25))

# Case 9: brownian-walk signal (many local maxima), even widths.
rng = np.random.default_rng(3)
add(np.cumsum(rng.standard_normal(120)), np.arange(2, 30, 2))

# Case 10: wide single bump plus high-frequency ripple, noise_perc=50.
x = np.linspace(-10, 10, 180)
add(np.exp(-x ** 2 / 8) + 0.01 * np.sin(50 * x), np.arange(1, 18), noise_perc=50.0)

out = {'module': 'signal', 'scipy': scipy.__version__, 'findpeakscwt_cases': cases}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'signal_findpeakscwt: {len(cases)} cases -> {OUT}')
