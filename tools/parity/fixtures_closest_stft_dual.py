#!/usr/bin/env python3
r"""SciPy reference fixtures for scipy.signal.closest_STFT_dual_window
(pure-PHP Tessero\Signal\ClosestStftDualWindow).

    python3 tools/parity/fixtures_closest_stft_dual.py

Writes tests/fixtures/parity/signal_closest_stft_dual.json with a
`closest_stft_dual_cases` list. Each case records the window, hop, optional
desired dual window, and the `scaled` flag; `expect` is {dual_win, alpha} from
scipy.signal.closest_STFT_dual_window. Complex entries are [re, im] pairs. The
run-tests `closest_stft_dual_cases` arm checks both backends.

Exactness note
--------------
This is a self-contained least-squares computation over a window and its
canonical dual window; it reproduces scipy to machine precision (worst
relative error ~4e-16 on non-trivial dual entries). It does NOT use the
mutable/stateful ShortTimeFFT machinery (in SciPy 1.17.1 it is a free function
`closest_STFT_dual_window(win, hop, desired_dual, *, scaled)`).
"""
import json
import os

import numpy as np
import scipy
from scipy.signal import closest_STFT_dual_window, windows

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'fixtures', 'parity')


def _toc(a):
    a = np.asarray(a)
    if np.iscomplexobj(a):
        return [[float(x.real), float(x.imag)] for x in a]
    return [float(x) for x in a]


def cases():
    out = []

    def add(win, hop, desired, scaled, name):
        dw, alpha = closest_STFT_dual_window(win, hop, desired, scaled=scaled)
        cplx = np.iscomplexobj(np.asarray(win)) or (
            desired is not None and np.iscomplexobj(np.asarray(desired)))
        al = [float(np.real(alpha)), float(np.imag(alpha))] if cplx else float(np.real(alpha))
        out.append({
            'name': name,
            'win': _toc(win),
            'hop': int(hop),
            'desired_dual': (_toc(desired) if desired is not None else None),
            'scaled': bool(scaled),
            'expect': {'dual_win': _toc(dw), 'alpha': al},
            'rtol': 1e-9,
            'atol': 1e-11,
        })

    rng = np.random.default_rng(7)
    for name, win in [('hann12', windows.hann(12, sym=False)),
                      ('gauss16', windows.gaussian(16, std=3.0, sym=False)),
                      ('hamming10', windows.hamming(10, sym=False)),
                      ('bartlett8', windows.bartlett(8, sym=False))]:
        for hop in sorted({1, 2, max(1, len(win) // 4), len(win) // 2}):
            for scaled in (True, False):
                add(win, hop, None, scaled, f'{name}_h{hop}_s{int(scaled)}')

    w = windows.hann(10, sym=False)
    add(w, 5, np.ones(10) * 0.5, True, 'explicit_half')
    add(w, 2, np.linspace(0.5, 1.5, 10), False, 'explicit_ramp_unscaled')

    w = rng.normal(1.0, 0.2, 14)
    add(w, 3, None, True, 'rand_real')

    wc = windows.hann(12, sym=False) * np.exp(1j * np.linspace(0, 1, 12))
    add(wc, 3, None, True, 'complex_scaled')
    add(wc, 4, None, False, 'complex_unscaled')
    add(wc, 3, np.ones(12, dtype=complex) * (0.5 + 0.1j), True, 'complex_desired')

    return out


if __name__ == '__main__':
    data = {'module': 'signal_closest_stft_dual', 'scipy': scipy.__version__,
            'closest_stft_dual_cases': cases()}
    path = os.path.join(OUTDIR, 'signal_closest_stft_dual.json')
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'), allow_nan=False)
    print('wrote', path, 'cases:', len(data['closest_stft_dual_cases']))
