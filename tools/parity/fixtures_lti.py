#!/usr/bin/env python3
r"""scipy.signal LTI system classes + the modern ShortTimeFFT.

Covers the pure-PHP Tessero\Signal facades:
  - lti / dlti factory base classes (dispatch on arg count),
  - StateSpace / TransferFunction / ZerosPolesGain (the three representations),
  - ShortTimeFFT (stft / istft / spectrogram, fft_mode in {onesided, twosided}).

Every numeric result is produced by the already-landed Tessero\Signal conversion
and response primitives (tf2ss / ss2tf / zpk2tf / tf2zpk / zpk2ss / ss2zpk,
impulse / step / bode / dimpulse / dstep / dbode) plus Tessero\Fft, so both
backends agree exactly.

Comparison basis
----------------
State-space realizations are NOT unique (any similarity transform yields the
same transfer function), so StateSpace equality is tested through the IMPULSE
and STEP response (invariants) and through the tf/zpk conversions, never the
raw A/B/C/D matrices. Poles/zeros are compared as sorted complex multisets.
The STFT is compared element-wise (deterministic FFT math), together with an
istft round-trip and the spectrogram.

Writes tests/fixtures/parity/signal_lti.json with three case blocks:
  lti_cases, dlti_cases, stft_cases.
"""
import json
import os

import numpy as np
import scipy
import scipy.signal as sig
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures", "parity", "signal_lti.json")


def cx(a):
    """Flatten an array to a list of floats, or of [re, im] pairs when complex."""
    a = np.asarray(a)
    if np.iscomplexobj(a):
        return [[float(v.real), float(v.imag)] for v in a.ravel()]
    return [float(v) for v in a.ravel()]


def lti_case(kind, args):
    s = sig.lti(*args)
    tf = s.to_tf()
    zpk = s.to_zpk()
    T = list(np.linspace(0.0, 5.0, 12))
    w = list(np.logspace(-1.0, 1.0, 8))
    _, yi = sig.impulse((tf.num, tf.den), T=T)
    _, ys = sig.step((tf.num, tf.den), T=T)
    _, mag, ph = sig.bode((tf.num, tf.den), w=w)
    return {
        "kind": kind,
        "args": [np.asarray(a).tolist() if not np.isscalar(a) else float(a) for a in args],
        "tf_num": cx(tf.num),
        "tf_den": cx(tf.den),
        "zpk_z": cx(zpk.zeros),
        "zpk_p": cx(zpk.poles),
        "zpk_k": float(zpk.gain),
        "poles": cx(s.poles),
        "zeros": cx(s.zeros),
        "T": [float(t) for t in T],
        "impulse": cx(yi),
        "step": cx(ys),
        "w": [float(x) for x in w],
        "mag": cx(mag),
        "phase": cx(ph),
    }


def dlti_case(kind, args, dt, n=12):
    d = sig.dlti(*args, dt=dt)
    tf = d.to_tf()
    _, yi = sig.dimpulse((tf.num, tf.den, dt), n=n)
    _, ys = sig.dstep((tf.num, tf.den, dt), n=n)
    return {
        "kind": kind,
        "args": [np.asarray(a).tolist() if not np.isscalar(a) else float(a) for a in args],
        "dt": float(dt),
        "poles": cx(d.poles),
        "zeros": cx(d.zeros),
        "n": n,
        "dimpulse": cx(np.asarray(yi).ravel()),
        "dstep": cx(np.asarray(ys).ravel()),
    }


def stft_case(win, hop, fs, mfft, mode, n=60):
    SFT = ShortTimeFFT(np.asarray(win, dtype=float), hop=hop, fs=fs, mfft=mfft, fft_mode=mode)
    t = np.arange(n)
    x = np.sin(2 * np.pi * 5 * t / fs) + 0.3 * np.cos(2 * np.pi * 11 * t / fs)
    Sx = SFT.stft(x)
    xr = SFT.istft(Sx, k1=n)
    Px = SFT.spectrogram(x)
    return {
        "win": [float(v) for v in win],
        "hop": int(hop),
        "fs": float(fs),
        "mfft": (None if mfft is None else int(mfft)),
        "fft_mode": mode,
        "x": [float(v) for v in x],
        "f_pts": int(SFT.f_pts),
        "delta_t": float(SFT.delta_t),
        "delta_f": float(SFT.delta_f),
        "p_min": int(SFT.p_min),
        "p_max": int(SFT.p_max(n)),
        "dual_win": [float(v) for v in SFT.dual_win],
        "f": [float(v) for v in SFT.f],
        "Sx_shape": list(Sx.shape),
        "Sx": [[[float(v.real), float(v.imag)] for v in row] for row in Sx],
        "istft": [float(v) for v in xr],
        "spectrogram": [[float(v) for v in row] for row in Px],
    }


lti_cases = [
    lti_case("tf", ([1.0, 3.0, 3.0], [1.0, 2.0, 1.0])),
    lti_case("tf", ([2.0], [1.0, 0.5, 0.25])),
    lti_case("tf", ([1.0, 0.0], [1.0, 0.4, 1.0])),
    lti_case("zpk", ([-1.0], [-2.0, -3.0], 5.0)),
    lti_case("zpk", ([-0.5, -4.0], [-1.0, -2.0, -3.0], 2.0)),
    lti_case("ss", ([[-2.0, -1.0], [1.0, 0.0]], [[1.0], [0.0]], [[1.0, 2.0]], [[1.0]])),
    lti_case("ss", ([[-0.5]], [[1.0]], [[2.0]], [[0.0]])),
]

dlti_cases = [
    dlti_case("tf", ([1.0, -0.5], [1.0, -0.3, 0.02]), 0.1),
    dlti_case("zpk", ([0.5], [0.2, 0.1], 1.0), 0.2),
    dlti_case("tf", ([0.2, 0.1], [1.0, -0.8, 0.15]), 0.05),
]

stft_cases = [
    stft_case(hann(16, sym=True).tolist(), 4, 50.0, None, "onesided"),
    stft_case(hann(32, sym=True).tolist(), 8, 100.0, None, "onesided"),
    stft_case(hann(16, sym=True).tolist(), 4, 50.0, 24, "onesided"),
    stft_case(hann(16, sym=True).tolist(), 8, 50.0, None, "onesided"),
    stft_case(hann(16, sym=True).tolist(), 4, 50.0, None, "twosided"),
]

out = {
    "module": "signal",
    "scipy": scipy.__version__,
    "lti_cases": lti_cases,
    "dlti_cases": dlti_cases,
    "stft_cases": stft_cases,
}

with open(OUT, "w") as f:
    json.dump(out, f, separators=(",", ":"), allow_nan=False)
print(f"signal_lti: {len(lti_cases)} lti + {len(dlti_cases)} dlti + {len(stft_cases)} stft -> {OUT}")
