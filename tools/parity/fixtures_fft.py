#!/usr/bin/env python3
"""numpy.fft / scipy.fft fixtures (module 'fft' -> Tessero\\Fft\\Fft and Tessero\\Ext\\Fft\\Fft). Array in, array
out, compared against numpy.fft. FFT values are host-independent (pocketfft-style), so no pinned env is needed."""
import json, os
import numpy as np
import numpy.fft as nf
import scipy.fft as sf
import fixtures_np as F  # enc (handles complex) / enc_result

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'fft.json')
rng = np.random.default_rng(7)


def call(fn, args, kwargs=None, mod=nf):
    kwargs = kwargs or {}
    r = getattr(mod, fn)(*args, **kwargs)
    return {'args': [F.enc(a) for a in args], 'kwargs': {k: F.enc(v) for k, v in kwargs.items()},
            'expect': F.enc_result(r, []), 'compare': 'tol'}


def scall(fn, args, kwargs=None):   # reference from scipy.fft (hfft2/hfftn/... are scipy-only)
    return call(fn, args, kwargs, mod=sf)


c8 = rng.standard_normal(8) + 1j * rng.standard_normal(8)
c6 = rng.standard_normal(6) + 1j * rng.standard_normal(6)
r8 = rng.standard_normal(8)
r7 = rng.standard_normal(7)
m44 = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
m34 = rng.standard_normal((3, 4))
m234 = rng.standard_normal((2, 3, 4)) + 1j * rng.standard_normal((2, 3, 4))
r234 = rng.standard_normal((2, 3, 4))
r24 = rng.standard_normal((2, 4))

blocks = {
    'fft': [call('fft', [c8]), call('fft', [r8]), call('fft', [c8], {'norm': 'ortho'}), call('fft', [c6], {'norm': 'forward'}),
            call('fft', [m34], {'axis': 0}), call('fft', [r7])],
    'ifft': [call('ifft', [c8]), call('ifft', [c6], {'norm': 'ortho'}), call('ifft', [nf.fft(r8)])],
    'rfft': [call('rfft', [r8]), call('rfft', [r7]), call('rfft', [r8], {'norm': 'ortho'}), call('rfft', [r8], {'norm': 'forward'})],
    'irfft': [call('irfft', [nf.rfft(r8)]), call('irfft', [nf.rfft(r7)], {'n': 7}), call('irfft', [nf.rfft(r8)], {'norm': 'ortho'})],
    'fft2': [call('fft2', [m44]), call('fft2', [m34 + 0j])],
    'ifft2': [call('ifft2', [m44])],
    'fftn': [call('fftn', [m44]), call('fftn', [m234]), call('fftn', [m44], {'norm': 'ortho'}),
             call('fftn', [m234], {'axes': [0, 2]}), call('fftn', [m234], {'norm': 'forward'})],
    'ifftn': [call('ifftn', [m44]), call('ifftn', [m234], {'norm': 'forward'}), call('ifftn', [nf.fftn(r234)])],
    'rfftn': [call('rfftn', [m34]), call('rfftn', [r234]), call('rfftn', [r234], {'axes': [0, 2]}),
              call('rfftn', [m34], {'norm': 'ortho'})],
    'irfftn': [call('irfftn', [nf.rfftn(m34)], {'s': [3, 4]}), call('irfftn', [nf.rfftn(r234)], {'s': [2, 3, 4]}),
               call('irfftn', [nf.rfftn(m34)], {'s': [3, 4], 'norm': 'ortho'})],
    'rfft2': [call('rfft2', [m34]), call('rfft2', [r24], {'norm': 'ortho'}), call('rfft2', [r234], {'axes': [1, 2]})],
    'irfft2': [call('irfft2', [nf.rfft2(m34)], {'s': [3, 4]})],
    'hfft': [call('hfft', [c6]), call('hfft', [c8], {'n': 14}), call('hfft', [c6], {'norm': 'ortho'}),
             call('hfft', [c8], {'norm': 'forward'})],
    'ihfft': [call('ihfft', [r8]), call('ihfft', [r7]), call('ihfft', [r8], {'norm': 'ortho'})],
    'hfftn': [scall('hfftn', [m44]), scall('hfftn', [m234], {'norm': 'ortho'}), scall('hfftn', [m234], {'axes': [0, 2]})],
    'hfft2': [scall('hfft2', [m44]), scall('hfft2', [m234], {'norm': 'forward'})],
    'ihfftn': [scall('ihfftn', [r234]), scall('ihfftn', [m34], {'norm': 'ortho'})],
    'ihfft2': [scall('ihfft2', [m34]), scall('ihfft2', [r234], {'axes': [1, 2]})],
    'next_fast_len': [scall('next_fast_len', [n]) for n in [1, 7, 11, 13, 22, 100, 143, 169]]
                     + [scall('next_fast_len', [n], {'real': True}) for n in [7, 11, 13, 25, 49]],
    'prev_fast_len': [scall('prev_fast_len', [n]) for n in [7, 11, 13, 100, 143, 170]]
                     + [scall('prev_fast_len', [n], {'real': True}) for n in [7, 11, 13, 26, 49]],
    'fftfreq': [call('fftfreq', [8]), call('fftfreq', [9], {'d': 0.5}), call('fftfreq', [10], {'d': 2.0})],
    'rfftfreq': [call('rfftfreq', [8]), call('rfftfreq', [9], {'d': 0.5})],
    'fftshift': [call('fftshift', [np.arange(8.0)]), call('fftshift', [np.arange(9.0)]), call('fftshift', [m34])],
    'ifftshift': [call('ifftshift', [np.arange(8.0)]), call('ifftshift', [np.arange(9.0)])],
}

out = {'module': 'fft', 'numpy': np.__version__,
       'calls': [{'fn': k, 'cases': v, 'tol': {'rtol': 1e-9, 'atol': 1e-11}} for k, v in blocks.items()]}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'fft: {len(blocks)} functions, {sum(len(v) for v in blocks.values())} cases')
