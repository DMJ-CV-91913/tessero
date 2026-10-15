#!/usr/bin/env python3
"""Composed-pipeline ("scenario") parity fixtures: realistic MULTI-CALL NumPy/SciPy workflows.

Unlike the per-function fixtures (one function, one call), each case here chains several operations the
way real library code does, and records the final result plus a few checkpoints. The Tessero mirror of
each pipeline (tools/parity/run-tests.php `scenario_cases`) must reproduce every checkpoint on BOTH
backends at machine precision. This catches *composition* bugs - dtype propagation, broadcasting across
steps, copy-vs-view, accumulation order - that isolated per-function tests can miss.

These are supplementary end-to-end checks; they do not map to census symbols and do not change coverage %.
"""
import json
import os

import numpy as np
from scipy.signal import fftconvolve

OUT = os.environ.get('SCENARIOS_OUT') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'tests', 'fixtures', 'scenarios.json')  # deliberately NOT under tests/fixtures/parity/ (not run-tests-globbed)

cases = []


def rnd(seed, *shape):
    return np.random.default_rng(seed).standard_normal(shape)


# 1. Linear solve + residual (A x = b on a well-conditioned SPD system)
M = rnd(1, 50, 50)
A = M @ M.T + 50.0 * np.eye(50)
b = rnd(2, 50)
x = np.linalg.solve(A, b)
cases.append({
    'name': 'linalg_solve_residual',
    'inputs': {'A': A.tolist(), 'b': b.tolist()},
    'expected': {
        'x_sum': float(x.sum()),
        'residual': float(np.linalg.norm(A @ x - b)),
        'x_head': x[:5].tolist(),
    },
})

# 2. Standardize -> covariance -> eigenvalues (PCA-style)
X = rnd(3, 200, 6) @ (rnd(4, 6, 6) * 0.3 + np.eye(6))
Xc = X - X.mean(axis=0)
Xs = Xc / X.std(axis=0)
C = np.cov(Xs, rowvar=False)
w = np.sort(np.linalg.eigvalsh(C))
cases.append({
    'name': 'pca_eigvals',
    'inputs': {'X': X.tolist()},
    'expected': {
        'eigvals': w.tolist(),
        'trace': float(np.trace(C)),
        'col_means_abs_max': float(np.abs(Xc.mean(axis=0)).max()),
    },
})

# 3. Polynomial fit -> evaluate -> RMSE
xp = np.linspace(-3.0, 3.0, 80)
yp = 2.0 * xp**3 - xp + 1.0 + 0.1 * rnd(5, 80)
coef = np.polyfit(xp, yp, 3)
yhat = np.polyval(coef, xp)
rmse = float(np.sqrt(np.mean((yhat - yp) ** 2)))
cases.append({
    'name': 'polyfit_rmse',
    'inputs': {'x': xp.tolist(), 'y': yp.tolist()},
    'expected': {'coef': coef.tolist(), 'rmse': rmse},
})

# 4. Reduction chain (deterministic inputs): arange -> cumsum -> diff -> mean
a = np.arange(1.0, 201.0)
bb = np.cumsum(a)
dd = np.diff(bb)
cases.append({
    'name': 'reduction_chain',
    'inputs': {},
    'expected': {
        'sum_a': float(a.sum()),
        'max_cumsum': float(bb.max()),
        'mean_diff': float(dd.mean()),
    },
})

# 5. Transcendental elementwise chain (deterministic): linspace -> sin*cos -> reduce
xt = np.linspace(0.0, 2.0 * np.pi, 1000)
yt = np.sin(xt) * np.cos(2.0 * xt) + 0.5
cases.append({
    'name': 'transcendental_chain',
    'inputs': {},
    'expected': {
        'sum': float(yt.sum()),
        'max': float(yt.max()),
        'argmax': int(np.argmax(yt)),
    },
})

# 6. Overdetermined least squares -> residual
Al = rnd(6, 80, 5)
bl = rnd(7, 80)
xl, *_ = np.linalg.lstsq(Al, bl, rcond=None)
cases.append({
    'name': 'lstsq_residual',
    'inputs': {'A': Al.tolist(), 'b': bl.tolist()},
    'expected': {
        'x': xl.tolist(),
        'residual': float(np.linalg.norm(Al @ xl - bl)),
    },
})

# 7. FFT convolution (signal) -> reduce
sig = rnd(8, 128)
ker = rnd(9, 32)
conv = fftconvolve(sig, ker, mode='full')
cases.append({
    'name': 'fftconvolve_signal',
    'inputs': {'a': sig.tolist(), 'v': ker.tolist()},
    'expected': {
        'len': int(conv.size),
        'sum': float(conv.sum()),
        'argmax_abs': int(np.argmax(np.abs(conv))),
        'head': conv[:5].tolist(),
    },
})

data = {'scenario_cases': cases}
with open(OUT, 'w') as f:
    json.dump(data, f)
print(f"wrote {OUT}: {len(cases)} scenarios -> " + ', '.join(c['name'] for c in cases))
