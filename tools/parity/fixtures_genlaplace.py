#!/usr/bin/env python3
# ============================================================================
# DELIVERABLE (2): fixtures for scipy.ndimage.generic_laplace
# ----------------------------------------------------------------------------
# Emits tests/fixtures/parity/ndimage_genlaplace.json with a 'genlaplace_cases'
# block. Each case names a derivative2 "kind" the PHP harness rebuilds as a PHP
# callable (kind='correlate1d' + weights -> Ndimage::correlate1d). The reference
# is scipy.ndimage.generic_laplace(input, derivative2, mode, cval) where
# derivative2 == correlate1d(..., weights, ...); for weights [1,-2,1] it must
# equal the builtin ndimage.laplace (asserted below), locking the canonical case.
#
# input/expect are encoded via fixtures_np.enc ({shape,dtype,data}) so they go
# through the standard harness decodeValue()/compareValue() path, identical to
# every other array-valued *_cases block.
#
# generic_laplace is pure correlate arithmetic (integer/float, no BLAS),
# host-independent, so this generator does not require the pinned environment.
# ============================================================================
import json, os
import numpy as np
import scipy.ndimage as nd
import fixtures_np as F  # enc / fixture_env

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   'tests', 'fixtures', 'parity', 'ndimage_genlaplace.json')
rng = np.random.default_rng(20261012)

cases = []


def d2_of(weights):
    """A derivative2 callable: correlate1d along one axis with the given 1-D weights
    (SciPy derivative2 signature: input, axis, output, mode, cval)."""
    w = list(weights)
    return lambda inp, axis, out, mode, cval, *a, **k: nd.correlate1d(inp, w, axis, out, mode, cval)


def case(img, weights, mode='reflect', cval=0.0):
    """Record one generic_laplace case. derivative2 is named by kind='correlate1d' + weights,
    which the PHP harness rebuilds as Ndimage::correlate1d(input, weights, axis, mode, cval)."""
    img = np.asarray(img, float)
    ref = np.asarray(nd.generic_laplace(img, d2_of(weights), mode=mode, cval=cval), float)
    cases.append({
        'fn': 'generic_laplace',
        'derivative2': {'kind': 'correlate1d', 'weights': list(map(float, weights))},
        'input': F.enc(img),
        'mode': mode,
        'cval': float(cval),
        'expect': F.enc(ref),
    })
    return ref


# ---- canonical weights [1,-2,1]: generic_laplace must equal the builtin laplace ----
LAP = [1.0, -2.0, 1.0]
x2 = rng.uniform(-5, 5, (6, 7))
x1 = rng.uniform(-5, 5, 9)
x3 = rng.uniform(-5, 5, (4, 5, 3))

builtin_checks = []
for img, mode, cval in [
    (x2, 'reflect', 0.0),
    (x2, 'nearest', 0.0),
    (x2, 'mirror', 0.0),
    (x2, 'wrap', 0.0),
    (x2, 'constant', 0.0),
    (x2, 'constant', 2.5),
    (x1, 'reflect', 0.0),           # 1-D
    (x3, 'reflect', 0.0),           # 3-D
    (x3, 'nearest', 0.0),
]:
    ref = case(img, LAP, mode=mode, cval=cval)
    lap = np.asarray(nd.laplace(img, mode=mode, cval=cval), float)
    err = float(np.max(np.abs(ref - lap)))
    builtin_checks.append(err)
    assert err == 0.0, f'generic_laplace([1,-2,1]) != laplace for mode={mode} cval={cval}: max err {err}'

# ---- non-canonical second-derivative weights (still a valid derivative2) ----
# wider 5-tap second differences; exercise the per-axis-sum path with other weights
case(x2, [1.0, -1.0, 0.0, -1.0, 1.0], mode='reflect')
case(x2, [0.5, 0.0, -1.0, 0.0, 0.5], mode='nearest')

out = {
    'module': 'ndimage',
    'scipy': __import__('scipy').__version__,
    'env': F.fixture_env.env(),
    'genlaplace_cases': cases,
}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'ndimage_genlaplace: {len(cases)} cases, builtin-laplace max err = {max(builtin_checks)} (expect 0.0)')
