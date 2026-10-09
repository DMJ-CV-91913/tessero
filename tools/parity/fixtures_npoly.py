#!/usr/bin/env python3
"""numpy.polynomial fixtures: evaluation of the six basis classes and their *val functions. Deterministic
Clenshaw arithmetic; generated in the pinned environment for consistency with the other groups."""
import json, os
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev, Legendre, Laguerre, Hermite, HermiteE
import numpy.polynomial.polynomial as _P
import numpy.polynomial.chebyshev as _C
import numpy.polynomial.legendre as _L
import numpy.polynomial.laguerre as _LA
import numpy.polynomial.hermite as _H
import numpy.polynomial.hermite_e as _HE
import fixtures_np as F  # enc / enc_result / fixture_env

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tests', 'fixtures', 'parity', 'npoly.json')
rng = np.random.default_rng(20261009)

cls = {n: [] for n in ('Polynomial', 'Chebyshev', 'Legendre', 'Laguerre', 'Hermite', 'HermiteE')}
val = {n: [] for n in ('polyval', 'chebval', 'legval', 'lagval', 'hermval', 'hermeval')}
CLASSES = {'Polynomial': Polynomial, 'Chebyshev': Chebyshev, 'Legendre': Legendre,
           'Laguerre': Laguerre, 'Hermite': Hermite, 'HermiteE': HermiteE}
VALS = {'polyval': _P.polyval, 'chebval': _C.chebval, 'legval': _L.legval,
        'lagval': _LA.lagval, 'hermval': _H.hermval, 'hermeval': _HE.hermeval}

for _ in range(4):
    c = rng.normal(size=int(rng.integers(1, 7)))
    x = rng.uniform(-1.5, 1.5, 9)
    for name, C in CLASSES.items():
        cls[name].append({'args': [F.enc(c), F.enc(x)], 'kwargs': {},
                          'expect': F.enc_result(np.asarray(C(c)(x), float), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})
    for name, fn in VALS.items():
        val[name].append({'args': [F.enc(x), F.enc(c)], 'kwargs': {},
                          'expect': F.enc_result(np.asarray(fn(x, c), float), []), 'compare': 'tol', 'tol': {'atol': 1e-9}})

calls = [{'fn': k, 'cases': v} for k, v in {**cls, **val}.items()]
out = {'module': 'npoly', 'numpy': np.__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'npoly: {len(CLASSES)} classes + {len(VALS)} val functions')
