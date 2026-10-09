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

def tol(atol=1e-9):
    return {'atol': atol}

def case(args, expect, names=None, atol=1e-9):
    return {'args': [F.enc(a) for a in args], 'kwargs': {},
            'expect': F.enc_result(expect, names or []), 'compare': 'tol', 'tol': tol(atol)}

# numpy.polynomial.polynomial power-basis arithmetic and calculus
ops = {n: [] for n in ('polyadd', 'polysub', 'polymul', 'polymulx', 'polypow', 'polydiv', 'polyder',
                        'polyint', 'polyfromroots', 'polyline', 'polytrim', 'polyvander', 'polyvalfromroots')}
for _ in range(6):
    c1 = rng.normal(size=int(rng.integers(2, 6)))
    c2 = rng.normal(size=int(rng.integers(2, 5)))
    x = rng.uniform(-1.5, 1.5, 7)
    roots = rng.uniform(-1.2, 1.2, int(rng.integers(1, 5)))
    off, scl = float(rng.normal()), float(rng.normal())
    ops['polyadd'].append(case([c1, c2], _P.polyadd(c1, c2)))
    ops['polysub'].append(case([c1, c2], _P.polysub(c1, c2)))
    ops['polymul'].append(case([c1, c2], _P.polymul(c1, c2)))
    ops['polymulx'].append(case([c1], _P.polymulx(c1)))
    pw = int(rng.integers(0, 5))
    ops['polypow'].append(case([c1, pw], _P.polypow(c1, pw)))
    cbig = rng.normal(size=6)
    ops['polydiv'].append(case([cbig, c2], _P.polydiv(cbig, c2), names=['quo', 'rem']))
    m = int(rng.integers(1, 3))
    ops['polyder'].append(case([c1, m], _P.polyder(c1, m)))
    ops['polyint'].append(case([c1, m], _P.polyint(c1, m)))
    ops['polyfromroots'].append(case([roots], _P.polyfromroots(roots)))
    ops['polyline'].append(case([off, scl], _P.polyline(off, scl)))
    ct = np.append(c1, [1e-14, 0.0])
    ops['polytrim'].append(case([ct, 1e-9], _P.polytrim(ct, 1e-9)))
    deg = int(rng.integers(1, 5))
    ops['polyvander'].append(case([x, deg], _P.polyvander(x, deg)))
    ops['polyvalfromroots'].append(case([x, roots], _P.polyvalfromroots(x, roots)))

# multi-dimensional evaluation, Vandermonde matrices, roots, companion, fit, and the module constants
ops2 = {n: [] for n in ('polyval2d', 'polyval3d', 'polygrid2d', 'polygrid3d', 'polyvander2d', 'polyvander3d',
                         'polycompanion', 'polyroots', 'polyfit', 'polyzero', 'polyone', 'polyx', 'polydomain')}
for _ in range(5):
    c2 = rng.normal(size=(3, 4))
    c3 = rng.normal(size=(2, 3, 2))
    x = rng.uniform(-1, 1, 5); y = rng.uniform(-1, 1, 5); z = rng.uniform(-1, 1, 5)
    ops2['polyval2d'].append(case([x, y, c2], _P.polyval2d(x, y, c2)))
    ops2['polyval3d'].append(case([x, y, z, c3], _P.polyval3d(x, y, z, c3)))
    ops2['polygrid2d'].append(case([x, y, c2], _P.polygrid2d(x, y, c2)))
    ops2['polygrid3d'].append(case([x, y, z, c3], _P.polygrid3d(x, y, z, c3)))
    ops2['polyvander2d'].append(case([x, y, np.array([2, 3])], _P.polyvander2d(x, y, [2, 3])))
    ops2['polyvander3d'].append(case([x, y, z, np.array([1, 2, 1])], _P.polyvander3d(x, y, z, [1, 2, 1])))
    rts = np.sort(rng.uniform(-2, 2, int(rng.integers(2, 6))))
    c = _P.polyfromroots(rts)
    ops2['polycompanion'].append(case([c], _P.polycompanion(c), atol=1e-10))
    ops2['polyroots'].append(case([c], np.sort(_P.polyroots(c).real), atol=1e-6))
    xf = np.sort(rng.uniform(-1, 1, 14)); yf = np.cos(1.5 * xf) + 0.3 * xf
    ops2['polyfit'].append(case([xf, yf, 5], _P.polyfit(xf, yf, 5), atol=1e-6))
ops2['polyzero'].append(case([], _P.polyzero))
ops2['polyone'].append(case([], _P.polyone))
ops2['polyx'].append(case([], _P.polyx))
ops2['polydomain'].append(case([], _P.polydomain))

# numpy.polynomial.chebyshev
cheb = {n: [] for n in ('chebadd', 'chebsub', 'chebmul', 'chebmulx', 'chebpow', 'chebdiv', 'chebder', 'chebint',
            'chebfromroots', 'chebline', 'chebtrim', 'chebvander', 'chebval2d', 'chebval3d', 'chebgrid2d',
            'chebgrid3d', 'chebvander2d', 'chebvander3d', 'chebcompanion', 'chebroots', 'chebfit', 'cheb2poly',
            'poly2cheb', 'chebpts1', 'chebpts2', 'chebgauss', 'chebweight', 'chebzero', 'chebone', 'chebx', 'chebdomain')}
for _ in range(6):
    c1 = rng.normal(size=int(rng.integers(2, 6))); c2 = rng.normal(size=int(rng.integers(2, 5)))
    x = rng.uniform(-1, 1, 5); y = rng.uniform(-1, 1, 5); z = rng.uniform(-1, 1, 5)
    cc2 = rng.normal(size=(3, 4)); cc3 = rng.normal(size=(2, 3, 2))
    roots = rng.uniform(-1, 1, int(rng.integers(1, 5)))
    off, scl = float(rng.normal()), float(rng.normal())
    cheb['chebadd'].append(case([c1, c2], _C.chebadd(c1, c2)))
    cheb['chebsub'].append(case([c1, c2], _C.chebsub(c1, c2)))
    cheb['chebmul'].append(case([c1, c2], _C.chebmul(c1, c2)))
    cheb['chebmulx'].append(case([c1], _C.chebmulx(c1)))
    pw = int(rng.integers(0, 5)); cheb['chebpow'].append(case([c1, pw], _C.chebpow(c1, pw)))
    cbig = rng.normal(size=6); cheb['chebdiv'].append(case([cbig, c2], _C.chebdiv(cbig, c2), names=['quo', 'rem']))
    m = int(rng.integers(1, 3))
    cheb['chebder'].append(case([c1, m], _C.chebder(c1, m)))
    cheb['chebint'].append(case([c1, m], _C.chebint(c1, m)))
    cheb['chebfromroots'].append(case([roots], _C.chebfromroots(roots), atol=1e-8))
    cheb['chebline'].append(case([off, scl], _C.chebline(off, scl)))
    ct = np.append(c1, [1e-14, 0.0]); cheb['chebtrim'].append(case([ct, 1e-9], _C.chebtrim(ct, 1e-9)))
    deg = int(rng.integers(1, 5)); cheb['chebvander'].append(case([x, deg], _C.chebvander(x, deg)))
    cheb['chebval2d'].append(case([x, y, cc2], _C.chebval2d(x, y, cc2)))
    cheb['chebval3d'].append(case([x, y, z, cc3], _C.chebval3d(x, y, z, cc3)))
    cheb['chebgrid2d'].append(case([x, y, cc2], _C.chebgrid2d(x, y, cc2)))
    cheb['chebgrid3d'].append(case([x, y, z, cc3], _C.chebgrid3d(x, y, z, cc3)))
    cheb['chebvander2d'].append(case([x, y, np.array([2, 3])], _C.chebvander2d(x, y, [2, 3])))
    cheb['chebvander3d'].append(case([x, y, z, np.array([1, 2, 1])], _C.chebvander3d(x, y, z, [1, 2, 1])))
    rts = np.sort(rng.uniform(-1, 1, int(rng.integers(2, 5)))); cc = _C.chebfromroots(rts)
    cheb['chebcompanion'].append(case([cc], _C.chebcompanion(cc), atol=1e-10))
    cheb['chebroots'].append(case([cc], np.sort(_C.chebroots(cc).real), atol=1e-6))
    xf = np.sort(rng.uniform(-1, 1, 14)); yf = np.cos(1.5 * xf) + 0.3 * xf
    cheb['chebfit'].append(case([xf, yf, 5], _C.chebfit(xf, yf, 5), atol=1e-6))
    pol = rng.normal(size=int(rng.integers(2, 6)))
    cheb['cheb2poly'].append(case([c1], _C.cheb2poly(c1)))
    cheb['poly2cheb'].append(case([pol], _C.poly2cheb(pol)))
for npts in (2, 5, 8):
    cheb['chebpts1'].append(case([npts], _C.chebpts1(npts)))
    cheb['chebpts2'].append(case([npts], _C.chebpts2(npts)))
    gx, gw = _C.chebgauss(npts)
    cheb['chebgauss'].append(case([npts], (gx, gw), names=['x', 'w']))
wx = rng.uniform(-0.9, 0.9, 7)
cheb['chebweight'].append(case([wx], _C.chebweight(wx)))
cheb['chebzero'].append(case([], _C.chebzero)); cheb['chebone'].append(case([], _C.chebone))
cheb['chebx'].append(case([], _C.chebx)); cheb['chebdomain'].append(case([], _C.chebdomain))

calls = [{'fn': k, 'cases': v} for k, v in {**cls, **val, **ops, **ops2, **cheb}.items()]
out = {'module': 'npoly', 'numpy': np.__version__, 'env': F.fixture_env.env(), 'calls': calls}
with open(OUT, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(f'npoly: {len(CLASSES)} classes + {len(VALS)} val functions')
