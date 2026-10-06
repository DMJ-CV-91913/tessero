#!/usr/bin/env python3
"""Generate SciPy reference fixtures for the scipy.stats distributions (stats.<name> in the registry).

    python3 tools/parity/fixtures_stats.py              # every distribution
    python3 tools/parity/fixtures_stats.py gamma norm   # some (merged into the existing file)
    python3 tools/parity/fixtures_stats.py --out=/tmp/x.json gamma   # some, into a separate file

Writes tests/fixtures/parity/stats.json in the call format of fixtures_np.py: for every distribution a list
of calls on a frozen distribution, `dist(*shapes, loc=, scale=).<method>(*args)`, and SciPy's result (or the
exception it raises). The parameter sets are SciPy's own test sets (scipy.stats._distr_params: distcont,
distdiscrete, invdistcont, invdistdiscrete) with two loc/scale settings; the evaluation points are the
distribution's quantiles, its support end points, points outside the support, and NaN/inf probes. rvs is
called with random_state=numpy.random.default_rng(seed), the stream Tessero\\Random\\Generator reproduces.

Each block carries `tol` when a distribution needs more than the default rtol 1e-12, with the reason.
"""
import json, math, os, sys, time, warnings

import numpy as np
import fixture_env
import scipy
import scipy.stats as st
from scipy.stats._distr_params import distcont, distdiscrete, invdistcont, invdistdiscrete

import fixtures_np as F

OUT = os.path.join(F.ROOT, 'tests', 'fixtures', 'parity', 'stats.json')
PROBS = [1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1 - 1e-6]
QS = [0.0, 1e-10, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0, -0.1, 1.1, float('nan')]

# parameter sets beyond SciPy's own test sets (scipy.stats._distr_params), for code paths those never reach:
# name -> [(shapes, valid)]
EXTRA = {
    # levy_stable_gen's public pdf, cdf and rvs move loc by 2 beta scale log(scale) / pi when alpha == 1
    'levy_stable': [((1.0, 0.5), True), ((1.0, -0.3), True)],
}

# distributions whose generic numerical methods (QUADPACK integration, root finding) are sensitive to
# last-bit differences in the integrand: reason -> names
TOL = {}


def enc_expect(r):
    if isinstance(r, tuple):
        if len(r) == 2:                                   # support / interval: a pair
            return {'list': [F.enc_result_one(x) for x in r]}
        names = ['mean', 'var', 'skew', 'kurtosis']
        return {'dict': {n: F.enc_result_one(x) for n, x in zip(names, r)}}
    return F.enc_result_one(r)


def record(frozen, method, args, kwargs=None):
    kwargs = kwargs or {}
    t = time.time()
    upstream = None
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            call_kw = dict(kwargs)
            if 'random_state' in call_kw:
                call_kw['random_state'] = np.random.default_rng(call_kw['random_state'])
            r = getattr(frozen, method)(*args, **call_kw)
            expect = enc_expect(r)
        except Exception as e:  # noqa: BLE001
            expect = {'error': type(e).__name__}
            dev = invalid_args_rule(frozen, method, e)
            if dev:
                upstream = {'rule': 'U1', 'scipy': expect}
                expect = dev
        dev = elementwise_rule(frozen, method, args, expect)
        if dev:
            upstream = {'rule': 'U2', 'scipy': expect}
            expect = dev
    kw = {}
    for k, v in kwargs.items():
        kw[k] = {'rng': v} if k == 'random_state' else F.enc(list(v) if isinstance(v, tuple) else v)
    rec = {'method': method, 'args': [F.enc(a) for a in args], 'kwargs': kw, 'expect': expect,
           'secs': round(time.time() - t, 3)}
    if upstream:
        rec['upstream'] = upstream
    return rec


# ---- upstream-bug rules (docs/project/upstream-bugs.md). A rule replaces SciPy's result by the result its own
# documented contract gives; the call record keeps SciPy's actual result under "upstream" {rule, scipy} so every
# deviation is visible and counted (tools/parity/report.py). No rule loosens a tolerance.

def args_valid(frozen):
    shapes = frozen.args
    ok = bool(np.all(frozen.dist._argcheck(*[np.asarray(a) for a in shapes])))
    scale = frozen.kwds.get('scale', 1.0)
    return ok and scale > 0


def invalid_args_rule(frozen, method, exc):
    """U1: for shape parameters that fail _argcheck, rv_generic's public methods return badvalue (nan): they
    evaluate the private method only on the (here empty) set of valid arguments. Some private methods raise on
    that empty set (0.0 ** negative, an index into an empty array, a bracket search on no data, BiasedUrn's own
    range check); SciPy's contract is nan."""
    if method not in ('entropy', 'moment', 'median', 'mean', 'var', 'std', 'stats', 'interval'):
        return None
    try:
        if args_valid(frozen):
            return None
    except Exception:  # noqa: BLE001 - _argcheck itself raising (BiasedUrn) means the arguments are invalid
        pass
    nan = F.enc_scalar(float('nan'))
    if method == 'stats':
        return {'dict': {n: nan for n in ('mean', 'var', 'skew', 'kurtosis')}}
    return {'list': [nan, nan]} if method == 'interval' else nan


def elementwise_rule(frozen, method, args, expect):
    """U2: a public method applied to an array equals the method applied to each element (numpy broadcasting).
    Where SciPy's vectorised result differs from its own element-by-element results (norminvgauss sf/isf
    truncate their output through zip() on arrays), the element-by-element results are the expectation."""
    if method not in ('pdf', 'logpdf', 'pmf', 'logpmf', 'cdf', 'logcdf', 'sf', 'logsf', 'ppf', 'isf'):
        return None
    if not args or not isinstance(args[0], np.ndarray) or args[0].ndim == 0 or (isinstance(expect, dict) and 'error' in expect):
        return None
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            vec = np.asarray(getattr(frozen, method)(args[0]), dtype=float)
            one = np.array([float(getattr(frozen, method)(float(x))) for x in args[0].ravel()]).reshape(args[0].shape)
        except Exception:  # noqa: BLE001
            return None
    if vec.shape == one.shape and np.allclose(vec, one, rtol=1e-12, atol=0, equal_nan=True):
        return None
    return F.enc(one)


def points(frozen, discrete):
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            xs = [float(v) for v in frozen.ppf(PROBS)]
            lo, hi = (float(v) for v in frozen.support())
        except Exception:  # noqa: BLE001
            xs, lo, hi = [0.5, 1.0, 2.0], -np.inf, np.inf
    pts = [x for x in xs if math.isfinite(x)]
    for e in (lo, hi):
        if math.isfinite(e):
            pts += [e, e - 1.0, e + 1.0] if discrete else [e, e - 0.5, e + 0.5, e + 1e-9 * max(1.0, abs(e))]
    if discrete:
        pts += [p + 0.5 for p in pts[:3]]                 # non-integers
    pts += [float('nan'), float('inf'), -float('inf')]
    seen, out = set(), []
    for p in pts:
        k = 'nan' if p != p else p
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def cases_for(name, sets, discrete):
    dist = getattr(st, name)
    cases = []
    for shapes, valid in sets:
        locscales = [{}, {'loc': 1.5}] if discrete else [{}, {'loc': 1.5, 'scale': 2.5}]
        for ls in locscales:
            args = list(shapes)
            try:
                frozen = dist(*args, **ls)
            except Exception:  # noqa: BLE001
                continue
            base = {'dist_args': [F.enc(np.asarray(a, dtype=float)) if isinstance(a, list) else F.enc(float(a)) for a in args],
                    'dist_kwargs': {k: F.enc(float(v)) for k, v in ls.items()}}
            calls = []
            xs = points(frozen, discrete) if valid else [0.5, 1.0, float('nan')]
            xarr = np.array(xs)
            for m in (['pmf', 'logpmf'] if discrete else ['pdf', 'logpdf']) + ['cdf', 'logcdf', 'sf', 'logsf']:
                calls.append(record(frozen, m, [xarr]))
                calls.append(record(frozen, m, [xs[0]]))
            for m in ('ppf', 'isf'):
                calls.append(record(frozen, m, [np.array(QS)]))
            calls.append(record(frozen, 'stats', [], {'moments': 'mvsk'}))
            calls.append(record(frozen, 'stats', [], {'moments': 'mv'}))
            for m in ('mean', 'var', 'std', 'median', 'entropy', 'support'):
                calls.append(record(frozen, m, []))
            calls.append(record(frozen, 'interval', [0.9]))
            for order in (1, 2, 3, 4, 5):
                calls.append(record(frozen, 'moment', [order]))
            for seed in (0, 42):
                calls.append(record(frozen, 'rvs', [], {'size': 5, 'random_state': seed}))
            calls.append(record(frozen, 'rvs', [], {'size': (2, 3), 'random_state': 7}))
            cases.append({**base, 'calls': calls})
    return cases


def main():
    global OUT
    argv = sys.argv[1:]
    separate = False
    for a in list(argv):
        if a.startswith('--out='):
            OUT = a[6:]
            separate = True
            argv.remove(a)
    want = set(argv)
    cont, disc = {}, {}
    for name, p in distcont:
        cont.setdefault(name, []).append((p, True))
    for name, p in invdistcont:
        cont.setdefault(name, []).append((p, False))
    for name, p in distdiscrete:
        disc.setdefault(name, []).append((p, True))
    for name, p in invdistdiscrete:
        disc.setdefault(name, []).append((p, False))
    for name, sets in EXTRA.items():
        (disc if name in disc else cont).setdefault(name, []).extend((list(p), v) for p, v in sets)
    blocks = []
    old = {}
    if want and not separate and os.path.exists(OUT):
        old = {b['fn']: b for b in json.load(open(OUT))['calls']}
    t0 = time.time()
    for name in sorted(set(cont) | set(disc)):
        if want and name not in want:
            if name in old:
                blocks.append(old[name])
            continue
        t = time.time()
        discrete = name in disc
        sets = disc[name] if discrete else cont[name]
        blk = {'fn': name, 'discrete': discrete, 'cases': cases_for(name, sets, discrete)}
        for reason, names in TOL.items():
            if name in names:
                blk['tol'] = {'rtol': reason[0], 'why': reason[1]}
        blocks.append(blk)
        print(f'{name}: {sum(len(c["calls"]) for c in blk["cases"])} calls, {time.time() - t:.1f}s', file=sys.stderr, flush=True)
    data = {'module': 'stats', 'kind': 'distributions', 'numpy': np.__version__, 'scipy': scipy.__version__, 'env': fixture_env.env(), 'calls': blocks}
    fixture_env.require_pinned()
    with open(OUT, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    print(f'stats: {len(blocks)} distributions, {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
