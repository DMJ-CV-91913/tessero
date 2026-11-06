"""Per-call latency of the parity fixture calls in NumPy / SciPy: the Python half of the speed sweep's per-call tier.

    python3 tools/parity/time_calls.py OUT.json [fixture.json ...]

Pairs with `php tools/parity/run-tests.php --backend=ffi --time=OUT.json`: the same fixture files
(tests/fixtures/parity/*.json, the "calls" sections), the same argument decoding as run-tests.php decodeValue(),
the same cases skipped (frozen distributions, seeded generator methods, in-place calls) and the same timing: one
call, then repeated calls for about 5 ms (3 to 200 calls), the median in microseconds. Results are keyed by fixture
file, "module.fn" and case index, so tools/parity/time_report.py compares the two case for case. A call that
raises, or a function this script cannot resolve, is left out (and counted).
"""
import glob
import importlib
import json
import os
import sys
import time

import numpy as np

# fixture module -> the namespaces its functions live in, searched in order
NAMESPACES = {
    'np': ['numpy'],
    'linalg': ['numpy.linalg'],
    'fft': ['numpy.fft'],
    'npoly': ['numpy.polynomial.polynomial', 'numpy.polynomial.chebyshev', 'numpy.polynomial.legendre',
              'numpy.polynomial.laguerre', 'numpy.polynomial.hermite', 'numpy.polynomial.hermite_e', 'numpy.polynomial'],
    'special': ['scipy.special'],
    'signal': ['scipy.signal'],
    'windows': ['scipy.signal.windows'],
    'slinalg': ['scipy.linalg'],
    'sparse': ['scipy.sparse'],
    'splinalg': ['scipy.sparse.linalg'],
    'csgraph': ['scipy.sparse.csgraph'],
    'stats': ['scipy.stats'],
    'distance': ['scipy.spatial.distance'],
    'cluster': ['scipy.cluster.vq'],
    'hierarchy': ['scipy.cluster.hierarchy'],
    'interpolate': ['scipy.interpolate'],
    'ndimage': ['scipy.ndimage'],
    'scipyfft': ['scipy.fft'],
}


def resolve(module, fn):
    for ns in NAMESPACES.get(module, []):
        f = getattr(importlib.import_module(ns), fn, None)
        if callable(f):
            return f
    return None


def dec(v):
    """run-tests.php decodeValue()"""
    if isinstance(v, str):
        return {'nan': np.nan, 'inf': np.inf, '-inf': -np.inf, 'undef': None}.get(v, float('nan'))
    if not isinstance(v, dict):
        return v
    if 's' in v:
        return v['s']
    if 'rng' in v:
        return np.random.default_rng(v['rng'])
    if 'list' in v:
        return [dec(x) for x in v['list']]
    data = [dec(x) if isinstance(x, str) else x for x in v['data']]
    if v.get('dtype') == 'complex128':
        a = np.array(data[0::2], dtype=np.float64) + 1j * np.array(data[1::2], dtype=np.float64)
    else:
        a = np.array(data, dtype=v.get('dtype', 'float64'))
    return a.reshape(v['shape'])


def time_call(f, args, kwargs):
    t0 = time.perf_counter_ns()
    f(*args, **kwargs)
    first = (time.perf_counter_ns() - t0) / 1e3
    reps = int(max(3, min(200, 5000.0 / max(first, 0.1))))
    t = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        f(*args, **kwargs)
        t.append((time.perf_counter_ns() - t0) / 1e3)
    t.sort()
    return t[len(t) // 2]


def main():
    out_path = sys.argv[1]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    files = sys.argv[2:] or sorted(glob.glob(f'{root}/tests/fixtures/parity/*.json'))
    timing, skipped = {}, {'unresolved': 0, 'raised': 0}
    for path in files:
        data = json.load(open(path))
        module = data.get('module', '')
        for blk in data.get('calls', []):
            fn = blk['fn']
            f = resolve(module, fn)
            if f is None:
                skipped['unresolved'] += 1
                continue
            for ci, case in enumerate(blk['cases']):
                if 'dist_args' in case or 'seed' in case or case.get('compare') == 'inplace':
                    continue
                if isinstance(case.get('expect'), dict) and 'error' in case['expect']:
                    continue                                   # NumPy raises: nothing to time
                try:
                    args = [dec(a) for a in case['args']]
                    kwargs = {k: dec(v) for k, v in case['kwargs'].items()}
                    us = time_call(f, args, kwargs)
                except Exception:  # noqa: BLE001 - counted, left out of the comparison
                    skipped['raised'] += 1
                    continue
                timing.setdefault(os.path.basename(path), {}).setdefault(f'{module}.{fn}', {})[str(ci)] = us
    json.dump({'backend': 'numpy/scipy', 'unit': 'us', 'skipped': skipped, 'calls': timing}, open(out_path, 'w'), indent=1)
    print(f'timed {sum(len(c) for fl in timing.values() for c in fl.values())} cases; skipped {skipped}', file=sys.stderr)


if __name__ == '__main__':
    main()
