#!/usr/bin/env python3
"""Find inputs on which SciPy's own implementation crashes (segfault or abort) for the spec'd functions.

    python3 tools/parity/crash_scan.py special [name ...]

Uses exactly the fixture inputs of tools/parity/fixtures.py and evaluates every row alone in a forked child
(run it with GLIBC_TUNABLES=glibc.malloc.check=3 so that heap corruption aborts instead of passing silently). Tessero shares SciPy's code, so every crash found here needs a `guard:`
in the spec (the kernel returns NaN instead of crashing PHP); fixtures.py then records NaN for those rows.
"""
import json, os, sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures  # noqa: E402

ROOT = fixtures.ROOT


def scan(name, rows, dtypes, timeout=10):
    """Evaluate every row alone in a forked child (the parent has SciPy imported, so a fork costs about a
    millisecond); rows whose child dies from a signal (a crash, or SIGALRM after `timeout` seconds: a hang)
    are returned."""
    import signal
    import warnings
    import scipy.special as sc
    fn = getattr(sc, name)
    bad = []
    for r in rows:
        pid = os.fork()
        if pid == 0:                                   # child
            try:
                signal.alarm(timeout)
                warnings.simplefilter('ignore')
                with np.errstate(all='ignore'):
                    fn(*[np.array([v], dtype=t) for v, t in zip(r, dtypes)])
            finally:
                os._exit(0)
        _, status = os.waitpid(pid, 0)
        if os.WIFSIGNALED(status):
            bad.append(r)
    return bad


def main():
    module = sys.argv[1]
    only = set(sys.argv[2:])
    spec = yaml.safe_load(open(os.path.join(ROOT, 'spec', f'{module}.yaml')))
    rng = np.random.default_rng(20260928)
    report = {}
    for fn in spec['functions']:
        if 'impl' not in fn:
            continue
        rows = fixtures.cases_for(fn, rng, int(fn.get('cases', 1500)))
        if only and fn['name'] not in only:
            continue
        dts = ['int64' if fn.get('integer_only') and k in fn['integer_only'] else 'float64' for k in range(len(fn['args']))]
        if fn.get('integer_only'):
            rows = [r for r in rows if all(np.isfinite(r[k]) for k in fn['integer_only'])]
            rows = [[int(v) if k in fn['integer_only'] else v for k, v in enumerate(r)] for r in rows]
        print('..', fn['name'], len(rows), file=sys.stderr, flush=True)
        bad = scan(fn['name'], rows, dts)
        if bad:
            report[fn['name']] = bad
            print(fn['name'], len(bad), bad[:12], flush=True)
    json.dump(report, open(os.path.join(ROOT, 'tools', 'parity', f'crashes-{module}.json'), 'w'), indent=0, default=str)


if __name__ == '__main__':
    main()
