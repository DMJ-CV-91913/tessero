"""SciPy timings for bench/tier1_bench.php: every scipy.special function on the parity fixture's rows tiled to
100,000 values (median of repeated runs, ms). One JSON line per function, each timed in a child process, so a
function that crashes or hangs SciPy on the fixture rows (tools/parity/crashes-special.json) costs only its line.

    OPENBLAS_NUM_THREADS=1 python3 bench/tier1_bench.py > scipy.jsonl
    ... bench/tier1_bench.py gamma             # only the functions whose name contains "gamma"
"""
import json
import os
import signal
import sys
import time

import numpy as np
import scipy.special as sc

N = 100000
TIMEOUT = 300  # seconds per function


def med(f):
    s = time.perf_counter()
    f()
    first = (time.perf_counter() - s) * 1e3
    if first > 30000:
        return first  # over 30 s: one run says enough
    inner = int(max(1, min(500, 10.0 / max(first, 1e-4))))
    n = 3 if first > 500 else 9
    t = []
    for _ in range(n):
        s = time.perf_counter()
        for _ in range(inner):
            f()
        t.append((time.perf_counter() - s) * 1e3 / inner)
    t.sort()
    return t[n // 2]


def tiled(col):
    vals = np.array([float('nan') if v == 'undef' else float(v) for v in col], dtype=np.float64)
    return np.resize(vals, N)  # repeats the column, as the PHP side does


def timed(fn, cols):
    """the median in ms, or "ERR ..."; runs in a child: a crash or a hang there is reported, not inherited"""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            args = [tiled(c) for c in cols]
            f = getattr(sc, fn)
            res = med(lambda: f(*args))
        except Exception as e:  # noqa: BLE001 - recorded per function
            res = f'ERR {type(e).__name__}: {str(e)[:160]}'
        os.write(w, json.dumps(res).encode())
        os._exit(0)
    os.close(w)
    signal.signal(signal.SIGALRM, lambda *_: os.kill(pid, signal.SIGKILL))
    signal.alarm(TIMEOUT)
    data = b''
    while chunk := os.read(r, 65536):
        data += chunk
    os.close(r)
    _, status = os.waitpid(pid, 0)
    signal.alarm(0)
    if data:
        return json.loads(data)
    return f'ERR child ended without a result (status {status}: crash or {TIMEOUT} s timeout)'


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixture = json.load(open(os.path.join(root, 'tests/fixtures/parity/special.json')))
    only = sys.argv[1] if len(sys.argv) > 1 else None
    seen = set()
    for blk in fixture['blocks']:
        fn = blk['fn']
        if blk['int'] or fn in seen or (only is not None and only not in fn):
            continue  # the float64 loop, once per function
        seen.add(fn)
        print(json.dumps({'fn': fn, 'ms': timed(fn, blk['args'])}), flush=True)


if __name__ == '__main__':
    main()
