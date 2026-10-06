"""NumPy/SciPy timings for the same cases as bench/run.php (median of N runs, ms)."""
import json, sys, time
import numpy as np
import scipy.linalg as sla

def med(fn, runs=15):
    fn()
    t = []
    for _ in range(runs):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1e3)
    t.sort(); return t[len(t) // 2]

rng = np.random.default_rng(0)
a = rng.random(1_000_000); b = rng.random(1_000_000); out = np.empty(1_000_000)
m600 = rng.random((600, 600)); m2 = rng.random((600, 600)); big2d = rng.random((1000, 1000))
spd = m600 @ m600.T + 600 * np.eye(600); sig = rng.random(1 << 20); lst = a.tolist()
cases = {
    'add 1M f64 (new array)': lambda: a + b,
    'add 1M f64 (out=)': lambda: np.add(a, b, out=out),
    'mul scalar 1M': lambda: a * 2.5,
    'sqrt 1M': lambda: np.sqrt(a),
    'exp 1M': lambda: np.exp(a),
    'log 1M': lambda: np.log(a),
    'sum 1M (pairwise)': lambda: a.sum(),
    'sum axis 0 1000x1000': lambda: big2d.sum(0),
    'transpose copy 1000x1000': lambda: big2d.T.copy(),
    'boolean mask 1M': lambda: a[a > 0.5],
    'sort 1M': lambda: np.sort(a),
    'matmul 600x600': lambda: m600 @ m2,
    'solve 600x600': lambda: np.linalg.solve(m600, b[:600]),
    'cholesky 600x600': lambda: np.linalg.cholesky(spd),
    'svd 300x300': lambda: np.linalg.svd(m600[:300, :300]),
    'fft 2^20 complex': lambda: np.fft.fft(sig),
    'rfft 2^20 real': lambda: np.fft.rfft(sig),
    'fft 1_000_003 (prime, Bluestein)': lambda: np.fft.fft(sig[:1000003]),
    'rng normal 1M': lambda: rng.normal(size=1_000_000),
    'PHP list -> NDArray 1M': lambda: np.array(lst),
    'NDArray -> PHP list 1M': lambda: a.tolist(),
}
res = {k: round(med(f, 5 if ('prime' in k or 'sort' in k) else 15), 3) for k, f in cases.items()}
if '--json' in sys.argv:
    print(json.dumps({'numpy': np.__version__, 'ms': res}, indent=2))
else:
    for k, v in res.items(): print(f"  {k:<36} {v:10.3f} ms")
