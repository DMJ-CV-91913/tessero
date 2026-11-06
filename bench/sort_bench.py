"""NumPy's np.sort (default and kind="stable") on the inputs of bench/sort_bench.php, as a Python app sees it.

    OMP_NUM_THREADS=1 python3 bench/sort_bench.py > numpy.json
"""
import json, sys, time
import numpy as np

n = 1000000
def best(f):
    f()
    b = float('inf')
    for _ in range(7):
        s = time.perf_counter_ns(); f(); b = min(b, (time.perf_counter_ns() - s) / 1e6)
    return round(b, 3)

rand = np.random.default_rng(7).random(n) * 2.0 - 1.0
srt = np.sort(rand)
moved = srt.copy()
for i in range(n // 100):
    j = (i * 104729) % n; k = (i * 1299709 + 12345) % n
    moved[j], moved[k] = moved[k], moved[j]
app = np.concatenate([srt[:n - n // 100], np.random.default_rng(8).random(n // 100) * 2.0 - 1.0])
cases = {
    'float64 random': rand,
    'float64 10 distinct values': np.floor((rand + 1.0) * 5.0),
    'float64 sorted': srt,
    'float64 reversed': srt[::-1].copy(),
    'float64 sorted, 1% of values moved': moved,
    'float64 sorted, 1% random values appended': app,
    'float32 random': rand.astype(np.float32),
    'int64 random': (rand * 4.0e18).astype(np.int64),
}
out = {}
for name, a in cases.items():
    out[name] = {'np.sort': best(lambda: np.sort(a)), 'np.sort stable': best(lambda: np.sort(a, kind='stable'))}
    print(name, 'done', file=sys.stderr)
print(json.dumps(out, indent=1))
