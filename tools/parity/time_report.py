"""Compare the per-call latencies of run-tests.php --time (Tessero) and time_calls.py (NumPy / SciPy).

    python3 tools/parity/time_report.py TESSERO.json NUMPY.json [--md OUT.md] [--json OUT.json]

Only cases timed on both sides count. Per function: the ratio of the summed medians (Tessero / reference; below 1
means Tessero is faster). Per module and overall: the geometric mean of the functions' ratios, and how many
functions are faster or slower.
"""
import json
import math
import sys
from collections import defaultdict


def cases(v):
    """{case index: microseconds}; PHP writes a contiguous 0..n map as a JSON list"""
    if isinstance(v, list):
        return {str(i): x for i, x in enumerate(v) if x is not None}
    return {k: x for k, x in v.items() if x is not None}


def main():
    a = json.load(open(sys.argv[1]))['calls']
    b = json.load(open(sys.argv[2]))['calls']
    opts = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    rows = []
    for fixture, fns in a.items():
        for key, tv in fns.items():
            rv = b.get(fixture, {}).get(key)
            if rv is None:
                continue
            t, r = cases(tv), cases(rv)
            common = [c for c in t if c in r and t[c] > 0 and r[c] > 0]
            if not common:
                continue
            ts, rs = sum(t[c] for c in common), sum(r[c] for c in common)
            rows.append({'fixture': fixture, 'fn': key, 'cases': len(common), 'tessero_us': ts / len(common),
                         'reference_us': rs / len(common), 'ratio': ts / rs})
    if not rows:
        sys.exit('no case timed on both sides')
    by_mod = defaultdict(list)
    for x in rows:
        by_mod[x['fn'].split('.')[0]].append(x['ratio'])

    def gm(v):
        return math.exp(sum(math.log(x) for x in v) / len(v))

    allr = [x['ratio'] for x in rows]
    summary = {'functions': len(rows), 'cases': sum(x['cases'] for x in rows), 'geomean_ratio': gm(allr),
               'faster': sum(r < 1 for r in allr), 'slower': sum(r >= 1 for r in allr),
               'modules': {m: {'functions': len(v), 'geomean_ratio': gm(v), 'faster': sum(r < 1 for r in v)}
                           for m, v in sorted(by_mod.items())}}
    print(f"{summary['functions']} functions, {summary['cases']} cases: geomean Tessero/reference "
          f"{summary['geomean_ratio']:.2f}; faster {summary['faster']}, slower {summary['slower']}")
    for m, s in summary['modules'].items():
        print(f"  {m:12s} {s['functions']:4d} functions  geomean {s['geomean_ratio']:.2f}  faster {s['faster']}")
    if '--json' in opts:
        json.dump({'summary': summary, 'rows': sorted(rows, key=lambda x: x['ratio'])}, open(opts['--json'], 'w'), indent=1)
    if '--md' in opts:
        with open(opts['--md'], 'w') as f:
            f.write('| module | functions | geomean ratio | faster |\n|---|---|---|---|\n')
            for m, s in summary['modules'].items():
                f.write(f"| {m} | {s['functions']} | {s['geomean_ratio']:.2f} | {s['faster']} |\n")
            f.write(f"| **all** | {summary['functions']} | **{summary['geomean_ratio']:.2f}** | {summary['faster']} |\n")


if __name__ == '__main__':
    main()
