#!/usr/bin/env python3
"""Compare a bench/run.php --json result with a committed baseline and fail on a regression.

    php -d ffi.enable=1 bench/run.php --json --no-gates > bench.json
    python3 .github/scripts/bench_compare.py .github/bench-baseline.json bench.json [--update]

A case regresses when it takes more than FACTOR x its baseline plus SLACK_MS (the slack keeps sub-millisecond
cases from failing on scheduler noise). The run also fails when the geometric mean of all ratios exceeds
GEOMEAN_MAX, which catches a broad slowdown that stays under the per-case factor. Cases missing from the
baseline are reported and ignored; cases missing from the result fail (a benchmark was dropped).

--update rewrites the baseline from the result (run it on the CI runner type the baseline describes).
The summary goes to stdout and, on GitHub Actions, to the step summary.
"""
import json
import math
import os
import sys

FACTOR = 2.0
SLACK_MS = 1.0
GEOMEAN_MAX = 1.35


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    if len(args) != 2:
        print(__doc__)
        return 2
    with open(args[0], encoding='utf-8') as f:
        base = json.load(f)
    with open(args[1], encoding='utf-8') as f:
        cur = json.load(f)
    if '--update' in argv:
        base['ms'] = cur['ms']
        base['info'] = cur.get('info', {})
        with open(args[0], 'w', encoding='utf-8') as f:
            json.dump(base, f, indent=2, sort_keys=False)
            f.write('\n')
        print(f'baseline {args[0]} updated from {args[1]}')
        return 0

    bms, cms = base['ms'], cur['ms']
    rows, failures, ratios = [], [], []
    for name, b in bms.items():
        if name not in cms:
            failures.append(f'{name}: missing from the result')
            rows.append((name, b, None, None, 'MISSING'))
            continue
        c = float(cms[name])
        ratio = c / b if b > 0 else 1.0
        ratios.append(ratio)
        limit = FACTOR * b + SLACK_MS
        status = 'ok'
        if c > limit:
            status = 'REGRESSION'
            failures.append(f'{name}: {c:.3f} ms > {limit:.3f} ms ({FACTOR}x baseline {b:.3f} + {SLACK_MS} ms)')
        rows.append((name, b, c, ratio, status))
    for name in cms:
        if name not in bms:
            rows.append((name, None, float(cms[name]), None, 'new (no baseline)'))
    geomean = math.exp(sum(math.log(max(r, 1e-9)) for r in ratios) / len(ratios)) if ratios else 1.0
    if geomean > GEOMEAN_MAX:
        failures.append(f'geometric mean of the ratios {geomean:.3f} > {GEOMEAN_MAX}')

    lines = [f'### Benchmarks vs baseline ({base.get("runner", "baseline")})', '',
             f'Per case: fail above {FACTOR}x baseline + {SLACK_MS} ms. Geometric mean of ratios: '
             f'**{geomean:.3f}** (fail above {GEOMEAN_MAX}).', '',
             '| case | baseline ms | this run ms | ratio | |', '|---|---:|---:|---:|---|']
    for name, b, c, r, s in rows:
        fb = f'{b:.3f}' if b is not None else ''
        fc = f'{c:.3f}' if c is not None else ''
        fr = f'{r:.2f}' if r is not None else ''
        lines.append(f'| {name} | {fb} | {fc} | {fr} | {s} |')
    if failures:
        lines += ['', '**Regressions:**', ''] + [f'- {x}' for x in failures]
    text = '\n'.join(lines) + '\n'
    print(text)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as f:
            f.write(text)
    for x in failures:
        print(f'::error::benchmark regression: {x}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
