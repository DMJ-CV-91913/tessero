"""Compare two {case: milliseconds or "ERR ..."} timing maps from the speed sweep (Tessero first, NumPy/SciPy second).

    python3 bench/sweep_report.py TESSERO.json REFERENCE.json[l] [--md OUT.md] [--top N]

Accepts the bench/*_bench.{php,py} output ({case: ms}), the tier-1 PHP output ({"n": N, "ms": {fn: ms}}) and the
tier-1 Python JSON lines ({"fn": fn, "ms": ms} per line). Ratio = Tessero / reference: below 1, Tessero is faster.
Cases where either side failed are listed separately, with the reason.
"""
import json
import math
import sys


def load(path):
    text = open(path).read()
    if path.endswith('.jsonl'):
        out = {}
        for line in text.splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec['fn']] = rec['ms']
        return out
    data = json.loads(text)
    return data['ms'] if isinstance(data, dict) and 'ms' in data and isinstance(data['ms'], dict) else data


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    opts = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    top = int(opts.get('--top', 12))
    rows, errs = [], []
    for k in sorted(set(a) | set(b)):
        ta, tb = a.get(k), b.get(k)
        if isinstance(ta, (int, float)) and isinstance(tb, (int, float)) and ta > 0 and tb > 0:
            rows.append((k, ta, tb, ta / tb))
        else:
            errs.append((k, ta, tb))
    gm = math.exp(sum(math.log(r[3]) for r in rows) / len(rows))
    faster = sum(r[3] < 1 for r in rows)
    print(f'{len(rows)} cases timed on both sides: geomean Tessero/reference {gm:.2f}; Tessero faster on {faster}, '
          f'slower on {len(rows) - faster}; {len(errs)} cases failed on a side')
    rows.sort(key=lambda r: r[3])
    print('fastest relative to the reference:')
    for k, ta, tb, r in rows[:top]:
        print(f'  {r:7.2f}  {k:55s} {ta:10.3f} ms vs {tb:10.3f} ms')
    print('slowest relative to the reference:')
    for k, ta, tb, r in rows[-top:][::-1]:
        print(f'  {r:7.2f}  {k:55s} {ta:10.3f} ms vs {tb:10.3f} ms')
    if errs:
        print('failed on a side:')
        for k, ta, tb in errs:
            print(f'  {k}: Tessero {str(ta)[:70]} | reference {str(tb)[:70]}')
    if '--md' in opts:
        with open(opts['--md'], 'w') as f:
            f.write('| case | Tessero ms | reference ms | ratio |\n|---|---|---|---|\n')
            for k, ta, tb, r in rows:
                f.write(f'| {k} | {ta:.3g} | {tb:.3g} | {r:.2f} |\n')


if __name__ == '__main__':
    main()
