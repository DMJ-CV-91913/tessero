#!/usr/bin/env python3
"""Coverage metrics of Tessero against the NumPy/SciPy census, computed from the code and the test results.

    python3 tools/parity/report.py                      # write docs/project/_generated/metrics.md + metrics.json
    python3 tools/parity/report.py --record "phase 1"   # also append a row to .work/metrics-history.csv
    python3 tools/parity/report.py --check              # fail when the written files are out of date

Inputs (nothing is typed by hand):
  census.json    every public NumPy/SciPy symbol (tools/parity/census.py)
  scope.yaml     exclusions by category E1-E7, phases, the map of functions older than the registry
  registry.json  the kernel registry's functions (tools/parity/gen-facades.php)
  results.json   fixture outcome per backend and registry function (tools/parity/run-tests.php) and the
                 PHPUnit suites that verify the older functions (tools/parity/record-suites.php)
  api.json       the public PHP API of both backends (tools/api-dump.php)
  usage.json     call counts of each symbol in the library code of widely used Python packages (tools/parity/usage.py)

Metrics (prompts/numpy-scipy-parity-program.md section 3.4):
  M2  in-scope symbol coverage per module and backend: implemented / (canonical symbols - excluded)
  M3  excluded share per module: excluded / canonical symbols
  M4  verified share: implemented symbols whose verification (fixture or suite) passed in the recorded run;
      M4 < M2 means something is counted that the last run did not verify
  M5  backend parity: symbols implemented on FFI that the extension also implements
Aliases (numpy.acos for numpy.arccos) inherit their canonical's status and are not counted.
  U   usage-weighted coverage: the share of in-scope call sites in the usage corpus whose callee is verified on
      a backend, and how many of the N most-called in-scope symbols are verified on both backends
M1 (the curated rows of the coverage page) stays with tools/gap_inventory.py.
"""
import csv, datetime, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scope  # noqa: E402

HERE = scope.HERE
ROOT = scope.ROOT
OUT_MD = os.path.join(ROOT, 'docs', 'project', '_generated', 'metrics.md')
OUT_JSON = os.path.join(HERE, 'metrics.json')
OUT_DEV = os.path.join(ROOT, 'docs', 'project', '_generated', 'deviations.md')
OUT_USAGE = os.path.join(ROOT, 'docs', 'project', '_generated', 'usage.md')
TOP_N = (25, 50, 100, 250)
FIXDIR = os.path.join(ROOT, 'tests', 'fixtures', 'parity')
HISTORY = os.path.join(ROOT, '.work', 'metrics-history.csv')

GROUPS = [
    ('NumPy core', ['numpy', 'numpy.ndarray', 'numpy.linalg', 'numpy.fft', 'numpy.polynomial',
                    'numpy.polynomial.polynomial', 'numpy.random', 'numpy.random.Generator']),
    ('SciPy', ['scipy.special', 'scipy.stats', 'scipy.linalg', 'scipy.fft', 'scipy.optimize', 'scipy.integrate',
               'scipy.interpolate', 'scipy.signal', 'scipy.signal.windows', 'scipy.sparse', 'scipy.sparse.linalg',
               'scipy.sparse.csgraph', 'scipy.spatial', 'scipy.spatial.distance', 'scipy.cluster.vq',
               'scipy.cluster.hierarchy', 'scipy.ndimage', 'scipy.constants']),
]


def pct(a, b):
    return 0.0 if b == 0 else round(100.0 * a / b, 1)


def tally(recs):
    t = {'symbols': 0, 'excluded': 0, 'in_scope': 0}
    for b in scope.BACKENDS:
        t[f'impl_{b}'] = 0
        t[f'verified_{b}'] = 0
    t['both'] = 0
    for r in recs:
        if 'alias_of' in r:
            continue
        t['symbols'] += 1
        if 'excluded' in r['status'].values():
            t['excluded'] += 1
            continue
        t['in_scope'] += 1
        for b in scope.BACKENDS:
            if r['status'][b] == 'implemented':
                t[f'impl_{b}'] += 1
                if r['verified'][b]:
                    t[f'verified_{b}'] += 1
        if all(r['status'][b] == 'implemented' for b in scope.BACKENDS):
            t['both'] += 1
    t['m2_ffi'] = pct(t['impl_ffi'], t['in_scope'])
    t['m2_ext'] = pct(t['impl_ext'], t['in_scope'])
    t['m3'] = pct(t['excluded'], t['symbols'])
    t['m4_ffi'] = pct(t['verified_ffi'], t['in_scope'])
    t['m4_ext'] = pct(t['verified_ext'], t['in_scope'])
    t['m5'] = pct(t['both'], t['impl_ffi']) if t['impl_ffi'] else 100.0
    return t


def compute():
    census = scope.load_json(os.path.join(HERE, 'census.json'), None)
    if census is None:
        sys.exit('census.json missing: run tools/parity/census.py')
    recs, problems = scope.classify(census)
    modules = []
    seen = []
    for r in recs:
        if r['module'] not in seen:
            seen.append(r['module'])
    per = {m: tally([r for r in recs if r['module'] == m]) for m in seen}
    groups = {g: tally([r for r in recs if r['module'] in mods]) for g, mods in GROUPS}
    cats = {}
    for r in recs:
        if 'category' in r and 'alias_of' not in r:
            cats.setdefault(r['module'], {}).setdefault(r['category'], 0)
            cats[r['module']][r['category']] += 1
    results = scope.load_json(os.path.join(HERE, 'results.json'), {})
    registry = scope.load_json(os.path.join(HERE, 'registry.json'), {}).get('functions', {})
    fix = {b: sum(1 for k in registry if results.get(b, {}).get(k) == 'pass') for b in scope.BACKENDS}
    fixfail = {b: sorted(k for k, v in results.get(b, {}).items() if v not in ('pass', 'missing')) for b in scope.BACKENDS}
    missing = sorted({k for b in scope.BACKENDS for k, v in results.get(b, {}).items() if v == 'missing'})
    untested = sorted(n for n in registry if any(n not in results.get(b, {}) for b in scope.BACKENDS))
    return {
        'numpy': census.get('numpy'), 'scipy': census.get('scipy'),
        'modules': per, 'groups': groups, 'excluded_by_category': cats,
        'registry_functions': len(registry), 'fixture_pass': fix, 'fixture_fail': fixfail, 'fixture_missing': missing,
        'registry_untested': untested, 'problems': problems,
        'usage': usage_summary(recs),
    }, recs


def usage(recs):
    """Usage-weighted coverage (U): recs from scope.classify, weights from usage.json."""
    data = scope.load_json(os.path.join(HERE, 'usage.json'), None)
    if data is None:
        return None
    idx = {r['module'] + '.' + r['name']: r for r in recs if 'alias_of' not in r}
    group_of = {m: g for g, mods in GROUPS for m in mods}
    rows, unmatched = [], 0
    for key, u in data['symbols'].items():
        r = idx.get(key)
        g = group_of.get(r['module']) if r else None
        if r is None or g is None:
            unmatched += u['calls']
            continue
        excluded = 'excluded' in r['status'].values()
        ver = {b: r['status'][b] == 'implemented' and bool(r['verified'][b]) for b in scope.BACKENDS}
        rows.append({'symbol': key, 'group': g, 'calls': u['calls'], 'packages': u['packages'], 'excluded': excluded,
                     'category': r.get('category', ''), **{f'ok_{b}': ver[b] for b in scope.BACKENDS}})
    rows.sort(key=lambda x: (-x['calls'], x['symbol']))

    def summ(rs):
        t = {'calls': sum(x['calls'] for x in rs), 'excluded_calls': sum(x['calls'] for x in rs if x['excluded'])}
        ins = [x for x in rs if not x['excluded']]
        t['in_scope_calls'] = sum(x['calls'] for x in ins)
        for b in scope.BACKENDS:
            t[f'covered_{b}'] = sum(x['calls'] for x in ins if x[f'ok_{b}'])
            t[f'u_{b}'] = pct(t[f'covered_{b}'], t['in_scope_calls'])
        t['covered_both'] = sum(x['calls'] for x in ins if all(x[f'ok_{b}'] for b in scope.BACKENDS))
        t['u_both'] = pct(t['covered_both'], t['in_scope_calls'])
        t['excluded_share'] = pct(t['excluded_calls'], t['calls'])
        t['top'] = {}
        for n in TOP_N:
            top = ins[:n]
            t['top'][str(n)] = {'n': len(top), 'both': sum(1 for x in top if all(x[f'ok_{b}'] for b in scope.BACKENDS)),
                                **{b: sum(1 for x in top if x[f'ok_{b}']) for b in scope.BACKENDS}}
        return t

    out = {'corpus': data['corpus'], 'total_calls': data['total_calls'], 'unmatched_calls': unmatched,
           'groups': {g: summ([x for x in rows if x['group'] == g]) for g, _ in GROUPS},
           'all': summ(rows), 'rows': rows}
    return out


def usage_summary(recs):
    u = usage(recs)
    if u is None:
        return None
    return {'corpus_packages': len(u['corpus']), 'corpus_files': sum(x['files'] for x in u['corpus']),
            'total_calls': u['total_calls'], 'unmatched_calls': u['unmatched_calls'],
            'groups': u['groups'], 'all': u['all']}


def render_usage(u):
    L = ['<!-- generated by tools/parity/report.py from usage.json, census.json, scope.yaml and results.json;'
         ' do not edit by hand -->', '']
    c = u['corpus']
    L.append(f"**Corpus.** The library code of {len(c)} packages ({sum(x['files'] for x in c)} files, tests and "
             f"vendored code excluded) makes {u['total_calls']} calls to NumPy and SciPy that resolve statically: "
             + ', '.join(f"{x['name']} {x['tag']}" for x in c) + '. Method calls on arrays (`x.sum()`) cannot be '
             'attributed statically and are not counted.')
    L.append('')
    L.append('| Scope | Calls | To excluded symbols | In scope | Verified FFI | Verified extension | Verified on both |')
    L.append('|---|---:|---:|---:|---:|---:|---:|')
    for name, t in [(g, u['groups'][g]) for g, _ in GROUPS] + [('NumPy and SciPy', u['all'])]:
        L.append(f"| **{name}** | {t['calls']} | {t['excluded_calls']} ({t['excluded_share']} %) | {t['in_scope_calls']} | "
                 f"{t['u_ffi']} % | {t['u_ext']} % | **{t['u_both']} %** |")
    L.append('')
    L.append('The most-called in-scope symbols, and how many of them are verified:')
    L.append('')
    L.append('| Most-called in scope | NumPy core: both backends (FFI, ext) | SciPy: both (FFI, ext) | Together: both (FFI, ext) |')
    L.append('|---|---:|---:|---:|')
    for n in TOP_N:
        cells = []
        for t in (u['groups']['NumPy core'], u['groups']['SciPy'], u['all']):
            x = t['top'][str(n)]
            cells.append(f"{x['both']} of {x['n']} ({x['ffi']}, {x['ext']})")
        L.append(f"| top {n} | " + ' | '.join(cells) + ' |')
    L.append('')
    ins = [x for x in u['rows'] if not x['excluded']]
    mark = lambda ok: '✔' if ok else '—'
    L.append('The 100 most-called in-scope symbols (calls in the corpus, packages calling them, verified per backend):')
    L.append('')
    L.append('| # | Symbol | Calls | Packages | FFI | Extension |')
    L.append('|---:|---|---:|---:|:---:|:---:|')
    for i, x in enumerate(ins[:100], 1):
        L.append(f"| {i} | `{x['symbol']}` | {x['calls']} | {x['packages']} | {mark(x['ok_ffi'])} | {mark(x['ok_ext'])} |")
    L.append('')
    gaps = [x for x in ins if not all(x[f'ok_{b}'] for b in scope.BACKENDS)][:30]
    L.append('The most-called in-scope symbols not yet verified on both backends: '
             + ', '.join(f"`{x['symbol']}` {x['calls']}" + ('' if x['ok_ffi'] else ' (neither)' if not x['ok_ext'] else ' (ext only)')
                         + (' (FFI only)' if x['ok_ffi'] and not x['ok_ext'] else '') for x in gaps) + '.')
    L.append('')
    exc = [x for x in u['rows'] if x['excluded']][:12]
    L.append('The most-called excluded symbols: ' + ', '.join(f"`{x['symbol']}` {x['calls']} ({x['category']})" for x in exc)
             + '. Their calls are counted above but not weighted into the in-scope figures.')
    L.append('')
    return '\n'.join(L)


def deviations():
    """Every place where a fixture does not simply hold SciPy's value compared at the default tolerance, counted
    from the fixture files and tools/parity/crashes-special.json (docs/project/reference-deviations.md)."""
    crashes = scope.load_json(os.path.join(HERE, 'crashes-special.json'), {})
    rows = {'guard': [], 'undef': [], 'upstream': [], 'tol': [], 'env': [], 'shape': []}
    for fname in sorted(os.listdir(FIXDIR)):
        if not fname.endswith('.json'):
            continue
        data = json.load(open(os.path.join(FIXDIR, fname)))
        env = data.get('env') or {}
        rows['env'].append((fname, bool(env.get('pinned')), env.get('numpy', data.get('numpy')), env.get('scipy', data.get('scipy'))))
        mod = data.get('module')
        for b in data.get('blocks', []):
            n = sum(1 for v in b['expect'][0] if v == 'undef')
            if n:
                rows['undef'].append((f"{mod}.{b['fn']}", n, len(b['expect'][0])))
            if b.get('tol'):
                rows['tol'].append((f"{mod}.{b['fn']}", 'every row', b['tol']))
        for blk in data.get('calls', []):
            key = f"{mod}.{blk['fn']}"
            if blk.get('tol'):
                rows['tol'].append((key, 'every call', blk['tol']))
            calls = []
            for c in blk['cases']:
                calls.extend(c.get('calls', [c]))
            ups = {}
            nshape = sum(1 for c in calls if c.get('compare') == 'shape')
            if nshape:
                rows['shape'].append((key, nshape, len(calls)))
            for c in calls:
                if 'upstream' in c:
                    r = c['upstream']['rule']
                    ups.setdefault(r, set()).add(c.get('method', blk['fn']))
                if 'tol' in c:
                    rows['tol'].append((key, c.get('method', 'call'), c['tol']))
            for r, ms in sorted(ups.items()):
                rows['upstream'].append((key, r, ', '.join(sorted(ms))))
    for fn, rs in sorted(crashes.items()):
        rows['guard'].append((f'special.{fn}', len(rs)))
    return rows


def render_deviations(d):
    L = ['<!-- generated by tools/parity/report.py from tests/fixtures/parity/*.json and tools/parity/crashes-special.json;'
         ' do not edit by hand -->', '']
    L.append('**Fixture environment.** ' + '; '.join(
        f"`{f}`: {'pinned' if p else '**not pinned**'} (NumPy {n}, SciPy {s or '-'})" for f, p, n, s in d['env']) + '.')
    L.append('')
    L.append(f"**Inputs on which SciPy crashes or never returns** (`guard:`/`capped:`, Tessero returns NaN): "
             f"{sum(n for _, n in d['guard'])} fixture rows in {len(d['guard'])} functions: "
             + ', '.join(f'`{k}` {n}' for k, n in d['guard']) + '.')
    L.append('')
    L.append(f"**Rows on which SciPy's value is undefined** (not compared; AddressSanitizer or history dependence): "
             f"{sum(n for _, n, _ in d['undef'])} of {sum(t for _, _, t in d['undef'])} rows in {len(d['undef'])} functions: "
             + ', '.join(f'`{k}` {n}/{t}' for k, n, t in d['undef']) + '.')
    L.append('')
    L.append(f"**Calls whose NumPy values are unspecified** (only the shape and dtype are compared): "
             f"{sum(n for _, n, _ in d['shape'])} calls in {len(d['shape'])} functions: "
             + ', '.join(f'`{k}` {n}/{t}' for k, n, t in d['shape']) + '.')
    L.append('')
    L.append('**Upstream-bug rules** (the expectation is SciPy\'s documented contract; SciPy\'s own result is kept in the fixture):')
    L.append('')
    L.append('| Function | Rule | Methods |')
    L.append('|---|---|---|')
    for k, r, ms in d['upstream']:
        L.append(f'| `{k}` | {r} | {ms} |')
    L.append('')
    L.append('**Tolerances above the default** (rtol 1e-13 element-wise, 1e-12 for calls):')
    L.append('')
    L.append('| Function | Applies to | Tolerance |')
    L.append('|---|---|---|')
    seen = {}
    for k, where, t in d['tol']:
        rt = t.get('rtol') if isinstance(t, dict) else t
        why = t.get('why', '') if isinstance(t, dict) else ''
        key = (k, where, why)
        seen.setdefault(key, []).append(float(rt) if rt is not None else 0.0)
    for (k, where, why), rts in seen.items():
        rng = f'rtol {min(rts):.3g}' if len(rts) == 1 or min(rts) == max(rts) else f'rtol {min(rts):.3g} to {max(rts):.3g} ({len(rts)} calls)'
        L.append(f"| `{k}` | {where} | {rng}{': ' + why if why else ''} |")
    L.append('')
    return '\n'.join(L)


def render(m):
    L = ['<!-- generated by tools/parity/report.py from census.json, scope.yaml, registry.json and results.json;'
         ' do not edit by hand -->', '']
    L.append(f"Reference versions: NumPy {m['numpy']}, SciPy {m['scipy']}. A symbol counts as implemented on a "
             'backend only when its NumPy/SciPy fixture test (or, for functions older than the function registry, '
             'its PHPUnit suite) passed there in the recorded run. Aliases are not counted.')
    L.append('')
    L.append('| Scope | Symbols | Excluded (M3) | In scope | FFI (M2) | Extension (M2) | Verified FFI / ext (M4) | Parity (M5) |')
    L.append('|---|---:|---:|---:|---:|---:|---:|---:|')

    def row(name, t, bold=False):
        f = (lambda s: f'**{s}**') if bold else (lambda s: s)
        return (f"| {f(name)} | {t['symbols']} | {t['excluded']} ({t['m3']} %) | {t['in_scope']} | "
                f"{f(str(t['impl_ffi']) + ' (' + str(t['m2_ffi']) + ' %)')} | "
                f"{f(str(t['impl_ext']) + ' (' + str(t['m2_ext']) + ' %)')} | "
                f"{t['m4_ffi']} % / {t['m4_ext']} % | {t['m5']} % |")

    for g, mods in GROUPS:
        L.append(row(g, m['groups'][g], True))
        for mod in mods:
            if mod in m['modules']:
                L.append(row('&nbsp;&nbsp;' + mod, m['modules'][mod]))
    L.append('')
    L.append(f"The kernel function registry holds {m['registry_functions']} functions; their fixtures pass on the FFI "
             f"backend for {m['fixture_pass']['ffi']} and on the extension for {m['fixture_pass']['ext']}.")
    fails = [f"{b}: {', '.join(v)}" for b, v in m['fixture_fail'].items() if v]
    if fails:
        L.append('Failing fixtures: ' + '; '.join(fails) + '.')
    if m['fixture_missing']:
        L.append(f"Fixtures exist for functions Tessero does not implement yet: {', '.join(m['fixture_missing'])}.")
    if m['registry_untested']:
        L.append(f"Registry functions without a recorded fixture run: {', '.join(m['registry_untested'])}.")
    L.append('')
    L.append('Exclusions by category (definitions in `tools/parity/scope.yaml`):')
    L.append('')
    for mod, cs in sorted(m['excluded_by_category'].items()):
        L.append(f"- `{mod}`: " + ', '.join(f'{c} {n}' for c, n in sorted(cs.items())))
    L.append('')
    return '\n'.join(L)


def main():
    m, recs = compute()
    md = render(m)
    dev = render_deviations(deviations())
    uu = usage(recs)
    um = render_usage(uu) if uu else ''
    js = json.dumps({k: v for k, v in m.items() if k != 'problems'}, indent=1, sort_keys=True) + '\n'
    if '--check' in sys.argv:
        stale = [p for p, t in ((OUT_MD, md), (OUT_JSON, js), (OUT_DEV, dev), (OUT_USAGE, um)) if not os.path.exists(p) or open(p).read() != t]
        if stale:
            print('out of date:', ' '.join(os.path.relpath(p, ROOT) for p in stale))
            sys.exit(1)
        return
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, 'w').write(md)
    open(OUT_JSON, 'w').write(js)
    open(OUT_DEV, 'w').write(dev)
    open(OUT_USAGE, 'w').write(um)
    for p in m['problems']:
        print('problem:', p)
    g = m['groups']
    print(f"NumPy core: FFI {g['NumPy core']['m2_ffi']} %, ext {g['NumPy core']['m2_ext']} %; "
          f"SciPy: FFI {g['SciPy']['m2_ffi']} %, ext {g['SciPy']['m2_ext']} %; M5 numpy {g['NumPy core']['m5']} %, scipy {g['SciPy']['m5']} %")
    if m.get('usage'):
        ua = m['usage']
        print(f"Usage-weighted (in-scope calls verified on both backends): NumPy {ua['groups']['NumPy core']['u_both']} %, "
              f"SciPy {ua['groups']['SciPy']['u_both']} %, together {ua['all']['u_both']} %")
    if '--record' in sys.argv:
        label = sys.argv[sys.argv.index('--record') + 1]
        new = not os.path.exists(HISTORY) or os.path.getsize(HISTORY) == 0
        with open(HISTORY, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['date', 'phase', 'm2_numpy_ffi', 'm2_numpy_ext', 'm2_scipy_ffi', 'm2_scipy_ext', 'm5_numpy', 'm5_scipy',
                            'm2_special_ffi', 'm2_special_ext', 'm2_stats_ffi', 'm2_stats_ext'])
            sp, st = m['modules']['scipy.special'], m['modules']['scipy.stats']
            w.writerow([datetime.date.today().isoformat(), label, g['NumPy core']['m2_ffi'], g['NumPy core']['m2_ext'],
                        g['SciPy']['m2_ffi'], g['SciPy']['m2_ext'], g['NumPy core']['m5'], g['SciPy']['m5'],
                        sp['m2_ffi'], sp['m2_ext'], st['m2_ffi'], st['m2_ext']])


if __name__ == '__main__':
    main()
