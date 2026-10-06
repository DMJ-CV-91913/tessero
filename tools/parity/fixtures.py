#!/usr/bin/env python3
"""Generate NumPy/SciPy reference fixtures for the registry functions in spec/*.yaml.

    python3 tools/parity/fixtures.py            # every module
    python3 tools/parity/fixtures.py special    # one module

Writes tests/fixtures/parity/<module>.json: for each function, the input columns and SciPy's outputs,
computed with the pinned NumPy/SciPy (tools/parity/requirements.txt). Floats are stored with Python's
shortest round-trip repr, so both backends read the exact doubles; NaN and infinities are strings.

Inputs: a grid of representative values per argument (by argument role, see ROLE), the full cross product
for one or two arguments and a seeded random subset for more, plus random draws over wide ranges, plus
NaN/inf probes. A function's `domain:` in the spec overrides the grids. Functions with an integer loop get a
second block with int64 inputs so the integer loop (NumPy's choice for integer arrays) is checked too.
"""
import itertools, json, math, os, sys, time, warnings

import numpy as np
import fixture_env
import scipy
import scipy.special as sc
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'tests', 'fixtures', 'parity')

REAL = [-50.0, -10.5, -3.0, -2.5, -1.0, -0.5, -1e-10, 0.0, 1e-10, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 7.3, 10.0, 25.5, 100.0, 1e3]
POS = [1e-10, 0.01, 0.5, 1.0, 1.5, 3.0, 10.0, 50.5, 200.0, 1e4]
PROB = [-0.5, 0.0, 1e-300, 1e-100, 1e-20, 1e-5, 1e-3, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 1 - 1e-10, 1.0, 1.5]
COUNT = [-1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 2.5]
INTS = [-2, -1, 0, 1, 2, 3, 5, 10, 20, 50]

ROLE = {
    'p': PROB, 'q': PROB, 'y': PROB,
    'n': COUNT, 'k': COUNT, 'm': COUNT, 'd': COUNT,
    'a': POS + [-1.5, 0.0], 'b': POS + [-1.5, 0.0], 'c': POS + [-1.5, 0.0],
    'v': [-2.5, -1.0, 0.0, 0.5, 1.0, 2.0, 3.5, 10.0, 50.0],
    'df': POS, 'dfn': POS, 'dfd': POS, 'nc': [0.0, 0.5, 1.0, 5.0, 20.0], 'nu': POS, 'alpha': POS + [-0.5], 'beta': POS,
    'lmbda': [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0], 'lmb': [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    'std': POS, 'sigma': POS, 'gamma': POS, 'delta': [0.0, 0.5, 1.0, 3.0],
}


def enc(v):
    if v is None:
        return 'undef'
    if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
        return int(v)   # integer columns stay integers: the runner then calls the integer loop, as SciPy did
    v = float(v)
    if math.isnan(v):
        return 'nan'
    if math.isinf(v):
        return 'inf' if v > 0 else '-inf'
    return v


def grid_for(fn, arg):
    dom = (fn.get('domain') or {}).get(arg)
    if dom is not None:
        return [float(x) for x in dom]
    return ROLE.get(arg, REAL)


def random_values(rng, arg, n):
    base = ROLE.get(arg, REAL)
    if base is PROB:
        return list(rng.uniform(0, 1, n))
    if base is COUNT:
        return list(rng.integers(0, 30, n).astype(float))
    if min(base) >= 0:
        return list(10 ** rng.uniform(-5, 4, n))
    return list(rng.uniform(-30, 30, n))


def cases_for(fn, rng, limit):
    args = fn['args']
    grids = [grid_for(fn, a) for a in args]
    nin = len(args)
    if nin == 1:
        rows = [[g] for g in grids[0]]
    else:
        prod = list(itertools.product(*grids))
        if len(prod) > limit:
            idx = rng.choice(len(prod), size=limit, replace=False)
            prod = [prod[i] for i in sorted(idx)]
        rows = [list(r) for r in prod]
    if not fn.get('domain'):
        rnd = [random_values(rng, a, 60) for a in args]
        rows += [list(r) for r in zip(*rnd)]
        # NaN and infinity probes in each position
        for k in range(nin):
            for probe in (math.nan, math.inf, -math.inf):
                r = [g[len(g) // 2] for g in grids]
                r[k] = probe
                rows.append(r)
    return rows


def row_key(r):
    return tuple('nan' if math.isnan(float(v)) else float(v) for v in r)


def load_crashes(module):
    """Rows on which SciPy itself crashes or hangs (tools/parity/crash_scan.py). SciPy cannot be asked about
    them, so the fixture records NaN, which is what the kernel's `guard:` returns there."""
    path = os.path.join(ROOT, 'tools', 'parity', f'crashes-{module}.json')
    if not os.path.exists(path):
        return {}
    return {k: {row_key(r) for r in v} for k, v in json.load(open(path)).items()}


def evaluate(u, cols, dtypes, bad=frozenset(), undef=frozenset()):
    """SciPy's outputs; rows in `bad` (SciPy crashes) become NaN, rows in `undef` (SciPy's code has undefined
    behaviour there, ub_scan.py) become None (recorded as "undef" and not compared)."""
    keep = list(range(len(cols[0])))
    if bad or undef:
        keep = [i for i in keep if row_key([c[i] for c in cols]) not in bad | undef]
    arrs = [np.array([c[i] for i in keep], dtype=t) for c, t in zip(cols, dtypes)]
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        try:
            res = u(*arrs)
        except Exception:  # noqa: BLE001 - SciPy raises for some rows (sf_error 'memory'): evaluate row by row
            res = None
    if res is None:
        # rows on which SciPy raises (for the whole call) get NaN, what Tessero returns there
        per = []
        for r in range(len(keep)):
            with warnings.catch_warnings(), np.errstate(all='ignore'):
                warnings.simplefilter('ignore')
                try:
                    v = u(*[a[r:r + 1] for a in arrs])
                    v = v if isinstance(v, tuple) else (v,)
                    per.append([float(np.asarray(x).ravel()[0]) for x in v])
                except Exception:  # noqa: BLE001
                    per.append(None)
        nout = next((len(p) for p in per if p is not None), 1)
        res = tuple(np.array([p[k] if p is not None else np.nan for p in per]) for k in range(nout))
    if not isinstance(res, tuple):
        res = (res,)
    res = [np.asarray(r, dtype=np.float64) for r in res]
    if len(keep) == len(cols[0]):
        return res
    full = []
    for r in res:
        f = np.full(len(cols[0]), np.nan, dtype=object)
        f[keep] = r
        for i in range(len(cols[0])):
            if row_key([c[i] for c in cols]) in undef:
                f[i] = None
        full.append(f)
    return full


FRESH = r"""
import sys, importlib, warnings, numpy as np
warnings.simplefilter('ignore'); np.seterr(all='ignore')
rev = sys.argv[4] == 'reverse'
z = np.load(sys.argv[1]); arrs = [z[f'a{k}'][::-1] if rev else z[f'a{k}'] for k in range(len(z.files))]
mod, name = sys.argv[2].rsplit('.', 1)
r = getattr(importlib.import_module(mod), name)(*arrs); r = r if isinstance(r, tuple) else (r,)
np.save(sys.argv[3], np.stack([np.asarray(x, dtype=np.float64)[::-1] if rev else np.asarray(x, dtype=np.float64) for x in r]))
"""

# re-evaluations that expose a dependence on memory the code never initialised: the reverse order in a fresh
# interpreter (a different heap history), and glibc's MALLOC_PERTURB_, which fills every allocation with the
# complement of a byte pattern: 165 and 90 give two garbage patterns, 255 zero-filled allocations (what
# Tessero's value-initialised work arrays hold, THIRD_PARTY_NOTICES.md)
REEVAL = (('reverse', None), ('forward', '165'), ('forward', '90'), ('forward', '255'))


def history_dependent(name, u, cols, dtypes, out, skip):
    """Rows whose SciPy value depends on memory its code never initialises (xsf's spheroidal routines read work
    arrays they do not fill for some inputs, so a value can change with the heap left by earlier rows or by the
    allocator): evaluate the kept rows again in fresh interpreters (reversed, and forward under three
    MALLOC_PERTURB_ patterns) and return the rows whose outputs differ from the first evaluation in any of them.
    A re-evaluation that crashes (glibc detects heap corruption: SciPy's code wrote outside its buffers) is split
    until the crashing rows are isolated; those rows are undefined as well. The rows are recorded as "undef" like
    the rows ub_scan.py finds. Rows SciPy crashes on in the first evaluation stay excluded."""
    import subprocess, tempfile
    keep = [i for i in range(len(cols[0])) if row_key([c[i] for c in cols]) not in skip]
    diff = set()

    def run(rows, order, perturb, d):
        """{row index: outputs} for these rows, or the set of rows that crash when evaluated alone"""
        np.savez(os.path.join(d, 'in.npz'), **{f'a{k}': np.array([c[i] for i in rows], dtype=t)
                                               for k, (c, t) in enumerate(zip(cols, dtypes))})
        env = dict(os.environ)
        if perturb:
            env['MALLOC_PERTURB_'] = perturb
        r = subprocess.run([sys.executable, '-c', FRESH, os.path.join(d, 'in.npz'), name, os.path.join(d, 'out.npy'), order],
                           capture_output=True, env=env)
        if r.returncode == 0:
            again = np.load(os.path.join(d, 'out.npy'))
            return {row: again[:, j] for j, row in enumerate(rows)}, set()
        if r.returncode > 0 and b'Traceback' in r.stderr:
            sys.exit(f'{name}: the fresh-process re-evaluation failed ({r.returncode}): {r.stderr.decode()[-300:]}')
        if len(rows) == 1:
            return {}, set(rows)
        h = len(rows) // 2
        a, ca = run(rows[:h], order, perturb, d)
        b, cb = run(rows[h:], order, perturb, d)
        a.update(b)
        return a, ca | cb

    with tempfile.TemporaryDirectory() as d:
        for order, perturb in REEVAL:
            vals, crashed = run(keep, order, perturb, d)
            if crashed:
                sys.stderr.write(f'  {name}: {len(crashed)} rows crash SciPy under {order}/{perturb} -> undef\n')
            diff |= {row_key([c[i] for c in cols]) for i in crashed}
            for i, v in vals.items():
                for k, o in enumerate(out):
                    a, b = float(o[i]), float(v[k])
                    same = (a == b and math.copysign(1, a) == math.copysign(1, b)) or (a != a and b != b)
                    if not same:
                        diff.add(row_key([c[i] for c in cols]))
    return diff


def resolve_ref(ref):
    """scipy.special.<name> or a private ufunc such as scipy.special._ufuncs._beta_pdf"""
    import importlib
    if not ref.startswith('scipy.special.'):
        return None
    mod, name = ref.rsplit('.', 1)
    return getattr(importlib.import_module(mod), name, None)


def module_fixtures(path):
    spec = yaml.safe_load(open(path))
    module = spec['module']
    if spec.get('internal'):
        return   # private ufuncs for the C++ API: verified through the scipy.stats fixtures that use them
    rng = np.random.default_rng(20260928)
    crashes = load_crashes(module)
    ub_names = {fn['name'] for fn in spec.get('functions', []) if fn.get('reference_ub') == 'asan' and 'impl' in fn}
    undefs = {}
    if ub_names:
        import ub_scan
        undefs = {k: {row_key(r) for r in v} for k, v in ub_scan.scan(module, ub_names).items()}
    blocks = []
    t0 = time.time()
    for fn in spec.get('functions', []):
        if 'impl' not in fn:
            continue
        ref = fn['ref']
        u = resolve_ref(ref)
        if u is None:
            continue
        limit = int(fn.get('cases', 1500))
        rows = cases_for(fn, rng, limit)
        cols = [list(c) for c in zip(*rows)]
        t = time.time()
        if os.environ.get('PARITY_TRACE'):
            sys.stderr.write(fn['name'] + '\n'); sys.stderr.flush()
        only_int = fn.get('integer_only')
        dtypes = [np.int64 if only_int and k in only_int else np.float64 for k in range(len(cols))]
        if only_int:
            keep = [i for i in range(len(rows)) if all(math.isfinite(rows[i][k]) for k in only_int)]
            cols = [[c[i] if k not in only_int else int(c[i]) for i in keep] for k, c in enumerate(cols)]
        bad = crashes.get(fn['name'], frozenset())
        und = undefs.get(fn['name'], frozenset())
        if (bad - und) and not (fn.get('guard') or fn.get('capped')):
            sys.exit(f"{fn['name']}: SciPy crashes on {len(bad - und)} fixture rows but the spec has no guard")
        out = evaluate(u, cols, dtypes, bad, und)
        if fn['name'] in ub_names:
            hist = history_dependent(ref, u, cols, dtypes, out, bad | und)
            if hist:
                sys.stderr.write(f'  {fn["name"]}: {len(hist)} history-dependent rows -> undef\n')
                und = und | hist
                out = evaluate(u, cols, dtypes, bad, und)
        blocks.append({'fn': fn['name'], 'int': bool(only_int), 'args': [[enc(v) for v in c] for c in cols],
                       'expect': [[enc(v) for v in o.ravel()] for o in out], 'tol': fn.get('tol', {}),
                       'undef': sum(1 for v in out[0].ravel() if v is None)})
        if fn.get('int_impl'):
            ia = fn['int_args']
            irows = []
            for r in rows[:400]:
                r = list(r)
                ok = True
                for k in ia:
                    if not math.isfinite(r[k]):
                        ok = False
                    r[k] = int(r[k]) if math.isfinite(r[k]) else 0
                if ok:
                    irows.append(r)
            for n in INTS:
                r = [grid_for(fn, a)[len(grid_for(fn, a)) // 2] for a in fn['args']]
                for k in ia:
                    r[k] = n
                irows.append(r)
            icols = [list(c) for c in zip(*irows)]
            idt = [np.int64 if k in ia else np.float64 for k in range(len(icols))]
            iout = evaluate(u, icols, idt, bad, und)
            blocks.append({'fn': fn['name'], 'int': True, 'args': [[enc(v) for v in c] for c in icols],
                           'expect': [[enc(v) for v in o.ravel()] for o in iout], 'tol': fn.get('tol', {})})
        dt = time.time() - t
        if dt > 2:
            sys.stderr.write(f'  {fn["name"]}: {dt:.1f}s\n')
    os.makedirs(OUT, exist_ok=True)
    data = {'module': module, 'numpy': np.__version__, 'scipy': scipy.__version__, 'env': fixture_env.env(), 'blocks': blocks}
    fixture_env.require_pinned()
    with open(os.path.join(OUT, f'{module}.json'), 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    n = sum(len(b['args'][0]) if b['args'] else 0 for b in blocks)
    print(f'{module}: {len(blocks)} blocks, {n} cases, {time.time() - t0:.1f}s')


def main():
    want = set(sys.argv[1:])
    for fname in sorted(os.listdir(os.path.join(ROOT, 'spec'))):
        if fname.endswith('.yaml') and (not want or fname[:-5] in want):
            module_fixtures(os.path.join(ROOT, 'spec', fname))


if __name__ == '__main__':
    main()
