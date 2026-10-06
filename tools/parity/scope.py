"""Classify every census symbol (implemented / excluded / planned) per backend.

Inputs: census.json, scope.yaml, registry.json (the kernel registry's functions, written by gen-facades.php),
spec/*.yaml (the `ref:` of element-wise registry functions), results.json (written by
tools/parity/run-tests.php: fixture outcome per backend and registry function, plus the legacy suites),
api.json (tools/api-dump.php).

A registry function "<module>.<name>" references <PREFIX[module]>.<name> unless its spec entry names another
`ref:`; it implements that symbol on a backend when its fixtures passed there in the last run.
"""
import glob, json, os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

ALIAS = {
    "N": "Tessero\\NDArray", "M": "Tessero\\Math", "T": "Tessero\\Tessero", "L": "Tessero\\Linalg\\Linalg",
    "F": "Tessero\\Fft\\Fft", "R": "Tessero\\Random\\Generator", "S": "Tessero\\Sparse\\CsrMatrix",
    "Min": "Tessero\\Optimize\\Minimize", "Root": "Tessero\\Optimize\\Root", "LS": "Tessero\\Optimize\\LeastSquares",
    "LP": "Tessero\\Optimize\\LinearProgramming", "ASG": "Tessero\\Optimize\\Assignment", "Const": "Tessero\\Constants", "Npy": "Tessero\\Io\\Npy", "Int": "Tessero\\Integrate\\Quad", "KD": "Tessero\\Spatial\\KDTree",
    "NpC": "Tessero\\Np", "XNpC": "Tessero\\Ext\\Np",
    "XN": "Tessero\\Ext\\NDArray", "XM": "Tessero\\Ext\\Math", "XE": "Tessero\\Ext\\Engine",
    "XR": "Tessero\\Ext\\Random\\Generator", "XF": "Tessero\\Ext\\Fft\\Fft", "DT": "Tessero\\Datetime\\Datetime",
}
BACKENDS = ('ffi', 'ext')


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


PREFIX = {'special': 'scipy.special', 'stats': 'scipy.stats', 'np': 'numpy', 'random': 'numpy.random.Generator', 'linalg': 'numpy.linalg', 'slinalg': 'scipy.linalg',
          'signal': 'scipy.signal', 'windows': 'scipy.signal.windows', 'integrate': 'scipy.integrate', 'interpolate': 'scipy.interpolate',
          'optimize': 'scipy.optimize', 'ndimage': 'scipy.ndimage', 'spatial': 'scipy.spatial', 'distance': 'scipy.spatial.distance',
          'sparse': 'scipy.sparse', 'csgraph': 'scipy.sparse.csgraph'}


def spec_entries():
    """Registry functions: {(module, name): [{'name': registry name}, ...]} keyed by the reference symbol."""
    refs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, 'spec', '**', '*.yaml'), recursive=True)):
        doc = load_yaml(path) or {}
        for e in doc.get('functions', []):
            if isinstance(e.get('ref'), str):
                refs[f"{doc['module']}.{e['name']}"] = [e['ref']]
            elif e.get('ref'):
                refs[f"{doc['module']}.{e['name']}"] = list(e['ref'])
    registry = load_json(os.path.join(HERE, 'registry.json'), {}).get('functions', {})
    out = {}
    for name in registry:
        mod, _, fn = name.partition('.')
        for ref in refs.get(name, [f'{PREFIX.get(mod, mod)}.{fn}']):
            m, _, n = ref.rpartition('.')
            out.setdefault((m, n), []).append({'name': name})
    return out


def api_methods(api):
    have = set()
    for c in api:
        for m in c.get('methods', []):
            have.add(c['class'] + '::' + (m['name'] if isinstance(m, dict) else m))
        for k in c.get('constants', []):          # class constants (e.g. Tessero\Constants scalars)
            have.add(c['class'] + '::' + k)
    return have


def classify(census, results=None, api=None):
    scope = load_yaml(os.path.join(HERE, 'scope.yaml'))
    results = results if results is not None else load_json(os.path.join(HERE, 'results.json'), {})
    api = api if api is not None else load_json(os.path.join(HERE, 'api.json'), [])
    have = api_methods(api)
    specs = spec_entries()
    suites = results.get('suites', {})

    excl = {}
    for rule in scope.get('exclude', []):
        mod = rule['module']
        names = rule['names']
        for s in census['symbols']:
            if s['module'] != mod:
                continue
            if (names == '*' and s['name'] not in rule.get('except', [])) or (isinstance(names, list) and s['name'] in names):
                excl[(mod, s['name'])] = rule['category']
    phase_of = {}
    for ph, mods in (scope.get('phase_names') or {}).items():
        for mod, names in mods.items():
            for n in names:
                phase_of[(mod, n)] = int(ph)

    recs = []
    problems = []
    for s in census['symbols']:
        key = (s['module'], s['name'])
        r = dict(s)
        r['status'] = {}
        r['via'] = {}
        r['verified'] = {b: False for b in BACKENDS}
        for b in BACKENDS:
            r['status'][b] = 'planned'
        # registry functions
        for e in specs.get(key, []):
            ok_all = True
            for b in BACKENDS:
                ids = e.get('tests') or [e['name']]
                outcome = [results.get(b, {}).get(i) for i in ids]
                if outcome and all(o == 'pass' for o in outcome):
                    r['status'][b] = 'implemented'
                    r['via'][b] = e['name']
                    r['verified'][b] = True
                else:
                    ok_all = False
            if not ok_all and all(results.get(b) for b in BACKENDS):
                problems.append(f"{key[0]}.{key[1]}: registry function {e['name']} has failing or missing fixture tests")
        # legacy map
        leg = (scope.get('legacy') or {}).get(s['module'], {}).get(s['name'])
        if leg:
            tests = scope['legacy_tests'][leg['tests']] if 'tests' in leg else []
            # without a recorded suite run the legacy function counts (M2) but is not verified (M4)
            passed_suites = all(suites.get(t) == 'pass' for t in tests) if suites else None
            for b in BACKENDS:
                refs = leg.get(b, [])
                if not refs or r['status'][b] == 'implemented':
                    continue
                passed = passed_suites
                if leg.get('fixtures'):   # verified by NumPy/SciPy fixtures on each backend (run-tests.php)
                    rb = results.get(b)
                    passed = all(rb.get(f) == 'pass' for f in leg['fixtures']) if rb else None
                    tests = leg['fixtures']
                full = [ALIAS[ref.split('::')[0]] + '::' + ref.split('::')[1] for ref in refs]
                missing = [f for f in full if have and f not in have]
                if missing:
                    problems.append(f"{key[0]}.{key[1]}: legacy reference(s) missing from the API: {missing}")
                    continue
                if passed is False:
                    problems.append(f"{key[0]}.{key[1]}: legacy suite(s) {tests} did not pass")
                    continue
                r['status'][b] = 'implemented'
                r['via'][b] = ' '.join(refs)
                r['verified'][b] = passed is True
        # scipy.constants scalars: implemented when Tessero\Constants::<name> exists and the values fixture passed
        if s['module'] == 'scipy.constants' and s['kind'] == 'constant':
            ref = 'Tessero\\Constants::' + s['name']
            for b in BACKENDS:
                if r['status'][b] == 'implemented' or not have or ref not in have:
                    continue
                if (results.get(b, {}) or {}).get('constants.values') == 'pass':
                    r['status'][b] = 'implemented'
                    r['via'][b] = 'Constants'
                    r['verified'][b] = True
        if key in excl:
            if any(v == 'implemented' for v in r['status'].values()):
                problems.append(f"{key[0]}.{key[1]}: both implemented and excluded")
            for b in BACKENDS:
                r['status'][b] = 'excluded'
            r['category'] = excl[key]
        r['phase'] = phase_of.get(key, scope['phases'].get(s['module']))
        recs.append(r)

    # aliases inherit their canonical's status
    by = {(r['module'], r['name']): r for r in recs}
    for r in recs:
        if 'alias_of' in r:
            c = by.get((r['module'], r['alias_of']))
            if c:
                for b in BACKENDS:
                    if r['status'][b] != 'implemented':
                        r['status'][b] = c['status'][b]
                        r['verified'][b] = c['verified'][b]
                if 'category' in c:
                    r['category'] = c['category']
    return recs, problems


def check(census):
    recs, problems = classify(census)
    for p in problems:
        print('scope:', p)
    unmapped = [r for r in recs if r['phase'] is None and 'excluded' not in r['status'].values()
                and 'implemented' not in r['status'].values()]
    for r in unmapped:
        print(f"scope: {r['module']}.{r['name']} has no phase")
    return 1 if problems or unmapped else 0
