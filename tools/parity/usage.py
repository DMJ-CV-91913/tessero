#!/usr/bin/env python3
"""How often real code calls each NumPy and SciPy function: the weights of the usage-weighted coverage figures.

    python3 tools/parity/usage.py            # clone the corpus (pinned tags) and write tools/parity/usage.json
    python3 tools/parity/usage.py --check    # rescan (cloning what is not cached) and fail when usage.json is out of date

The corpus is the library code of widely used scientific Python packages at pinned release tags (CORPUS below;
NumPy and SciPy themselves are left out). Tests, docs, examples, benchmarks and vendored copies are skipped, so
the counts measure what libraries call in production code, not what their test suites exercise.

What is counted: call sites whose callee resolves statically to a NumPy or SciPy symbol of the census
(tools/parity/census.json), through the file's imports:

    import numpy as np; np.linalg.norm(x)         -> numpy.linalg.norm
    from scipy import stats; stats.norm.cdf(x)    -> scipy.stats.norm (the distribution object, once per call)
    from scipy.special import gammaln; gammaln(x) -> scipy.special.gammaln

Aliases count for their canonical symbol (numpy.abs -> numpy.absolute). Method calls on arrays (x.sum()) are
not counted: which object a method is called on cannot be decided statically, so numpy.ndarray has no usage
weight. References that are not calls (np.float64 passed as a dtype, np.nan) are not counted either.
"""
import ast, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'usage.json')
CACHE = os.environ.get('USAGE_CACHE', '/tmp/tessero-usage')

# (name, GitHub repository, release tag, package directory in the repository)
CORPUS = [
    ('scikit-learn', 'scikit-learn/scikit-learn', '1.9.1', 'sklearn'),
    ('pandas', 'pandas-dev/pandas', 'v3.0.6', 'pandas'),
    ('statsmodels', 'statsmodels/statsmodels', 'v0.15.0', 'statsmodels'),
    ('scikit-image', 'scikit-image/scikit-image', 'v0.26.0', 'src/skimage'),
    ('astropy', 'astropy/astropy', 'v8.0.1', 'astropy'),
    ('networkx', 'networkx/networkx', 'networkx-3.7', 'networkx'),
    ('matplotlib', 'matplotlib/matplotlib', 'v3.11.2', 'lib/matplotlib'),
    ('xarray', 'pydata/xarray', 'v2026.07.0', 'xarray'),
    ('seaborn', 'mwaskom/seaborn', 'v0.13.2', 'seaborn'),
    ('librosa', 'librosa/librosa', '1.0.0', 'librosa'),
    ('arviz-stats', 'arviz-devs/arviz-stats', 'v1.3.3', 'src/arviz_stats'),   # arviz 1.x's numerics
    ('mne', 'mne-tools/mne-python', 'v1.13.2', 'mne'),
    ('nilearn', 'nilearn/nilearn', '0.14.1', 'nilearn'),
    ('pingouin', 'raphaelvallat/pingouin', 'v0.7.0', None),
    ('lifelines', 'CamDavidsonPilon/lifelines', 'v0.30.3', 'lifelines'),
    ('pvlib', 'pvlib/pvlib-python', 'v0.16.1', 'pvlib'),
    ('pymc', 'pymc-devs/pymc', 'v6.3.2', 'pymc'),
    ('scanpy', 'scverse/scanpy', '1.12.4', None),
]

SKIP_DIRS = {'tests', 'test', 'testing', '_testing', 'conftest', 'benchmarks', 'asv_bench', 'doc', 'docs',
             'examples', 'externals', '_vendor', 'vendor', 'extern', '_externals'}


def clone(repo, tag):
    dst = os.path.join(CACHE, repo.replace('/', '__') + '@' + tag)
    if not os.path.isdir(dst):
        os.makedirs(CACHE, exist_ok=True)
        subprocess.run(['git', '-c', 'advice.detachedHead=false', 'clone', '-q', '--depth', '1', '--branch', tag,
                        f'https://github.com/{repo}', dst], check=True)
    commit = subprocess.run(['git', '-C', dst, 'rev-parse', 'HEAD'], check=True, capture_output=True,
                            text=True).stdout.strip()
    return dst, commit


def package_dir(root, sub, name):
    if sub:
        return os.path.join(root, sub)
    for cand in (name, 'src/' + name, name.replace('-', '_'), 'src/' + name.replace('-', '_')):
        if os.path.isdir(os.path.join(root, cand)):
            return os.path.join(root, cand)
    sys.exit(f'{name}: package directory not found in {root}')


def py_files(top):
    for d, dirs, files in os.walk(top):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith('.'))
        for f in sorted(files):
            if f.endswith('.py') and not f.startswith('test_') and f != 'conftest.py':
                yield os.path.join(d, f)


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def scan_file(path, resolve):
    try:
        tree = ast.parse(open(path, encoding='utf-8', errors='replace').read())
    except (SyntaxError, ValueError):
        return {}
    alias = {}                                            # local name -> fully qualified module or symbol
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split('.')[0] in ('numpy', 'scipy'):
                    if a.asname:
                        alias[a.asname] = a.name
                    else:
                        alias[a.name.split('.')[0]] = a.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0 and n.module.split('.')[0] in ('numpy', 'scipy'):
            for a in n.names:
                if a.name != '*':
                    alias[a.asname or a.name] = n.module + '.' + a.name
    if not alias:
        return {}
    counts = {}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        parts = dotted(n.func)
        if not parts or parts[0] not in alias:
            continue
        full = alias[parts[0]].split('.') + parts[1:]
        # the longest prefix that is a census symbol: scipy.stats.norm.cdf -> scipy.stats.norm
        for k in range(len(full), 1, -1):
            key = resolve.get('.'.join(full[:k]))
            if key:
                counts[key] = counts.get(key, 0) + 1
                break
    return counts


def census_index():
    census = json.load(open(os.path.join(HERE, 'census.json')))
    resolve = {}
    for s in census['symbols']:
        full = s['module'] + '.' + s['name']
        resolve[full] = s['module'] + '.' + s.get('alias_of', s['name'])
    return resolve, census


def scan():
    resolve, census = census_index()
    counts, packages, corpus = {}, {}, []
    for name, repo, tag, sub in CORPUS:
        root, commit = clone(repo, tag)
        top = package_dir(root, sub, name)
        nfiles, ncalls, used = 0, 0, set()
        for f in py_files(top):
            nfiles += 1
            for k, v in scan_file(f, resolve).items():
                counts[k] = counts.get(k, 0) + v
                ncalls += v
                used.add(k)
        for k in used:
            packages[k] = packages.get(k, 0) + 1
        corpus.append({'name': name, 'repo': repo, 'tag': tag, 'commit': commit, 'files': nfiles, 'calls': ncalls})
    symbols = {k: {'calls': counts[k], 'packages': packages[k]} for k in sorted(counts)}
    return {'numpy': census.get('numpy'), 'scipy': census.get('scipy'), 'corpus': corpus,
            'total_calls': sum(counts.values()), 'symbols': symbols}


def main():
    check = '--check' in sys.argv
    data = scan()
    text = json.dumps(data, indent=1, sort_keys=True) + '\n'
    if check:
        if not os.path.exists(OUT) or open(OUT).read() != text:
            print('out of date: tools/parity/usage.json')
            sys.exit(1)
        return
    open(OUT, 'w').write(text)
    print(f"{len(data['corpus'])} packages, {sum(c['files'] for c in data['corpus'])} files, "
          f"{data['total_calls']} NumPy/SciPy call sites, {len(data['symbols'])} distinct symbols")


if __name__ == '__main__':
    main()
