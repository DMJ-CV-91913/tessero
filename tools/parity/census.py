#!/usr/bin/env python3
"""Census of the public NumPy / SciPy API: the denominator of Tessero's coverage figures.

    python3 tools/parity/census.py            # writes tools/parity/census.json
    python3 tools/parity/census.py --check    # also checks tools/parity/scope.yaml maps every symbol

A symbol is a public name of a module (its __all__, or dir() without underscores when there is no
__all__), a method of numpy.ndarray or numpy.random.Generator, or a scipy.stats distribution.
Two names bound to the same object are recorded once as canonical and once as an alias.
"""
import importlib, inspect, json, os, sys
import numpy as np, scipy

HERE = os.path.dirname(os.path.abspath(__file__))

MODULES = [
    'numpy', 'numpy.linalg', 'numpy.fft', 'numpy.polynomial', 'numpy.polynomial.polynomial',
    'numpy.polynomial.chebyshev', 'numpy.polynomial.legendre', 'numpy.polynomial.laguerre',
    'numpy.polynomial.hermite', 'numpy.polynomial.hermite_e', 'numpy.random',
    'scipy.special', 'scipy.stats', 'scipy.linalg', 'scipy.optimize', 'scipy.sparse', 'scipy.sparse.linalg',
    'scipy.sparse.csgraph', 'scipy.interpolate', 'scipy.signal', 'scipy.signal.windows', 'scipy.integrate',
    'scipy.spatial', 'scipy.spatial.distance', 'scipy.cluster.vq', 'scipy.cluster.hierarchy', 'scipy.ndimage',
    'scipy.fft', 'scipy.constants',
]


def public(mod):
    names = getattr(mod, '__all__', None)
    if names is None:
        names = [n for n in dir(mod) if not n.startswith('_')]
    return sorted(n for n in set(names) if not n.startswith('_') and hasattr(mod, n))


def kind_of(obj, modname):
    import scipy.stats as st
    if modname == 'scipy.stats' and isinstance(obj, (st.rv_continuous, st.rv_discrete)):
        return 'distribution'
    if inspect.ismodule(obj):
        return 'module'
    if inspect.isclass(obj):
        return 'class'
    if isinstance(obj, np.ufunc):
        return 'ufunc'
    if callable(obj):
        return 'function'
    return 'constant'


def census():
    out = []
    for modname in MODULES:
        mod = importlib.import_module(modname)
        seen = {}
        for name in public(mod):
            obj = getattr(mod, name)
            k = kind_of(obj, modname)
            if k == 'module':
                continue                                  # submodules are censused separately (or out of scope)
            rec = {'module': modname, 'name': name, 'kind': k}
            key = id(obj) if k not in ('constant',) else None
            canonical = getattr(obj, '__name__', None)
            if key is not None and key in seen:
                first = seen[key]
                # prefer the object's own name as canonical
                if canonical == name and first['name'] != name:
                    first['alias_of'] = name
                    seen[key] = rec
                else:
                    rec['alias_of'] = first['name']
            elif key is not None:
                seen[key] = rec
            if modname == 'scipy.stats' and k == 'distribution':
                rec['discrete'] = isinstance(obj, __import__('scipy.stats', fromlist=['rv_discrete']).rv_discrete)
                rec['shapes'] = obj.shapes.split(', ') if obj.shapes else []
            out.append(rec)
    for cls, modname in ((np.ndarray, 'numpy.ndarray'), (np.random.Generator, 'numpy.random.Generator')):
        for name in sorted(n for n in dir(cls) if not n.startswith('_')):
            out.append({'module': modname, 'name': name, 'kind': 'method'})
    return out


def main():
    data = {'numpy': np.__version__, 'scipy': scipy.__version__, 'symbols': census()}
    with open(os.path.join(HERE, 'census.json'), 'w') as f:
        json.dump(data, f, indent=0, sort_keys=True)
    counts = {}
    for s in data['symbols']:
        counts[s['module']] = counts.get(s['module'], 0) + 1
    for m, n in counts.items():
        print(f'{m:28s} {n:5d}')
    print(f"{'total':28s} {len(data['symbols']):5d}")
    if '--check' in sys.argv:
        sys.path.insert(0, HERE)
        import scope
        sys.exit(scope.check(data))


if __name__ == '__main__':
    main()
