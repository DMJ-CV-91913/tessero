#!/usr/bin/env python3
"""One-time bootstrap of spec/special.yaml from SciPy's own ufunc wiring (kept for provenance).

For every public real-valued scipy.special ufunc it records the C++ call SciPy 1.17.1 makes for float64
inputs (and, where SciPy has one, the integer-argument loop), taken from scipy/special/functions.json and
_special_ufuncs.cpp. The argument names come from the ufunc docstring, the output names from its Returns
section. After bootstrapping, spec/special.yaml is maintained by hand.

    python3 tools/parity/bootstrap_special.py <scipy checkout>/scipy/special > spec/special.yaml
    python3 tools/parity/bootstrap_special.py <scipy checkout>/scipy/special --internal > spec/special_internal.yaml
"""
import json, re, sys
import numpy as np
import scipy.special as sc

D = sys.argv[1]
fj = json.load(open(f'{D}/functions.json'))
cpp = open(f'{D}/_special_ufuncs.cpp').read()

xsf_map = {}
for m in re.finditer(r'xsf::numpy::(?:ufunc|gufunc)\(\s*\{(.*?)\}\s*,\s*(?:\d+\s*,\s*)?"([a-z_0-9]+)"', cpp, re.S):
    body, name = m.group(1), m.group(2)
    for t, f in re.findall(r'static_cast<xsf::numpy::([a-z_]+)>\(([^)]*)\)', body):
        ins, outs = t.split('_')
        xsf_map.setdefault(name, []).append((ins, outs, f.strip()))

PORT = {'_legacy.pxd', '_convex_analysis.pxd', '_boxcox.pxd', 'orthogonal_eval.pxd', '_hyp0f1.pxd',
        '_hypergeometric.pxd', '_agm.pxd', '_cdflib_wrappers.pxd', '_ndtri_exp.pxd', '_factorial.pxd'}
RENAME = {'_hyp0f1_real': 'hyp0f1', '_factorial': 'factorial'}


def callee(fn, src):
    fn = fn.replace('[double]', '')
    if src == 'boost_special_functions.h++':
        return fn
    if src == 'xsf_wrappers.h':
        return 'scipy_xsfw::' + fn
    if src in PORT:
        return 'scipy_port::' + RENAME.get(fn, fn)
    if src == 'ellint_carlson_wrap.hh++':
        return 'tsr_ellint::' + fn
    if src == '_cosine.h':
        return fn
    if src == '_wright.h++':
        return 'wright::wrightomega_real'
    if src == 'stirling2.h++':
        return fn
    return None


def call_expr(f, ins, outs, argn, outn, ptr_out=False):
    a = ', '.join(argn)
    if len(outs) == 1:
        return f'o[0] = {f}({a});'
    refs = ', '.join((f'&o[{k}]' if ptr_out else f'o[{k}]') for k in range(len(outs)))
    return f'{f}({a}, {refs});'


def doc_paras(u):
    import textwrap
    return [textwrap.dedent(p).strip() for p in re.split(r'\n\s*\n', u.__doc__ or '') if p.strip()]


def doc_args(u):
    for p in doc_paras(u):
        m = re.match(r'^' + u.__name__ + r'\((.*)\)$', p.split('\n')[0].strip())
        if m and '/' not in m.group(1):
            args = [a.strip() for a in m.group(1).split(',')]
            return [a for a in args if a and not a.startswith(('out', '*'))]
    return []


def doc_outs(u, nout):
    doc = u.__doc__ or ''
    sec = re.split(r'\n\s*Returns\s*\n\s*-+\s*\n', doc)
    names = []
    if len(sec) > 1:
        body = sec[1]
        indent = None
        for line in body.split('\n'):
            if re.match(r'^\s*(See Also|Notes|References|Examples|Raises|Warns)\s*$', line):
                break
            m = re.match(r'^(\s*)([A-Za-z_][\w]*(?:\s*,\s*[A-Za-z_][\w]*)*)\s*:\s', line)
            if m:
                ind = len(m.group(1))
                if indent is None:
                    indent = ind
                if ind == indent:
                    names += [n.strip() for n in m.group(2).split(',')]
    if len(names) != nout:
        names = ['y'] if nout == 1 else [f'y{k}' for k in range(nout)]
    return names


def summary(u):
    paras = doc_paras(u)
    for k, p in enumerate(paras):
        if re.match(r'^' + u.__name__ + r'\(', p):
            continue
        return ' '.join(p.split()).rstrip('.') + '.'
    return ''


INTERNAL = len(sys.argv) > 2 and sys.argv[2] == '--internal'
if INTERNAL:
    import scipy.special._ufuncs as scu
    NAMES = sorted(n for n in dir(scu) if n.startswith('_') and not n.startswith('__') and isinstance(getattr(scu, n), np.ufunc))
    get = lambda n: getattr(scu, n)
else:
    NAMES = sorted(n for n in dir(sc) if isinstance(getattr(sc, n), np.ufunc) and not n.startswith('_'))
    get = lambda n: getattr(sc, n)

entries, missing = [], []
for name in NAMES:
    u = get(name)
    real = [t for t in u.types if all(c in 'dlqif?' for c in t.split('->')[0]) and all(c in 'dlqif?' for c in t.split('->')[1])]
    if not real:
        continue
    impl = int_impl = None
    int_args = []
    nin, nout = u.nin, u.nout
    argn = doc_args(u)
    if len(argn) != nin:
        argn = [f'x{k}' for k in range(nin)]
    outn = doc_outs(u, nout)
    a_in = [f'i[{k}]' for k in range(nin)]
    if name in xsf_map:
        for ins, outs, f in xsf_map[name]:
            if set(ins) == {'d'} and set(outs) == {'d'}:
                impl = call_expr(f, ins, outs, a_in, outn)
            elif 'l' in ins and set(ins) <= {'l', 'd'} and set(outs) == {'d'}:
                int_args = [k for k, c in enumerate(ins) if c == 'l']
                ai = [f'(long)i[{k}]' if c == 'l' else f'i[{k}]' for k, c in enumerate(ins)]
                int_impl = call_expr(f, ins, outs, ai, outn)
    if name in fj:
        for src, fns in fj[name].items():
            for fn, sig in fns.items():
                for s in (sig if isinstance(sig, list) else [sig]):
                    lhs, rhs = s.split('->')
                    ins = lhs.split('*')[0]
                    outs = lhs.split('*')[1] if '*' in lhs else rhs
                    ptr = '*' in lhs
                    c = callee(fn, src)
                    if c is None or re.search(r'[DFGf]', ins + outs):
                        continue
                    if set(ins) == {'d'}:
                        impl = call_expr(c, ins, outs, a_in, outn, ptr)
                    elif set(ins) <= {'d', 'p', 'l', 'i'}:
                        int_args = [k for k, ch in enumerate(ins) if ch in 'pli']
                        ai = [f'(long)i[{k}]' if ch in 'pli' else f'i[{k}]' for k, ch in enumerate(ins)]
                        int_impl = call_expr(c, ins, outs, ai, outn, ptr)
    if impl is None and int_impl is not None:
        impl, int_impl, int_args_only = int_impl, None, int_args     # only an integer loop: truncate doubles
        entries.append((name, argn, outn, impl, None, int_args_only, summary(u), True))
        continue
    if impl is None:
        missing.append(name)
        continue
    entries.append((name, argn, outn, impl, int_impl, int_args, summary(u), False))

if INTERNAL:
    print('# The private scipy.special ufuncs (scipy.special._ufuncs._*) that scipy.stats calls, for the C++ API of')
    print('# cxx/gen_sc.hpp (not registered as public functions). Bootstrapped by')
    print('# tools/parity/bootstrap_special.py <scipy>/scipy/special --internal from SciPy 1.17.1 wiring.')
    print('module: special_internal')
    print('internal: true')
else:
    print('# scipy.special functions in the kernel registry (ADR 0011).')
    print('# Bootstrapped by tools/parity/bootstrap_special.py from SciPy 1.17.1 wiring; maintained by hand since.')
    print('# impl: C++ statement over inputs i[] and outputs o[] (float64). int_impl: the loop SciPy uses when the')
    print('# arguments listed in int_args are integers (NumPy picks it for integer arrays).')
    print('module: special')
print('functions:')
for name, argn, outn, impl, int_impl, int_args, doc, only_int in entries:
    print(f'  - name: {name}')
    print(f'    ref: scipy.special.{"_ufuncs." if INTERNAL else ""}{name}')
    print(f'    args: [{", ".join(argn)}]')
    print(f'    outs: [{", ".join(outn)}]')
    print(f'    impl: "{impl}"')
    if int_impl:
        print(f'    int_impl: "{int_impl}"')
        print(f'    int_args: {int_args}')
    if only_int:
        print(f'    integer_only: {int_args}')
    print(f'    doc: {json.dumps(doc)}')
sys.stderr.write(f'{len(entries)} functions; unmapped: {missing}\n')
