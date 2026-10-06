#!/usr/bin/env python3
"""Extract the real-valued wrappers of SciPy's scipy/special/xsf_wrappers.cpp into a header-only C++ file.

    python3 tools/parity/extract_xsf_wrappers.py <scipy checkout>/scipy/special/xsf_wrappers.cpp \
        > csrc/third_party/scipy/xsf_wrappers_real.hpp

Functions whose signature or body uses NumPy complex types are skipped; Py_ssize_t becomes long.
"""
import re, sys

src = open(sys.argv[1]).read()
incs = [l for l in src.split('\n') if l.startswith('#include <xsf/')]
out = ['/*', ' * Real-valued wrappers extracted from SciPy 1.17.1 scipy/special/xsf_wrappers.cpp (BSD-3-Clause,',
       ' * see LICENSE.txt) by tools/parity/extract_xsf_wrappers.py. Py_ssize_t parameters became long;',
       ' * complex wrappers are omitted. Do not edit by hand.', ' */', '#pragma once', ''] + incs + ['#include <limits>', '', 'namespace scipy_xsfw {', '']
pos = 0
n = 0
head = re.compile(r'^(double|int|void|float)\s+(\w+)\(([^)]*)\)\s*\{', re.M)
while True:
    m = head.search(src, pos)
    if not m:
        break
    i = m.end()
    depth = 1
    while depth:
        c = src[i]
        depth += c == '{'
        depth -= c == '}'
        i += 1
    body = src[m.end():i - 1]
    pos = i
    ret, name, params = m.group(1), m.group(2), m.group(3)
    if 'npy' in params or 'npy' in body or 'complex' in body or 'to_c' in body or 'complex' in params:
        continue
    params = params.replace('Py_ssize_t', 'long')
    out.append(f'inline {ret} {name}({params}) {{{body}}}')
    n += 1
out += ['', '} // namespace scipy_xsfw', '']
sys.stdout.write('\n'.join(out))
sys.stderr.write(f'{n} wrappers\n')
