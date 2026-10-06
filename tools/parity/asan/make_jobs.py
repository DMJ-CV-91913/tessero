#!/usr/bin/env python3
"""The fuzz job list from the kernel registry (tools/parity/registry.json): one "<kind> <method>" line per
function, where kind selects the argument generator in harness.php (special, np, stats, dist, gen) and method is
the PHP façade name (camelCase, as tools/parity/gen-facades.php makes it)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
reg = json.load(open(os.path.join(HERE, '..', 'registry.json')))['functions']


def camel(s):
    parts = s.split('_')
    return parts[0] + ''.join('_' if p == '' else p[:1].upper() + p[1:] for p in parts[1:])


for name, info in sorted(reg.items()):
    mod, fn = name.split('.', 1)
    if mod == 'stats':
        kind = 'dist' if info.get('kind') == 'dist' else 'stats'
    elif mod == 'random':
        kind = 'gen'
    else:
        kind = mod
    print(kind, camel(fn))
# Generator methods older than the registry
for m in ('integers', 'choice', 'permutation'):
    print('gen', m)
