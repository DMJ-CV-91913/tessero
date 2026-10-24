#!/usr/bin/env python3
"""CycloneDX 1.5 SBOM for a Tessero release.

    python3 tools/sbom.py --version 0.2.0 [--output tessero.cdx.json]

Scanners cannot see the C/C++ code vendored under csrc/third_party (no package manifest), so the components come
from the tables in THIRD_PARTY_NOTICES.md: the vendored table (bundled, scope "required") and the table of
libraries linked at build or run time (dynamic dependencies, scope "optional" because the binary tarball does not
ship them; the statically linked GCC runtime is "required"). The root component is tessero/tessero from
composer.json, with tessero/laravel from laravel/composer.json as a sibling package.

Standard library only; the output is deterministic for a given commit (SOURCE_DATE_EPOCH sets the timestamp, the
serial number is derived from the version and the content).
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_URL = 'https://github.com/DMJ-CV-91913/tessero'

# known upstream coordinates for the vendored components (matched against the "Component" cell, in order)
KNOWN = [
    (r'^SciPy|FITPACK', 'scipy', 'pkg:github/scipy/scipy@v{v}', 'https://github.com/scipy/scipy'),
    (r'^xsf', 'xsf', 'pkg:github/scipy/xsf@{v}', 'https://github.com/scipy/xsf'),
    (r'^Boost', 'boost-math', 'pkg:github/boostorg/math@{v}', 'https://github.com/boostorg/math'),
    (r'^NumPy|^MT19937', 'numpy', 'pkg:github/numpy/numpy@v{v}', 'https://github.com/numpy/numpy'),
    (r'f2c', 'f2c', None, 'https://www.netlib.org/f2c/'),
    (r'Joe & Kuo', 'sobol-direction-numbers', None, 'https://web.maths.unsw.edu.au/~fkuo/sobol/'),
]

SPDX = {
    'BSD-3-Clause': 'BSD-3-Clause',
    'Boost Software License 1.0': 'BSL-1.0',
    'MIT': 'MIT',
    'GPL-3.0-or-later WITH GCC-exception-3.1': 'GPL-3.0-or-later WITH GCC-exception-3.1',
}


def strip_md(text):
    return re.sub(r'[`*]', '', text).strip()


def tables(path):
    """Every markdown table in the file as (heading, header cells, rows)."""
    out, heading, cur = [], '', None
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                heading = line.lstrip('#').strip()
            if line.startswith('|'):
                cells = [c.strip() for c in line.strip().strip('|').split('|')]
                if cur is None:
                    cur = (heading, cells, [])
                elif all(re.fullmatch(r':?-+:?', c) for c in cells):
                    continue
                else:
                    cur[2].append(cells)
            elif cur is not None:
                out.append(cur)
                cur = None
    if cur is not None:
        out.append(cur)
    return out


def licence_of(cell):
    text = strip_md(cell)
    if text.startswith('GPL-3.0-or-later WITH GCC-exception-3.1'):
        return [{'expression': SPDX['GPL-3.0-or-later WITH GCC-exception-3.1']}]
    found = [spdx for name, spdx in SPDX.items() if name in text and 'GCC' not in name]
    if found:
        if len(found) == 1:
            return [{'license': {'id': found[0]}}]
        return [{'expression': ' AND '.join(dict.fromkeys(found))}]
    # a non-SPDX licence (the f2c notice, the Joe & Kuo licence): name it, point at the text
    return [{'license': {'name': text.split('(')[0].strip().rstrip(',')}}]


def version_of(cell):
    text = strip_md(cell)
    m = re.search(r'(?:SciPy|NumPy)\s+([0-9][0-9.]*)', text) or re.search(r'commit\s+([0-9a-f]{7,40})', text)
    return m.group(1) if m else text


def bom_ref(name, version):
    return 'component:' + re.sub(r'[^A-Za-z0-9.+-]+', '-', f'{name}@{version}').strip('-').lower()


def vendored_component(cells):
    name, version_cell, licence_cell, location, used_for = (cells + [''] * 5)[:5]
    version = version_of(version_cell)
    comp = {
        'type': 'library',
        'name': strip_md(name),
        'version': version,
        'scope': 'required',
        'licenses': licence_of(licence_cell),
        'properties': [
            {'name': 'tessero:location', 'value': strip_md(location)},
            {'name': 'tessero:used-for', 'value': strip_md(used_for)},
            {'name': 'tessero:inclusion', 'value': 'vendored source'},
        ],
    }
    for pattern, short, purl, url in KNOWN:
        if re.search(pattern, strip_md(name)):
            comp['group'] = short
            if purl and re.fullmatch(r'[0-9a-f]{7,40}|[0-9][0-9.]*', version):
                comp['purl'] = purl.format(v=version)
            comp['externalReferences'] = [{'type': 'website', 'url': url}]
            break
    comp['bom-ref'] = bom_ref(comp['name'], version)
    return comp


def runtime_component(cells):
    name, licence_cell, how = (cells + [''] * 3)[:3]
    static = 'statically' in how
    comp = {
        'type': 'library',
        'name': strip_md(name),
        'scope': 'required' if static else 'optional',
        'licenses': licence_of(licence_cell),
        'properties': [
            {'name': 'tessero:inclusion', 'value': 'statically linked' if static else 'dynamic, provided by the system'},
            {'name': 'tessero:how', 'value': strip_md(how)},
        ],
    }
    comp['bom-ref'] = bom_ref(comp['name'], 'system')
    return comp


def composer_component(path, version, ctype):
    with open(path, encoding='utf-8') as f:
        meta = json.load(f)
    group, name = meta['name'].split('/', 1)
    return {
        'type': ctype,
        'bom-ref': f'pkg:composer/{meta["name"]}@{version}',
        'group': group,
        'name': name,
        'version': version,
        'description': meta.get('description', ''),
        'licenses': [{'license': {'id': meta.get('license', 'BSD-3-Clause')}}],
        'purl': f'pkg:composer/{meta["name"]}@{version}',
        'externalReferences': [{'type': 'vcs', 'url': REPO_URL}],
    }


def build(version):
    notices = os.path.join(ROOT, 'THIRD_PARTY_NOTICES.md')
    vendored, runtime = [], []
    for heading, header, rows in tables(notices):
        cols = [strip_md(h).lower() for h in header]
        if cols[:2] == ['component', 'version']:
            vendored += [vendored_component(r) for r in rows]
        elif cols[:1] == ['library']:
            runtime += [runtime_component(r) for r in rows]
    if not vendored:
        raise SystemExit('no vendored-component table found in THIRD_PARTY_NOTICES.md')

    root = composer_component(os.path.join(ROOT, 'composer.json'), version, 'library')
    laravel = composer_component(os.path.join(ROOT, 'laravel', 'composer.json'), version, 'library')
    components = [laravel] + vendored + runtime
    epoch = int(os.environ.get('SOURCE_DATE_EPOCH', '0') or 0)
    stamp = datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc) if epoch else datetime.datetime.now(datetime.timezone.utc)
    body = json.dumps([root, components], sort_keys=True).encode()
    serial = uuid.UUID(hashlib.sha256(version.encode() + body).hexdigest()[:32], version=4)
    return {
        'bomFormat': 'CycloneDX',
        'specVersion': '1.5',
        'serialNumber': f'urn:uuid:{serial}',
        'version': 1,
        'metadata': {
            'timestamp': stamp.replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
            'tools': {'components': [{'type': 'application', 'name': 'tools/sbom.py', 'version': version}]},
            'component': root,
            'properties': [{'name': 'tessero:artifact', 'value': 'tessero-binaries.tgz'}],
        },
        'components': components,
        'dependencies': [
            {'ref': root['bom-ref'], 'dependsOn': [c['bom-ref'] for c in vendored + runtime]},
            {'ref': laravel['bom-ref'], 'dependsOn': [root['bom-ref']]},
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--version', required=True, help='release version without the leading v')
    ap.add_argument('--output', default='-', help='file to write (default stdout)')
    a = ap.parse_args()
    if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?', a.version):
        raise SystemExit(f'not a semantic version: {a.version}')
    doc = json.dumps(build(a.version), indent=2) + '\n'
    if a.output == '-':
        sys.stdout.write(doc)
    else:
        with open(a.output, 'w', encoding='utf-8') as f:
            f.write(doc)
        print(f'wrote {a.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
