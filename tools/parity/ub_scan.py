#!/usr/bin/env python3
"""Find fixture rows on which SciPy's own code has undefined behaviour (out-of-bounds memory access).

    python3 tools/parity/ub_scan.py special obl_ang1 [...]      # prints the flagged rows per function

Some routines SciPy calls overrun fixed-size work arrays for part of their domain (the spheroidal wave
functions of xsf/specfun for large c). SciPy then returns whatever the overrun reads, or corrupts its heap;
the result is not a reference value. Tessero sizes those arrays for the loop bounds instead (a local
modification of csrc/third_party/xsf), so its value exists but cannot be checked against SciPy.

This tool builds the spec'd functions from the vendored sources with the original array sizes
(-DTSR_XSF_ORIGINAL_BUFFERS, i.e. exactly SciPy's code) under AddressSanitizer and evaluates every fixture
row in a forked child; a row is flagged when AddressSanitizer reports an invalid access. fixtures.py records
flagged rows as "undef" (not compared) for spec entries marked `reference_ub: asan`, and report.py counts them.
"""
import os, subprocess, sys, tempfile

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures  # noqa: E402

ROOT = fixtures.ROOT
CSRC = os.path.join(ROOT, 'csrc')

HARNESS = r'''
#include "special_common.hpp"
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
namespace tsd { void fail(const char *) {} }   /* Boost's overflow hook (cxx/stats_dist.cpp); the harness only watches for crashes */
static void eval(int f, const double *i, double *o) {
    switch (f) {
%CASES%
    }
}
int main() {
    int f, nin;
    double v[8], o[8];
    setvbuf(stdout, NULL, _IOLBF, 0);
    while (scanf("%d %d", &f, &nin) == 2) {
        for (int k = 0; k < nin; k++) scanf("%lf", &v[k]);
        pid_t pid = fork();
        if (pid == 0) { alarm(60); eval(f, v, o); _exit(0); }
        int st = 0;
        waitpid(pid, &st, 0);
        printf("%d\n", (WIFEXITED(st) && WEXITSTATUS(st) == 0) ? 0 : 1);
    }
    return 0;
}
'''


def build(entries, workdir):
    cases = '\n'.join(f'    case {k}: {{ {e["impl"]} }} break;' for k, e in enumerate(entries))
    src = os.path.join(workdir, 'ub_harness.cpp')
    open(src, 'w').write(HARNESS.replace('%CASES%', cases))
    exe = os.path.join(workdir, 'ub_harness')
    inc = ['-I' + os.path.join(CSRC, p) for p in ('cxx', 'src', 'third_party', 'third_party/xsf', 'third_party/scipy',
                                                   'third_party/boost')]
    cdflib = os.path.join(CSRC, 'third_party', 'scipy', 'cdflib.c')
    obj = os.path.join(workdir, 'cdflib.o')
    subprocess.run(['gcc', '-O1', '-c', cdflib, '-o', obj, '-w'], check=True)
    subprocess.run(['g++', '-std=c++17', '-O1', '-g', '-fsanitize=address', '-DTSR_XSF_ORIGINAL_BUFFERS',
                    '-DBOOST_MATH_STANDALONE', '-w', *inc, src, obj, '-o', exe], check=True)
    return exe


def scan(module, names):
    spec = yaml.safe_load(open(os.path.join(ROOT, 'spec', f'{module}.yaml')))
    entries = [e for e in spec['functions'] if e['name'] in names]
    rng = np.random.default_rng(20260928)
    rows_by = {}
    for fn in spec['functions']:                      # the same rng sequence as fixtures.py
        if 'impl' not in fn:
            continue
        rows = fixtures.cases_for(fn, rng, int(fn.get('cases', 1500)))
        if fn['name'] in names:
            rows_by[fn['name']] = rows
    out = {}
    with tempfile.TemporaryDirectory() as d:
        exe = build(entries, d)
        for k, e in enumerate(entries):
            rows = rows_by[e['name']]
            data = ''.join(f'{k} {len(r)} ' + ' '.join(repr(float(v)) for v in r) + '\n' for r in rows)
            env = dict(os.environ, ASAN_OPTIONS='exitcode=99:detect_leaks=0:abort_on_error=0')
            res = subprocess.run([exe], input=data, capture_output=True, text=True, env=env, timeout=36000)
            flags = [int(x) for x in res.stdout.split()]
            out[e['name']] = [rows[i] for i, f in enumerate(flags) if f]
            print(f"{e['name']}: {len(out[e['name']])} of {len(rows)} rows are undefined in the reference", file=sys.stderr)
    return out


if __name__ == '__main__':
    import json
    res = scan(sys.argv[1], set(sys.argv[2:]))
    json.dump({k: [[float(x) for x in r] for r in v] for k, v in res.items()}, sys.stdout)
