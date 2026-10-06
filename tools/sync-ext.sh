#!/usr/bin/env bash
# Copy libtessero's sources into ext/libtessero so the Zend extension builds
# from a self-contained directory (phpize, PIE and pecl packages cannot reach
# ../csrc), and write the kernel source lists into ext/config.m4 and
# ext/config.w32. CI runs this with --check and fails on drift.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/ext/libtessero"

stage() {                       # $1 = target directory
    mkdir -p "$1/src" "$1/include" "$1/cxx" "$1/third_party"
    cp "$ROOT"/csrc/src/*.c "$ROOT"/csrc/src/*.h "$1/src/"
    cp "$ROOT"/csrc/include/tessero.h "$1/include/"
    cp "$ROOT"/csrc/cxx/*.cpp "$ROOT"/csrc/cxx/*.hpp "$ROOT"/csrc/cxx/*.inc "$1/cxx/"
    cp -r "$ROOT"/csrc/third_party/. "$1/third_party/"
}

rewrite() {                     # $1 = config.m4, $2 = config.w32: write the source lists between the markers
    local c cxx
    c="$(cd "$ROOT/csrc/src" && ls *.c | sed 's|^|libtessero/src/|' | tr '\n' ' ')libtessero/third_party/scipy/cdflib.c libtessero/third_party/scipy/_cosine.c libtessero/third_party/scipy/quadpack/quadpack.c libtessero/third_party/scipy/zeros/brentq.c $(cd "$ROOT/csrc/third_party/numpy/random/src" && ls *.c | sed 's|^|libtessero/third_party/numpy/random/src/|' | tr '\n' ' ')"
    cxx="$(cd "$ROOT/csrc/cxx" && ls *.cpp | sed 's|^|libtessero/cxx/|' | tr '\n' ' ')libtessero/third_party/scipy/wright.cc"
    python3 - "$1" "$2" "$c" "$cxx" <<'PY'
import re, sys
m4, w32, c, cxx = sys.argv[1:5]
s = open(m4).read()
s = re.sub(r'(dnl BEGIN sources \(tools/sync-ext\.sh\)\n).*?(\n\s*dnl END sources)',
           lambda m: m.group(1) + f'  TESSERO_KERNEL="{c.strip()}"\n  TESSERO_KERNEL_CXX="{cxx.strip()}"' + m.group(2), s, flags=re.S)
open(m4, 'w').write(s)
s = open(w32).read()
csrc = ' '.join(x.split('/')[-1] for x in c.split() if x.startswith('libtessero/src/'))
cxxs = ' '.join(x.split('/')[-1] for x in cxx.split() if x.startswith('libtessero/cxx/'))
s = re.sub(r'(// BEGIN sources \(tools/sync-ext\.sh\)\n).*?(\n\s*// END sources)',
           lambda m: m.group(1) + f'    var TESSERO_KERNEL = "{csrc}";\n    var TESSERO_KERNEL_CXX = "{cxxs}";' + m.group(2), s, flags=re.S)
open(w32, 'w').write(s)
PY
}

if [ "${1:-}" = "--check" ]; then
    TMP="$(mktemp -d)"
    stage "$TMP"
    cp "$ROOT/ext/config.m4" "$TMP/config.m4"
    cp "$ROOT/ext/config.w32" "$TMP/config.w32"
    rewrite "$TMP/config.m4" "$TMP/config.w32"
    status=0
    for d in src include cxx third_party; do
        diff -r "$TMP/$d" "$DST/$d" >/dev/null 2>&1 || status=1
    done
    cmp -s "$TMP/config.m4" "$ROOT/ext/config.m4" || status=1
    cmp -s "$TMP/config.w32" "$ROOT/ext/config.w32" || status=1
    rm -rf "$TMP"
    if [ $status -eq 0 ]; then echo "ext/libtessero is in sync"; exit 0; fi
    echo "ext/libtessero is out of date: run tools/sync-ext.sh" >&2
    exit 1
fi
rm -rf "$DST"
stage "$DST"
rewrite "$ROOT/ext/config.m4" "$ROOT/ext/config.w32"
echo "synced $(ls "$DST/src" | wc -l) kernel files and $(ls "$DST/cxx" | wc -l) C++ files into ext/libtessero"
