#!/usr/bin/env bash
# Update a Tessero dev/reference checkout: pull origin/main, rebuild the kernel and the
# native extension, then sanity-check (C tests + the FFI backend against the committed fixtures).
#
# Meant for a checkout that only tracks origin/main (a read-only remote is fine - this pulls, never pushes).
#
#   bash tools/pull-build.sh             # pull, build kernel + extension, C tests + FFI parity
#   bash tools/pull-build.sh --quick     # skip the ~4 min FFI parity run (C tests only)
#   bash tools/pull-build.sh --no-ext    # skip the extension build (FFI kernel only)
#   SKIP_PULL=1 bash tools/pull-build.sh # rebuild/verify without pulling
#
# Env: EXT_BUILD (scratch dir for the extension build, default ~/ext-build). The extension is
# built in a copy, never in-place, so it can't pollute the tracked ext/ tree.
set -euo pipefail
cd "$(dirname "$0")/.."

EXT_BUILD=${EXT_BUILD:-$HOME/ext-build}
QUICK=0
NO_EXT=0
for a in "$@"; do
    case $a in
        --quick)  QUICK=1 ;;
        --no-ext) NO_EXT=1 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $a (try --help)" >&2; exit 2 ;;
    esac
done
step() { printf '\n== %s\n' "$*"; }

if [ -z "${SKIP_PULL:-}" ]; then
    step "pull (fast-forward only)"
    before=$(git rev-parse HEAD)
    git pull --ff-only
    after=$(git rev-parse HEAD)
    [ "$before" = "$after" ] && echo "already up to date ($after)" || echo "$before -> $after"
fi

step "kernel (libtessero.so)"
make -C csrc -j"$(nproc)"

if [ "$NO_EXT" -eq 0 ]; then
    step "native extension (scratch build in $EXT_BUILD)"
    rm -rf "$EXT_BUILD"
    cp -r ext "$EXT_BUILD"
    # PHP's -j build intermittently races on the per-source .dep files for the libtessero/ subtree
    # ("*.dep: No such file or directory"); fall back to a serial make, which creates those build
    # dirs in order. A genuine compile error still fails the serial pass (and set -e aborts).
    # PHP's phpize.m4 emits the obsolete AC_PROG_LIBTOOL macro into configure.ac; rewrite it to LT_INIT
    # (what autoupdate would do) and regenerate configure, so the build is warning-free without needing a
    # writable system phpize.m4.
    ( cd "$EXT_BUILD" && phpize >/dev/null 2>&1 \
        && sed -i 's/^AC_PROG_LIBTOOL$/LT_INIT/' configure.ac && autoconf -f >/dev/null 2>&1 \
        && ./configure --enable-tessero >/dev/null \
        && { make -j"$(nproc)" >/dev/null 2>&1 || make; } )
    echo "built $EXT_BUILD/modules/tessero.so"
fi

step "C tests"
make -C csrc test

if [ "$QUICK" -eq 0 ]; then
    step "FFI parity (against the committed fixtures; no record)"
    echo "   (checks ~223k values; runs a few minutes with NO output - this is not a hang. Pass --quick to skip.)"
    php -d ffi.enable=1 -d memory_limit=8G tools/parity/run-tests.php --backend=ffi --no-record
fi

step "done"
echo "HEAD $(git rev-parse --short HEAD) ready. For the full gate: QUICK=1 EXT_DIR=$EXT_BUILD bash tools/parity/gate.sh"
