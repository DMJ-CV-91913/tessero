#!/usr/bin/env bash
# The full gate of the parity program (prompts/numpy-scipy-parity-program.md section 8), adapted to the function
# registry (ADR 0011): the code generators are gen_registry.py and gen-facades.php.
#
#   bash tools/parity/gate.sh                    # everything
#   QUICK=1 bash tools/parity/gate.sh            # without sanitizers, fuzzing, soak and benchmarks
#
# Environment: BUILD (kernel build dir, default csrc/build), EXT_DIR (a writable copy of ext/ to build in,
# default ext), PHPUNIT (default vendor/bin/phpunit), PYTHON (default python3 with numpy/scipy pinned by
# tools/parity/requirements.txt).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
BUILD=${BUILD:-$ROOT/csrc/build}
EXT_DIR=${EXT_DIR:-$ROOT/ext}
PHPUNIT=${PHPUNIT:-vendor/bin/phpunit}
PY=${PYTHON:-python3}
step() { printf '\n== %s\n' "$*"; }

step "generated files are current"
bash tools/sync-ext.sh --check
$PY tools/parity/gen_registry.py --check
$PY tools/parity/census.py --check
php tools/beta-issues.php check
php tools/gen-example-docs.php --check
php tools/gen-translation-table.php --check
php tools/gen-examples-manifest.php --check

step "kernel"
make -C csrc BUILD="$BUILD" -j"$(nproc)"
make -C csrc BUILD="$BUILD" test
[ -n "${QUICK:-}" ] || make -C csrc BUILD="$BUILD" sanitize
php -d ffi.enable=1 tools/parity/gen-facades.php --check

step "extension"
if [ "$EXT_DIR" != "$ROOT/ext" ]; then rsync -a --exclude modules "$ROOT/ext/" "$EXT_DIR/"; fi
(cd "$EXT_DIR" && { [ -f Makefile ] || { phpize >/dev/null 2>&1 && sed -i 's/^AC_PROG_LIBTOOL$/LT_INIT/' configure.ac && autoconf -f >/dev/null 2>&1 && ./configure --enable-tessero; }; } && make -j"$(nproc)" \
    && NO_INTERACTION=1 REPORT_EXIT_STATUS=1 make test TESTS=tests)
EXT_SO=$EXT_DIR/modules/tessero.so

step "NumPy/SciPy fixtures on both backends"
php -d ffi.enable=1 -d memory_limit=8G tools/parity/run-tests.php --backend=ffi
php -d extension="$EXT_SO" -d memory_limit=8G tools/parity/run-tests.php --backend=ext
# the vendored special functions again with every heap allocation pre-filled with a garbage pattern: a result that
# depended on uninitialised memory would change (reference-deviations.md, "Undefined rows")
MALLOC_PERTURB_=165 php -d ffi.enable=1 -d memory_limit=8G tools/parity/run-tests.php --backend=ffi --module=special --no-record

step "PHPUnit suites"
php -d ffi.enable=1 "$PHPUNIT"
php -d ffi.enable=1 -d extension="$EXT_SO" "$PHPUNIT" --testsuite ext,unit
php -d ffi.enable=1 -d opcache.enable_cli=1 -d opcache.jit=tracing "$PHPUNIT" --testsuite unit
php -d ffi.enable=1 "$PHPUNIT" --testsuite laravel
php -d ffi.enable=1 -d extension="$EXT_SO" tests/docs/examples.php
bash tests/e2e/preload.sh

step "runnable examples cookbook (both backends)"
php tools/run-examples.php --ext-so="$EXT_SO"

if [ -z "${QUICK:-}" ]; then
    step "usage corpus is current (clones pinned tags from GitHub)"
    $PY tools/parity/usage.py --check

    step "fuzzing, soak, benchmarks"
    make -C csrc BUILD="$BUILD" fuzz-standalone FUZZ_RUNS=200000
    BUILD="$BUILD/asan" bash tools/parity/asan/run.sh | tee "$BUILD/asan-report.txt"   # review: no SAN/CRASH lines
    php -d ffi.enable=1 tools/soak.php --backend=ffi --duration=600
    php -d extension="$EXT_SO" tools/soak.php --backend=ext --duration=600
    php -d ffi.enable=1 -d memory_limit=2G bench/run.php
fi

step "metrics and claims"
$PY tools/parity/report.py --check
php tools/parity/claims-check.php
mkdocs build --strict -d "$BUILD/site"
echo "gate: ok"
