#!/usr/bin/env bash
# Memory-safety sweep of every registry function under AddressSanitizer + UBSan (the phase gate of the parity
# program; first run 2026-09-28: 539 functions, ~370k calls, 10 root causes fixed).
#
#   bash tools/parity/asan/run.sh                 # build the sanitizer library, run all jobs, print the report
#   JOBS="stats np" bash tools/parity/asan/run.sh # only some kinds (special np stats dist gen)
#
# Environment: BUILD (default /tmp/tessero-asan), PAR (parallel jobs, default 2), HANG (seconds, default 20).
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT=$PWD
BUILD=${BUILD:-/tmp/tessero-asan}
SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1"
mkdir -p "$BUILD/lib"
make -C csrc BUILD="$BUILD" OUT="$BUILD/lib" CC="gcc $SAN" CXX="g++ $SAN" -j"${PAR:-2}"
nm -D "$BUILD/lib/libtessero.so" > "$BUILD/nm.txt"          # not piped into grep -q: SIGPIPE would fail the pipeline
grep -q __asan_report "$BUILD/nm.txt" || { echo "not an ASan build" >&2; exit 1; }
python3 tools/parity/asan/make_jobs.py > "$BUILD/jobs.txt"
if [ -n "${JOBS:-}" ]; then
    grep -E "^($(echo "$JOBS" | tr ' ' '|')) " "$BUILD/jobs.txt" > "$BUILD/jobs.sel" || true
    mv "$BUILD/jobs.sel" "$BUILD/jobs.txt"
fi
rm -f "$BUILD/results.jsonl"
TESSERO_LIB="$BUILD/lib/libtessero.so" LOGDIR="$BUILD/logs" \
    python3 tools/parity/asan/driver.py "$BUILD/jobs.txt" "$BUILD/results.jsonl" "${PAR:-2}"
python3 tools/parity/asan/analyze.py "$BUILD/results.jsonl"
