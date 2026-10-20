#!/usr/bin/env bash
# run_selfcheck.sh -- build libtessero and run the canonical self-check against the
# REAL dispatched kernel. This is the integration check: it proves the built library's
# gemm is canonical (fast clone == explicit-fma reference), and on the first run prints
# the frozen KAT digest to bake into TSR_CANON_GEMM_F32_KAT (selfcheck.c).
#
# Usage, from csrc/repro:  bash run_selfcheck.sh        (default cc)
#                     or:  CC=gcc-12 bash run_selfcheck.sh
# Needs the tessero build deps (liblapacke-dev libopenblas-dev build-essential).
# No em dashes. Per-process temp (multi-user build server).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
CSRC="$(cd "$HERE/.." && pwd)"
CC="${CC:-cc}"
TMP="/tmp/tsr_selfcheck.$$"
echo "compiler: $($CC --version 2>/dev/null | head -1)"

echo "building libtessero (CC=$CC) ..."
if ! make -C "$CSRC" CC="$CC" -j"$(nproc)" >"$TMP.build" 2>&1; then
  echo "BUILD FAILED (last lines):"; tail -20 "$TMP.build"; rm -f "$TMP.build"; exit 1
fi
LIB="$(find "$CSRC/../lib" -name 'libtessero.so' 2>/dev/null | head -1)"
if [ -z "$LIB" ]; then echo "libtessero.so not found under $CSRC/../lib"; rm -f "$TMP.build"; exit 1; fi
LIBDIR="$(dirname "$LIB")"
echo "lib: $LIB"

echo "building + running the self-check driver ..."
if ! $CC -O2 "$HERE/selfcheck_run.c" -o "$TMP.run" -L"$LIBDIR" -ltessero -lm 2>"$TMP.cc"; then
  echo "DRIVER COMPILE FAILED:"; cat "$TMP.cc"; rm -f "$TMP".*; exit 1
fi
LD_LIBRARY_PATH="$LIBDIR" "$TMP.run"; rc=$?
rm -f "$TMP".*
echo
case $rc in
  0) echo "VERDICT: CANONICAL. Bake the printed digest into TSR_CANON_GEMM_F32_KAT to arm the hard gate." ;;
  *) echo "VERDICT: DIVERGED (rc=$rc) -- this build is NOT canonical; do not ship it." ;;
esac
exit $rc
