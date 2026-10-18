#!/usr/bin/env bash
# csrc/repro/run_determinism.sh
#
# Proof of reproducibility for the f32 primitives in csrc/src/repro_f32.c.
# Builds that real ABI source + determinism_test.c several ways, all with
# -fno-fast-math -ffp-contract=off (never fast-math: it would swap in the
# non-reproducible approximations), and checks that the raw f32 output is
# byte-identical across:
#   - SIMD level: scalar (no vectorization) vs AVX2+FMA vs AVX-512 (if present)
#   - thread count: 1 vs 4 OpenMP threads
# Equal bytes == bit-identical == reproducible, and that SHA-256 is the golden.
# Then prints the edge-input classification, the accuracy-vs-double report, and
# the canonical spec. Needs a C compiler (OpenMP) + sha256sum.
#
# Run: bash csrc/repro/run_determinism.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"               # csrc/
IMPL="$ROOT/src/repro_f32.c"
TEST="$HERE/determinism_test.c"
CC="${CC:-cc}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

COMMON="-fno-fast-math -ffp-contract=off -fopenmp"   # -lm goes AFTER the sources

echo "== compiling (all with -fno-fast-math -ffp-contract=off) =="
echo "-- scalar (no vectorization) --"
$CC -O2 -std=gnu11 -fno-tree-vectorize $COMMON "$IMPL" "$TEST" -o "$TMP/dt_scalar" -lm
echo "-- AVX2 + FMA --"
$CC -O3 -std=gnu11 -mavx2 -mfma        $COMMON "$IMPL" "$TEST" -o "$TMP/dt_avx2" -lm

HAVE_AVX512=0
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
  echo "-- AVX-512 (x86-64-v4) --"
  $CC -O3 -std=gnu11 -march=x86-64-v4  $COMMON "$IMPL" "$TEST" -o "$TMP/dt_avx512" -lm
  HAVE_AVX512=1
else
  echo "-- AVX-512: not available on this host, skipping --"
fi

h() { sha256sum | cut -d' ' -f1; }          # hash stdin

echo
echo "================= determinism across SIMD level and thread count ================="
FAIL=0
declare -A GOLDEN
for fn in rsqrt exp2 exp erf; do
  HS=$(OMP_NUM_THREADS=1 "$TMP/dt_scalar" "$fn" | h)
  HA1=$(OMP_NUM_THREADS=1 "$TMP/dt_avx2"  "$fn" | h)
  HA4=$(OMP_NUM_THREADS=4 "$TMP/dt_avx2"  "$fn" | h)
  LINE="$fn: scalar=$HS avx2/1t=$HA1 avx2/4t=$HA4"
  OK=1
  [ "$HS" = "$HA1" ] || OK=0
  [ "$HS" = "$HA4" ] || OK=0
  if [ "$HAVE_AVX512" = 1 ]; then
    HV=$(OMP_NUM_THREADS=1 "$TMP/dt_avx512" "$fn" | h)
    LINE="$LINE avx512/1t=$HV"
    [ "$HS" = "$HV" ] || OK=0
  fi
  if [ "$OK" = 1 ]; then
    GOLDEN[$fn]="$HS"
    echo "  [OK]   $fn  GOLDEN=$HS"
  else
    FAIL=1
    echo "  [DIFF] $LINE"
  fi
done

echo
echo "================= edge inputs (classified, not hashed) ================="
for fn in rsqrt exp2 exp erf; do OMP_NUM_THREADS=1 "$TMP/dt_avx2" "$fn" edges; done

echo
echo "================= accuracy vs double reference ================="
for fn in rsqrt exp2 exp erf; do OMP_NUM_THREADS=1 "$TMP/dt_avx2" "$fn" accuracy; done

echo
echo "================= canonical spec (exact hex floats) ================="
"$TMP/dt_avx2" spec

echo
if [ "$FAIL" = 0 ]; then
  echo "RESULT: ALL BIT-IDENTICAL across SIMD level and thread count. Goldens:"
  for fn in rsqrt exp2 exp erf; do echo "  GOLDEN($fn,f32,N=2^20) = ${GOLDEN[$fn]}"; done
else
  echo "RESULT: a primitive DIFFERED across builds. Investigate (fast-math / contraction leak)."
  exit 1
fi
