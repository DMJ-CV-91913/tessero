#!/usr/bin/env bash
# csrc/repro/run_demo.sh
# Compile demo_rsqrt.c two ways (scalar vs AVX2+FMA), hash each output, compare.
# Equal hashes == rsqrt is bit-identical across SIMD == reproducible. Prints the golden.
# Run from anywhere: bash csrc/repro/run_demo.sh  (needs a C compiler + sha256sum).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/demo_rsqrt.c"
CC="${CC:-cc}"

# IMPORTANT: no -ffast-math anywhere (it would swap in the non-reproducible approx rsqrt).
echo "== compiling scalar (no vectorization) =="
$CC -O2 -std=c11 -fno-fast-math -ffp-contract=off -fno-tree-vectorize "$SRC" -o /tmp/repro_rsqrt_scalar -lm
echo "== compiling AVX2 + FMA (vectorized) =="
$CC -O3 -std=c11 -fno-fast-math -ffp-contract=off -mavx2 -mfma "$SRC" -o /tmp/repro_rsqrt_avx2 -lm

H_SCALAR="$(/tmp/repro_rsqrt_scalar | sha256sum | cut -d' ' -f1)"
H_AVX2="$(/tmp/repro_rsqrt_avx2   | sha256sum | cut -d' ' -f1)"

echo
echo "scalar sha256: $H_SCALAR"
echo "avx2   sha256: $H_AVX2"
echo
if [ "$H_SCALAR" = "$H_AVX2" ]; then
  echo "RESULT: BIT-IDENTICAL across scalar and AVX2+FMA."
  echo "        rsqrt is reproducible by construction. GOLDEN(rsqrt,f32,N=2^20) = $H_SCALAR"
else
  echo "RESULT: DIFFER - investigate (likely a fast-math/approx-rsqrt leak in the build)."
  exit 1
fi
