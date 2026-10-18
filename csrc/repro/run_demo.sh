#!/usr/bin/env bash
# csrc/repro/run_demo.sh
# Arch-aware reproducibility check for the repro-f32 primitives (currently rsqrt).
# Compiles demo_rsqrt.c on every vector path the CPU supports, hashes each output,
# and compares them to each other AND to the recorded GOLDEN. Equal everywhere ==
# reproducible across SIMD width and (on ARM) across CPU ISA.
#
#   x86_64 : scalar, AVX2+FMA, and AVX-512 (if present)
#   aarch64: scalar, NEON/ASIMD (and SVE if the compiler takes it)
#
# No -ffast-math anywhere (it would swap in the non-reproducible approx rsqrt).
# Run: bash csrc/repro/run_demo.sh   (needs a C compiler + sha256sum)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/demo_rsqrt.c"
CC="${CC:-cc}"
GOLDEN="fce3ce9bb62423fb56ec0c331d6b6262d8f76be629718572b37d31d80b9a696b"   # rsqrt f32 N=2^20

ARCH="$(uname -m)"
BASE="-O2 -std=c11 -fno-fast-math -ffp-contract=off"
VEC="-O3 -std=c11 -fno-fast-math -ffp-contract=off"
echo "arch: $ARCH   compiler: $($CC --version | head -1)"

# build list: "label|flags"
BUILDS=()
case "$ARCH" in
  x86_64|amd64)
    BUILDS+=("scalar|$BASE -fno-tree-vectorize")
    grep -qiE '(^|[^a-z])avx2([^a-z]|$)' /proc/cpuinfo && BUILDS+=("avx2|$VEC -mavx2 -mfma")
    grep -qi 'avx512f' /proc/cpuinfo && BUILDS+=("avx512|$VEC -mavx512f -mavx512dq -mavx512vl -mfma")
    ;;
  aarch64|arm64)
    BUILDS+=("scalar|$BASE -fno-tree-vectorize")
    BUILDS+=("neon|$VEC")                               # NEON/ASIMD is baseline on ARMv8; -O3 vectorizes
    grep -qi 'sve' /proc/cpuinfo 2>/dev/null && BUILDS+=("sve|$VEC -march=armv9-a+sve2")
    ;;
  *) BUILDS+=("scalar|$BASE") ;;
esac

declare -A H
FAIL=0
for b in "${BUILDS[@]}"; do
  label="${b%%|*}"; flags="${b#*|}"
  out="/tmp/repro_rsqrt_${label}"
  if ! $CC $flags "$SRC" -o "$out" -lm 2>/tmp/cc_err_$label; then
    echo "  [$label] compile FAILED (skipped): $(tail -1 /tmp/cc_err_$label)"; continue
  fi
  h="$("$out" | sha256sum | cut -d' ' -f1)"
  H[$label]="$h"
  tag="golden-MATCH"; [ "$h" = "$GOLDEN" ] || { tag="golden-DIFF"; FAIL=1; }
  printf "  [%-7s] %s  %s\n" "$label" "$h" "$tag"
done

echo
# all local builds equal each other?
uniq_hashes="$(printf '%s\n' "${H[@]}" | sort -u | wc -l)"
if [ "$uniq_hashes" = "1" ] && [ "$FAIL" = "0" ]; then
  echo "RESULT: BIT-IDENTICAL across all vector paths on $ARCH, and == GOLDEN."
  echo "        rsqrt reproducible on this architecture. GOLDEN=$GOLDEN"
elif [ "$uniq_hashes" = "1" ] && [ "$FAIL" = "1" ]; then
  echo "RESULT: self-consistent on $ARCH but != the x86 golden."
  echo "        => a cross-ISA/arch divergence. Record this arch's hash: ${H[scalar]:-?}"
  exit 2
else
  echo "RESULT: vector paths DISAGREE on $ARCH (investigate a fast-math/approx leak)."
  exit 1
fi
