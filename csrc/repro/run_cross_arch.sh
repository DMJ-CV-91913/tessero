#!/usr/bin/env bash
# csrc/repro/run_cross_arch.sh
#
# Arch-aware cross-ISA / cross-width / cross-thread determinism check for the
# reproducible-f32 primitives in csrc/src/repro_f32.c. Complements the x86-only
# run_determinism.sh: this one also runs on ARM (NEON/SVE) and, for every build,
# compares each primitive's hash to the RECORDED x86 golden. Equal everywhere ==
# reproducible across SIMD width, thread count, AND CPU instruction set.
#
#   x86_64 : scalar / AVX2+FMA / AVX-512 (if present)
#   aarch64: scalar / NEON / SVE (if present)
#
# All builds -fno-fast-math -ffp-contract=off (never fast-math). Needs a C compiler
# with OpenMP + sha256sum. Run: bash csrc/repro/run_cross_arch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"
IMPL="$ROOT/src/repro_f32.c"; TEST="$HERE/determinism_test.c"; CC="${CC:-cc}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
COMMON="-fno-fast-math -ffp-contract=off -fopenmp"

# Recorded x86 goldens (csrc/repro/GOLDENS.md).
declare -A G=(
  [rsqrt]=fce3ce9bb62423fb56ec0c331d6b6262d8f76be629718572b37d31d80b9a696b
  [exp2]=d7b57bb03e2c26b20642f0697daf64a366bedd9d7a3ee5edd79d8dae502aabba
  [exp]=7c0748f832cf0c6ea64d0382af76615033014602c90e882437dac5108d0aa336
  [erf]=e268a1e2103e4ab6e33af6324bc5f01a208f18a0e3152f00309e9740c032a6eb
)
ARCH="$(uname -m)"; echo "arch: $ARCH   cc: $($CC --version | head -1)"

declare -a NAMES=(); declare -A FLAGS=()
add(){ NAMES+=("$1"); FLAGS[$1]="$2"; }
case "$ARCH" in
  x86_64|amd64)
    add scalar "-O2 -fno-tree-vectorize"
    grep -qi avx2 /proc/cpuinfo && add avx2 "-O3 -mavx2 -mfma"
    grep -qi avx512f /proc/cpuinfo && add avx512 "-O3 -march=x86-64-v4"
    ;;
  aarch64|arm64)
    add scalar "-O2 -fno-tree-vectorize"
    add neon "-O3"
    grep -qi sve /proc/cpuinfo 2>/dev/null && add sve "-O3 -march=armv9-a+sve2"
    ;;
  *) add scalar "-O2 -fno-tree-vectorize";;
esac

echo "== compiling: ${NAMES[*]} =="
for n in "${NAMES[@]}"; do
  $CC ${FLAGS[$n]} -std=gnu11 $COMMON "$IMPL" "$TEST" -o "$TMP/dt_$n" -lm 2>"$TMP/e_$n" \
    || echo "  [$n] compile failed (skipped): $(tail -1 "$TMP/e_$n")"
done

h(){ sha256sum | cut -d' ' -f1; }
echo; echo "== each primitive vs the recorded x86 golden, across all paths and 1/4 threads =="
FAIL=0
for fn in rsqrt exp2 exp erf recip tanh; do
  ok=1; detail=""
  for n in "${NAMES[@]}"; do
    [ -x "$TMP/dt_$n" ] || continue
    for t in 1 4; do
      hh="$(OMP_NUM_THREADS=$t "$TMP/dt_$n" "$fn" | h)"
      [ "$hh" = "${G[$fn]}" ] || { ok=0; detail="$detail ${n}/${t}t=$hh"; }
    done
  done
  if [ "$ok" = 1 ]; then echo "  [MATCH] $fn == x86 golden (${G[$fn]:0:16}...) on every path/thread"
  else echo "  [DIFF]  $fn$detail  (x86 golden ${G[$fn]})"; FAIL=1; fi
done
echo
if [ "$FAIL" = 0 ]; then
  echo "RESULT: all 4 primitives bit-identical to the x86 golden on $ARCH."
  echo "        reproducible across SIMD width, thread count, and CPU ISA."
else
  echo "RESULT: divergence on $ARCH (see DIFF). Record this arch's hashes."
  exit 1
fi
