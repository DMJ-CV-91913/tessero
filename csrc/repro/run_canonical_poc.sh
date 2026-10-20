#!/usr/bin/env bash
# run_canonical_poc.sh -- compile canonical_poc.c under a flag matrix and print the
# hashes side by side, so the root cause is MEASURED, not guessed. Run on an x86-64
# box to reproduce the exact issue (aarch64 will show CANON stable but not the x86
# clone behaviour). Set CC to compare compilers, e.g.  CC=gcc-12 bash run_canonical_poc.sh
#
# Read the output:
#   - CANON identical on every row            -> the explicit-fmaf fix is canonical.
#   - the row(s) where `contracted` or NAIVE
#     differ from the others                  -> the measured root cause (contraction
#                                                 flag and/or clone selection).
# No em dashes. Per-process temp (build server is multi-user).
set -u
CC="${CC:-cc}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/canonical_poc.c"
BIN="/tmp/canon_poc.$$"
echo "compiler: $($CC --version 2>/dev/null | head -1)"
echo "source:   $SRC"
echo

# label | extra flags
MATRIX=(
  "O3-default            | -O3"
  "O3-contract-off       | -O3 -ffp-contract=off"
  "O3-contract-on        | -O3 -ffp-contract=on"
  "O3-contract-fast      | -O3 -ffp-contract=fast"
  "O3-mfma-contract-fast | -O3 -mfma -ffp-contract=fast"
  "O3-ffast-math         | -O3 -ffast-math"
  "O0                    | -O0"
  "O3-NO_CLONES          | -O3 -DNO_CLONES"
)
for row in "${MATRIX[@]}"; do
  label="${row%%|*}"; flags="${row#*|}"
  if $CC $flags "$SRC" -o "$BIN" -lm 2>/tmp/canon_cc.$$; then
    printf '%-24s ' "$label"; "$BIN"
  else
    printf '%-24s COMPILE FAILED: %s\n' "$label" "$(head -1 /tmp/canon_cc.$$)"
  fi
done
rm -f "$BIN" /tmp/canon_cc.$$
echo
echo "VERDICT (kernel): the fix works iff the CANON column is one identical value on every row."

# --- approval gate layer: fast clone must equal the scalar reference, else substitute ---
echo
echo "=== approval gate (runtime substitution layer) ==="
GBIN="/tmp/canon_gate.$$"
if $CC -O3 "$HERE/approval_gate_poc.c" -o "$GBIN" -lm 2>/tmp/canon_cc.$$; then
  "$GBIN"; rc=$?
  case $rc in
    0) echo "VERDICT (gate): APPROVED -- vectorized clone is bit-identical to the reference." ;;
    1) echo "VERDICT (gate): SUBSTITUTED -- clone diverged, reference was forced (gate did its job)." ;;
    *) echo "VERDICT (gate): HARD FAILURE -- reference self-inconsistent (investigate before shipping)." ;;
  esac
else
  echo "approval gate: COMPILE FAILED: $(head -1 /tmp/canon_cc.$$)"
fi
rm -f "$GBIN" /tmp/canon_cc.$$
