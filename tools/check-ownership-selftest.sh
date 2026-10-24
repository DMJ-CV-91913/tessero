#!/usr/bin/env bash
# Self-test of tools/check-ownership.sh: a scratch tree with one planted violation per rule must report every
# rule, and the same tree with the violations corrected (or explained) must be clean. Deterministic; bash only.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/tools" "$T/csrc/src" "$T/csrc/cxx" "$T/csrc/include" "$T/ext" "$T/src/Native" "$T/src/Other"
cp "$HERE/check-ownership.sh" "$T/tools/"

plant() {
cat > "$T/csrc/src/k.c" <<'EOF'
/* free(p) in a comment and "free(p)" in a string are not findings */
static void k1(void)
{
    double *x = fn_arg_doubles(&a[0], &n);
    free(x);
    char *s = malloc(8);
    tsr_free(s, 8);
}
static void k2(tsr_result *r)
{
    r->arr.data = malloc(64);
    zval z;
}
EOF
cat > "$T/ext/e.c" <<'EOF'
static void e(tsr_obj *o, tsr_result *res)
{
    void *p = malloc(4);
    efree(o->block);
#pragma omp parallel for
    for (int i = 0; i < 4; i++) {}
}
EOF
cat > "$T/csrc/include/tessero.h" <<'EOF'
int tsr_with_callback(double (*f)(double), double x);
EOF
cat > "$T/src/Native/Registry.php" <<'EOF'
<?php
$ffi->tsr_free($r->arr->data, $r->bytes);
FFI::free($r->arr->data);
$owned = $ffi->new('char[8]');
FFI::free($owned);
EOF
cat > "$T/src/Other/Leak.php" <<'EOF'
<?php
$ffi->tsr_free($p, 8);
EOF
}

fix() {
cat > "$T/csrc/src/k.c" <<'EOF'
static void k1(void)
{
    double *x = fn_arg_doubles(&a[0], &n);
    fn_free_doubles(x, n);
    char *s = malloc(8);
    free(s);
}
static void k2(tsr_result *r)
{
    r->arr.data = tsr_alloc(64);
}
EOF
cat > "$T/ext/e.c" <<'EOF'
static void e(tsr_obj *o)
{
    void *p = emalloc(4);
    efree(p);
    tsr_free(o->block, o->block_bytes);
}
EOF
cat > "$T/csrc/include/tessero.h" <<'EOF'
int tsr_with_callback(double (*f)(double), double x);   /* ADR 0015: calling thread only */
EOF
cat > "$T/src/Native/Registry.php" <<'EOF'
<?php
$ffi->tsr_free($r->arr->data, $r->bytes);
$buf = $ffi->new('char[8]', false);
$keep[] = $buf;
foreach ($keep as $b) {
    FFI::free($b);
}
EOF
rm -f "$T/src/Other/Leak.php"
}

plant
out="$(bash "$T/tools/check-ownership.sh" 2>&1 || true)"
fail=0
for rule in K1 K2 K3 E1 E2 E3 H1 P1 P2; do
    if ! grep -q "^$rule " <<<"$out"; then echo "self-test: rule $rule did not fire"; fail=1; fi
done
[ "$(grep -c '^K1 ' <<<"$out")" = 2 ] || { echo "self-test: K1 should fire twice (free of counted, tsr_free of libc)"; fail=1; }
[ "$(grep -c '^P1 ' <<<"$out")" = 2 ] || { echo "self-test: P1 should fire twice (kernel data, owned FFI::new)"; fail=1; }
if grep -q "k.c:1:" <<<"$out"; then echo "self-test: a comment or string was reported"; fail=1; fi
fix
if ! bash "$T/tools/check-ownership.sh" >/dev/null 2>&1; then
    echo "self-test: the corrected tree is not clean:"; bash "$T/tools/check-ownership.sh" || true; fail=1
fi
[ $fail = 0 ] && echo "check-ownership self-test: every rule fires on its violation, the corrected tree is clean"
exit $fail
