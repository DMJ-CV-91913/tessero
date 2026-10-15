<?php

// Independent arbitrary-precision oracle: verify Tessero's double-precision results against references
// computed from scratch in BCMath (arbitrary-precision decimal) - a completely separate code path from the
// NumPy/SciPy fixtures. Transcendentals come from BCMath Taylor series at 50 digits; combinatorics/gamma at
// integers are exact big-integer arithmetic. Runs on one backend; exits non-zero on any mismatch.
//
//   php -d ffi.enable=1 tools/parity/run-oracle.php --backend=ffi
//   php -d extension=.../tessero.so tools/parity/run-oracle.php --backend=ext

declare(strict_types=1);

require __DIR__ . '/../../tests/autoload.php';

$backend = 'ffi';
foreach ($argv as $a) {
    if (preg_match('/^--backend=(ffi|ext)$/', $a, $m)) {
        $backend = $m[1];
    }
}
$ext = $backend === 'ext';
$MATH = $ext ? 'Tessero\\Ext\\Math' : 'Tessero\\Math';
$SPEC = $ext ? 'Tessero\\Ext\\Special' : 'Tessero\\Special';

const SCALE = 50;

/** Extract a PHP float from a Tessero scalar result (0-d NDArray, float, or int). */
function item(mixed $v): float
{
    if (is_int($v) || is_float($v)) {
        return (float) $v;
    }
    if (is_object($v)) {
        if (method_exists($v, 'item')) {
            return (float) $v->item();
        }
        if (method_exists($v, 'toList')) {
            $l = $v->toList();
            while (is_array($l)) {
                $l = $l[0];
            }
            return (float) $l;
        }
    }
    return (float) $v;
}

// ---- BCMath arbitrary-precision references (independent of NumPy/SciPy) ----
function bc_fact(int $n): string
{
    $r = '1';
    for ($i = 2; $i <= $n; $i++) {
        $r = bcmul($r, (string) $i);
    }
    return $r;
}

function bc_exp(string $x, int $s = SCALE): string
{
    $term = '1';
    $sum = '1';
    for ($n = 1; $n <= 80; $n++) {
        $term = bcdiv(bcmul($term, $x, $s + 12), (string) $n, $s + 12);
        $sum = bcadd($sum, $term, $s + 12);
    }
    return $sum;
}

function bc_sin(string $x, int $s = SCALE): string
{
    $x2 = bcmul($x, $x, $s + 12);
    $term = $x;          // x^1 / 1!
    $sum = $x;
    for ($k = 1; $k <= 40; $k++) {
        $den = (string) ((2 * $k) * (2 * $k + 1));
        $term = bcdiv(bcmul($term, $x2, $s + 12), $den, $s + 12);
        $sum = ($k % 2 === 1) ? bcsub($sum, $term, $s + 12) : bcadd($sum, $term, $s + 12);
    }
    return $sum;
}

function bc_cos(string $x, int $s = SCALE): string
{
    $x2 = bcmul($x, $x, $s + 12);
    $term = '1';         // x^0 / 0!
    $sum = '1';
    for ($k = 1; $k <= 40; $k++) {
        $den = (string) ((2 * $k - 1) * (2 * $k));
        $term = bcdiv(bcmul($term, $x2, $s + 12), $den, $s + 12);
        $sum = ($k % 2 === 1) ? bcsub($sum, $term, $s + 12) : bcadd($sum, $term, $s + 12);
    }
    return $sum;
}

function bc_ln(string $x, int $s = SCALE): string
{
    // ln(x) = 2 * atanh(u), u = (x-1)/(x+1); converges for x > 0
    $u = bcdiv(bcsub($x, '1', $s + 12), bcadd($x, '1', $s + 12), $s + 12);
    $u2 = bcmul($u, $u, $s + 12);
    $term = $u;
    $sum = $u;
    for ($k = 1; $k <= 120; $k++) {
        $term = bcmul($term, $u2, $s + 12);
        $sum = bcadd($sum, bcdiv($term, (string) (2 * $k + 1), $s + 12), $s + 12);
    }
    return bcmul('2', $sum, $s + 12);
}

const BC_PI = '3.14159265358979323846264338327950288419716939937511';

$fail = 0;
$checked = 0;

function check(string $label, float $got, string $bcRef, float $rtol): void
{
    global $fail, $checked;
    $checked++;
    $exp = (float) $bcRef;
    if (abs($got - $exp) > $rtol * (1.0 + abs($exp))) {
        $fail++;
        printf("FAIL %s: tessero %.17g vs bcmath %.17g (rel %.2e)\n", $label, $got, $exp, abs($got - $exp) / (1.0 + abs($exp)));
    }
}

function checkExactInt(string $label, float $got, string $bcRef): void
{
    global $fail, $checked;
    $checked++;
    // exact when the integer fits a double (< 2^53)
    if ((string) (int) round($got) !== $bcRef) {
        $fail++;
        printf("FAIL %s: tessero %s vs bcmath %s (exact int)\n", $label, (string) (int) round($got), $bcRef);
    }
}

// 1. Transcendentals vs BCMath Taylor (double precision ~1e-14)
foreach (['0.1', '0.5', '1.0', '1.5', '2.0', '3.0'] as $xs) {
    $x = (float) $xs;
    check("exp($xs)", item($MATH::exp($x)), bc_exp($xs), 1e-13);
    check("sin($xs)", item($MATH::sin($x)), bc_sin($xs), 1e-12);
    check("cos($xs)", item($MATH::cos($x)), bc_cos($xs), 1e-12);
    check("sqrt($xs)", item($MATH::sqrt($x)), bcsqrt($xs, SCALE), 1e-14);
    check("log($xs)", item($MATH::log($x)), bc_ln($xs), 1e-12);
    check("tan($xs)", item($MATH::tan($x)), bcdiv(bc_sin($xs), bc_cos($xs), SCALE), 1e-11);
}

// 2. Exact integer combinatorics vs BCMath (exact, must match to the integer)
foreach ([[10, 3], [20, 7], [30, 15], [40, 10]] as [$n, $k]) {
    checkExactInt("comb($n,$k)", item($SPEC::comb($n, $k)), bcdiv(bc_fact($n), bcmul(bc_fact($k), bc_fact($n - $k)), 0));
}
foreach ([5, 10, 15, 18] as $n) {
    checkExactInt("factorial($n)", item($SPEC::factorial($n)), bc_fact($n));
}

// 3. gamma at integers (= (n-1)!) and gamma(1/2) = sqrt(pi)
foreach ([3, 5, 7, 10] as $n) {
    checkExactInt("gamma($n)", item($SPEC::gamma((float) $n)), bc_fact($n - 1));
}
check('gamma(0.5)=sqrt(pi)', item($SPEC::gamma(0.5)), bcsqrt(BC_PI, SCALE), 1e-13);

// 4. Constant: exp(1) == e, cos(pi) == -1
check('exp(1)=e', item($MATH::exp(1.0)), bc_exp('1'), 1e-14);
check('cos(pi)', item($MATH::cos(M_PI)), '-1', 1e-14);

echo "oracle ({$GLOBALS['backend']}): {$checked} checks, {$fail} fail\n";
exit($fail === 0 ? 0 : 1);
