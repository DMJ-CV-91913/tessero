<?php

// Benchmark: Tessero (FFI) vs Tessero (native extension) vs NumPower (ext-ndarray, C/AVX2) on common ops.
// Run with all three available:
//   php -d ffi.enable=1 -d extension=<tessero.so> -d extension=<ndarray.so> bench-numpower.php

declare(strict_types=1);

require __DIR__ . '/../tessero/tests/autoload.php';

$haveNP  = class_exists('NDArray');
$haveExt = class_exists('Tessero\\Ext\\NDArray');

function med(callable $fn, int $iters = 7): float
{
    $fn(); $fn();                 // warmup
    $t = [];
    for ($i = 0; $i < $iters; $i++) {
        $s = hrtime(true);
        $fn();
        $t[] = (hrtime(true) - $s) / 1e6;   // ms
    }
    sort($t);
    return $t[intdiv(count($t), 2)];
}

function row(string $label, ?float $ffi, ?float $ext, ?float $np, ?float $flops = null): void
{
    $f = static fn (?float $v): string => $v === null ? '    -   ' : sprintf('%8.2f', $v);
    $g = '';
    if ($flops !== null) {
        $gf = static fn (?float $ms): string => $ms === null ? '  -  ' : sprintf('%5.1f', $flops / ($ms / 1e3) / 1e9);
        $g = "  | GFLOPS ffi {$gf($ffi)} ext {$gf($ext)} np {$gf($np)}";
    }
    printf("%-22s ffi %s ms | ext %s ms | np %s ms%s\n", $label, $f($ffi), $f($ext), $f($np), $g);
}

echo "backends: Tessero-FFI=yes  Tessero-ext=" . ($haveExt ? 'yes' : 'NO') . "  NumPower=" . ($haveNP ? 'yes' : 'NO') . "\n";
echo str_repeat('-', 92) . "\n";

// ---- matmul (n x n) ----
foreach ([256, 512, 1024] as $n) {
    $flops = 2.0 * $n * $n * $n;
    $aF = Tessero\NDArray::ones([$n, $n]); $bF = Tessero\NDArray::ones([$n, $n]);
    $ffi = med(fn () => $aF->matmul($bF));
    $ext = null;
    if ($haveExt) { $aE = Tessero\Ext\NDArray::ones([$n, $n]); $bE = Tessero\Ext\NDArray::ones([$n, $n]); $ext = med(fn () => $aE->matmul($bE)); }
    $np = null;
    if ($haveNP) { $aN = NDArray::ones([$n, $n]); $bN = NDArray::ones([$n, $n]); $np = med(fn () => NDArray::matmul($aN, $bN)); }
    row("matmul {$n}x{$n}", $ffi, $ext, $np, $flops);
}

// ---- elementwise sin on a large vector ----
$m = 2_000_000;
$vF = Tessero\NDArray::ones([$m]);
$ffi = med(fn () => $vF->sin());
$ext = $haveExt ? (function () use ($m) { $v = Tessero\Ext\NDArray::ones([$m]); return med(fn () => $v->sin()); })() : null;
$np  = $haveNP ? (function () use ($m) { $v = NDArray::ones([$m]); return med(fn () => NDArray::sin($v)); })() : null;
row("sin ({$m})", $ffi, $ext, $np);

// ---- elementwise add of two large vectors ----
$aF = Tessero\NDArray::ones([$m]); $bF2 = Tessero\NDArray::ones([$m]);
$ffi = med(fn () => $aF->add($bF2));
$ext = $haveExt ? (function () use ($m) { $a = Tessero\Ext\NDArray::ones([$m]); $b = Tessero\Ext\NDArray::ones([$m]); return med(fn () => $a->add($b)); })() : null;
$np  = $haveNP ? (function () use ($m) { $a = NDArray::ones([$m]); $b = NDArray::ones([$m]); return med(fn () => NDArray::add($a, $b)); })() : null;
row("add ({$m})", $ffi, $ext, $np);

// ---- sum reduction ----
$vF = Tessero\NDArray::ones([$m]);
$ffi = med(fn () => $vF->sum());
$ext = $haveExt ? (function () use ($m) { $v = Tessero\Ext\NDArray::ones([$m]); return med(fn () => $v->sum()); })() : null;
$np  = $haveNP ? (function () use ($m) { $v = NDArray::ones([$m]); return med(fn () => NDArray::sum($v)); })() : null;
row("sum ({$m})", $ffi, $ext, $np);

echo str_repeat('-', 92) . "\n";
echo "(median of 7 iters after 2 warmups; OPENBLAS_NUM_THREADS from env)\n";
