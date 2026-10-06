<?php

/**
 * Shared harness for the Tessero examples cookbook (not part of the public API).
 *
 * tools/run-examples.php auto-prepends this file, so every example in examples/ can:
 *   - run on either backend: backend() is 'ffi' or 'ext' (from TESSERO_EXAMPLE_BACKEND);
 *   - use the short, backend-selected aliases ND, NP, M, RNG, SP, ST, FFT for the classes
 *     that exist on BOTH backends (e.g. ND::array(...) is Tessero\NDArray on FFI and
 *     Tessero\Ext\NDArray on the extension);
 *   - mark an FFI-only example with ffi_only('scipy.optimize'), which SKIPs it under ext;
 *   - assert on results with check()/check_close() so the example fails loudly on drift.
 *
 * FFI-only surfaces (optimize, linalg, sparse, io, mdp, csgraph, integrate, spatial,
 * datetime, Laravel) use their real class names directly and call ffi_only() at the top.
 */

declare(strict_types=1);

require __DIR__ . '/../vendor/autoload.php';

function backend(): string
{
    return getenv('TESSERO_EXAMPLE_BACKEND') === 'ext' ? 'ext' : 'ffi';
}

$__ext = backend() === 'ext';
if ($__ext && ! extension_loaded('tessero')) {
    fwrite(STDOUT, "SKIP: ext backend requested but the tessero extension is not loaded\n");
    exit(0);
}

// Short aliases for the classes that exist on both backends.
foreach ([
    'ND' => ['Tessero\\NDArray', 'Tessero\\Ext\\NDArray'],
    'NP' => ['Tessero\\Np', 'Tessero\\Ext\\Np'],
    'M' => ['Tessero\\Math', 'Tessero\\Ext\\Math'],
    'RNG' => ['Tessero\\Random\\Generator', 'Tessero\\Ext\\Random\\Generator'],
    'SP' => ['Tessero\\Special', 'Tessero\\Ext\\Special'],
    'ST' => ['Tessero\\Stats', 'Tessero\\Ext\\Stats'],
    'FFT' => ['Tessero\\Fft\\Fft', 'Tessero\\Ext\\Fft\\Fft'],
] as $alias => [$ffi, $ext]) {
    if (! class_exists($alias, false)) {
        class_alias($__ext ? $ext : $ffi, $alias);
    }
}

/** Print a line (examples echo their result, then assert on it). */
function say(string $msg): void
{
    fwrite(STDOUT, $msg . "\n");
}

/** Skip the example on this backend with a clear reason (exit 0; the harness counts it as a skip). */
function skip(string $why): never
{
    fwrite(STDOUT, "SKIP: {$why}\n");
    exit(0);
}

/** Skip on the extension backend: the feature only exists in the FFI package. */
function ffi_only(string $feature = 'this feature'): void
{
    if (backend() === 'ext') {
        skip("{$feature} is FFI-only (no native-extension equivalent yet)");
    }
}

function to_list(mixed $v): mixed
{
    return is_object($v) && method_exists($v, 'toList') ? $v->toList() : $v;
}

/** Flatten a scalar / nested array / NDArray to a flat list of values. */
function flat(mixed $v): array
{
    $v = to_list($v);
    if (! is_array($v)) {
        return [$v];
    }
    $out = [];
    array_walk_recursive($v, static function ($x) use (&$out): void { $out[] = $x; });

    return $out;
}

function close(float $a, float $b, float $rtol = 1e-9, float $atol = 1e-12): bool
{
    if (is_nan($a) && is_nan($b)) {
        return true;
    }

    return abs($a - $b) <= $atol + $rtol * abs($b);
}

/** Assert a condition; on failure print FAIL and exit non-zero so the harness marks it failed. */
function check(bool $cond, string $msg): void
{
    if (! $cond) {
        fwrite(STDERR, "FAIL: {$msg}\n");
        exit(1);
    }
}

/** Assert two values (scalars, lists or NDArrays) are element-wise close. */
function check_close(mixed $got, mixed $want, string $msg, float $rtol = 1e-6, float $atol = 1e-9): void
{
    $g = flat($got);
    $w = flat($want);
    if (count($g) !== count($w)) {
        fwrite(STDERR, sprintf("FAIL: %s: %d values vs %d\n", $msg, count($g), count($w)));
        exit(1);
    }
    foreach ($w as $i => $wi) {
        if (! close((float) $g[$i], (float) $wi, $rtol, $atol)) {
            fwrite(STDERR, sprintf("FAIL: %s: element %d got %.12g want %.12g\n", $msg, $i, (float) $g[$i], (float) $wi));
            exit(1);
        }
    }
}
