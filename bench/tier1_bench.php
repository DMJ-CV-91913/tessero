<?php

/**
 * Tessero vs SciPy, the element-wise tier of the speed sweep: every scipy.special function on 100,000 values. The
 * inputs are the parity fixture's rows (tests/fixtures/parity/special.json, the first float64 block of each
 * function), each argument column tiled to N values. bench/tier1_bench.py times SciPy on the same inputs;
 * bench/sweep_report.py pairs the two files.
 *
 *   OPENBLAS_NUM_THREADS=1 php -d ffi.enable=1 -d memory_limit=-1 bench/tier1_bench.php > tessero.json
 *   ... bench/tier1_bench.php gamma            # only the functions whose name contains "gamma"
 */

declare(strict_types=1);
require __DIR__ . '/../tests/autoload.php';
use Tessero\NDArray;
use Tessero\Special;

const N = 100000;

/** median of repeated runs in ms (as bench/tier2_bench.php) */
function med(callable $f): float
{
    $s = hrtime(true);
    $f();
    $first = (hrtime(true) - $s) / 1e6;
    if ($first > 30000) {
        return $first;                           /* over 30 s: one run says enough */
    }
    $inner = (int) max(1, min(500, 10.0 / max($first, 1e-4)));
    $n = $first > 500 ? 3 : 9;
    $t = [];
    for ($i = 0; $i < $n; $i++) {
        $s = hrtime(true);
        for ($j = 0; $j < $inner; $j++) {
            $f();
        }
        $t[] = (hrtime(true) - $s) / 1e6 / $inner;
    }
    sort($t);

    return $t[intdiv($n, 2)];
}

/** a fixture column ("nan", "inf", "-inf" as strings) tiled to N float64 values */
function tiled(array $col): NDArray
{
    $vals = array_map(static fn ($v): float => match ($v) {
        'nan', 'undef' => NAN, 'inf' => INF, '-inf' => -INF, default => (float) $v,
    }, $col);
    $out = [];
    while (count($out) < N) {
        array_push($out, ...$vals);
    }

    return NDArray::array(array_slice($out, 0, N), 'float64');
}

$fixture = json_decode((string) file_get_contents(__DIR__ . '/../tests/fixtures/parity/special.json'), true, 64, JSON_THROW_ON_ERROR);
$only = $argv[1] ?? null;
$ms = [];
foreach ($fixture['blocks'] as $blk) {
    $fn = $blk['fn'];
    if ($blk['int'] || isset($ms[$fn]) || ($only !== null && ! str_contains($fn, $only))) {
        continue;                                /* the float64 loop, once per function */
    }
    $parts = explode('_', $fn);                  /* the facade's name, as gen-facades.php camel() makes it */
    $method = array_shift($parts);
    foreach ($parts as $p) {
        $method .= $p === '' ? '_' : ucfirst($p);
    }
    fwrite(STDERR, "{$fn}\n");
    try {
        $args = array_map('tiled', $blk['args']);
        $ms[$fn] = med(static fn () => Special::$method(...$args));
    } catch (\Throwable $e) {
        $ms[$fn] = 'ERR ' . get_class($e) . ': ' . substr($e->getMessage(), 0, 160);
    }
}
echo json_encode(['n' => N, 'ms' => $ms], JSON_PRETTY_PRINT), "\n";
