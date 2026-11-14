<?php

/**
 * Per-call times of the most-called NumPy/SciPy functions of the usage corpus (tools/parity/usage.json) that the
 * speed sweep's tier 3 does not time: array creation, element-wise math, reductions, dot. Small inputs, as tier 3
 * uses: 8 values, a 3 x 3 matrix, a PHP list where NumPy code passes a Python list. bench/usage_bench.py times the
 * same calls with the same inputs. Writes {"backend", "unit": "us", "calls": {"np.name": best-of us}}.
 *
 *   OPENBLAS_NUM_THREADS=1 php -d ffi.enable=1 bench/usage_bench.php [ext] > usage_ffi.json
 *   python3 bench/usage_bench.py > usage_numpy.json
 *   php bench/usage_rows.php usage_ffi.json usage_numpy.json > usage_rows_ffi.json   (rows for tools/parity/usage_speed.php)
 */

declare(strict_types=1);

require __DIR__ . '/../tests/autoload.php';

$ext = ($argv[1] ?? '') === 'ext';
$N = $ext ? \Tessero\Ext\NDArray::class : \Tessero\NDArray::class;
$M = $ext ? \Tessero\Ext\Math::class : \Tessero\Math::class;
$S = $ext ? \Tessero\Ext\Stats::class : \Tessero\Stats::class;

$list = [0.5, -1.25, 2.0, 3.75, -0.5, 1.5, 4.25, 0.125];
$x = $N::array($list);
$pos = $N::array([0.5, 1.25, 2.0, 3.75, 0.25, 1.5, 4.25, 0.125]);
$y = $N::array([1.0, 0.5, -2.0, 0.25, 3.0, -1.5, 0.75, 2.5]);
$m = $N::array([[2.0, 0.5, 0.25], [0.5, 3.0, 0.125], [0.25, 0.125, 4.0]]);
$mask = $x->gt(0.0);
$norm = $S::norm();
$chi2 = $S::chi2(3.0);

$cases = [
    'np.array' => static fn () => $N::array($list),
    'np.asarray' => static fn () => $ext ? $N::array($list) : $N::asArray($list),
    'np.zeros' => static fn () => $N::zeros([8]),
    'np.dot' => static fn () => $m->dot($m),
    'np.arange' => static fn () => $N::arange(8),
    'np.sqrt' => static fn () => $M::sqrt($pos),
    'np.sum' => static fn () => $x->sum(),
    'np.absolute' => static fn () => $M::absolute($x),
    'np.log' => static fn () => $M::log($pos),
    'np.empty' => static fn () => $N::empty([8]),
    'np.where' => static fn () => $N::where($mask, $x, $y),
    'np.ones' => static fn () => $N::ones([8]),
    'np.exp' => static fn () => $M::exp($x),
    'np.any' => static fn () => $mask->any(),
    'np.isnan' => static fn () => $M::isnan($x),
    'np.mean' => static fn () => $x->mean(),
    'np.max' => static fn () => $x->max(),
    'np.all' => static fn () => $mask->all(),
    'np.eye' => static fn () => $N::eye(3),
    'np.linspace' => static fn () => $N::linspace(0.0, 1.0, 8),
    'np.isfinite' => static fn () => $M::isfinite($x),
    'np.clip' => static fn () => $x->clip(-1.0, 1.0),
    'np.cos' => static fn () => $M::cos($x),
    'np.maximum' => static fn () => $M::maximum($x, $y),
    'np.sin' => static fn () => $M::sin($x),
    'np.argsort' => static fn () => $x->argsort(),
    'np.min' => static fn () => $x->min(),
    'np.full' => static fn () => $N::full([8], 2.5),
    'np.cumsum' => static fn () => $x->cumsum(),
    'np.ceil' => static fn () => $M::ceil($x),
    'np.sort' => static fn () => $x->sort(),
    'np.reshape' => static fn () => $x->reshape(2, 4),
    'np.ravel' => static fn () => $m->ravel(),
    'np.argmax' => static fn () => $x->argmax(),
    'np.prod' => static fn () => $x->prod(),
    'np.log10' => static fn () => $M::log10($pos),
    'np.logical_and' => static fn () => $M::logicalAnd($mask, $mask),
    'np.sign' => static fn () => $M::sign($x),
    'np.floor' => static fn () => $M::floor($x),
    'np.minimum' => static fn () => $M::minimum($x, $y),
    'np.argmin' => static fn () => $x->argmin(),
    'np.power' => static fn () => $M::power($pos, 1.5),
    'stats.norm' => static fn () => $norm->cdf($x),
    'stats.chi2' => static fn () => $chi2->sf($pos),
];

$out = [];
foreach ($cases as $name => $f) {
    try {
        $f();
    } catch (\Throwable $e) {
        fwrite(STDERR, "{$name}: " . $e->getMessage() . "\n");
        continue;
    }
    $best = INF;
    for ($r = 0; $r < 7; $r++) {
        $t = hrtime(true);
        for ($i = 0; $i < 2000; $i++) {
            $f();
        }
        $best = min($best, (hrtime(true) - $t) / 2000 / 1e3);
    }
    $out[$name] = round($best, 4);
}
echo json_encode(['backend' => $ext ? 'ext' : 'ffi', 'unit' => 'us', 'calls' => $out], JSON_PRETTY_PRINT), "\n";
