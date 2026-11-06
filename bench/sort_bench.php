<?php
/**
 * Tessero's sorts against NumPy, as a PHP app sees them: 1,000,000 values per case, best of 7 calls; each call returns
 * a new sorted array (as np.sort does). bench/sort_bench.py runs NumPy on the same inputs (same seeds).
 *
 *   OPENBLAS_NUM_THREADS=1 php -d ffi.enable=1 -d memory_limit=-1 bench/sort_bench.php > tessero.json
 */
declare(strict_types=1);
require __DIR__ . '/../tests/autoload.php';
use Tessero\NDArray;
use Tessero\Random\Generator as G;
use Tessero\Tessero;

$n = 1000000;
$best = function (callable $f): float {
    $f();
    $b = INF;
    for ($i = 0; $i < 7; $i++) { $s = hrtime(true); $f(); $b = min($b, (hrtime(true) - $s) / 1e6); }
    return round($b, 3);
};
$rand = G::defaultRng(7)->random(size: $n)->mul(2.0)->sub(1.0);
$sorted = $rand->sort();
$sv = $sorted->toArray();
$moved = $sv;
for ($i = 0; $i < intdiv($n, 100); $i++) {                // 1% of the values swapped to other places
    $j = ($i * 104729) % $n; $k = ($i * 1299709 + 12345) % $n;
    [$moved[$j], $moved[$k]] = [$moved[$k], $moved[$j]];
}
$app = array_merge(array_slice($sv, 0, $n - intdiv($n, 100)), G::defaultRng(8)->random(size: intdiv($n, 100))->mul(2.0)->sub(1.0)->toArray());
$cases = [
    'float64 random' => $rand,
    'float64 10 distinct values' => $rand->add(1.0)->mul(5.0)->floor(),
    'float64 sorted' => $sorted,
    'float64 reversed' => NDArray::array(array_reverse($sv)),
    'float64 sorted, 1% of values moved' => NDArray::array($moved),
    'float64 sorted, 1% random values appended' => NDArray::array($app),
    'float32 random' => $rand->astype('float32'),
    'int64 random' => $rand->mul(4.0e18)->astype('int64'),
];
$out = [];
foreach ($cases as $name => $a) {
    Tessero::setThreads(1);
    $r = ['sort' => $best(fn () => $a->sort()), 'sortSimd' => $best(fn () => $a->sortSimd()), 'sortRuns' => $best(fn () => $a->sortRuns())];
    Tessero::setThreads(4);
    $r['sortSimd 4 threads'] = $best(fn () => $a->sortSimd());
    Tessero::setThreads(1);
    $out[$name] = $r;
    fwrite(STDERR, "$name done\n");
}
echo json_encode($out, JSON_PRETTY_PRINT), "\n";
