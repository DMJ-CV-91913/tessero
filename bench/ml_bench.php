<?php

/**
 * The functions classical machine learning calls (scikit-learn style), for the speed sweep's "Classical machine
 * learning" section: per-call times of those tier 3 and bench/usage_bench.php do not time (small inputs, as tier 3
 * uses), and realistic-size cases of the heavy ones. bench/ml_bench.py times the same calls on the same inputs (the
 * large inputs come from the same PCG64 streams: Tessero's generator reproduces NumPy's).
 *
 *   OPENBLAS_NUM_THREADS=1 php -d ffi.enable=1 bench/ml_bench.php [ffi|ext] calls > ml_calls_ffi.json   (us per call)
 *   OPENBLAS_NUM_THREADS=1 php -d ffi.enable=1 bench/ml_bench.php [ffi|ext] heavy > ml_heavy_ffi.json   (ms per case)
 *
 * A case the backend cannot run is reported on stderr and left out. With "ext", arrays are Tessero\Ext\NDArray and
 * the functions Tessero\Ext\*; the PHP-level classes (Minimize, LeastSquares, KDTree) are the same code on both and
 * take the backend's arrays. The large inputs are drawn through FFI and copied into extension arrays by bytes.
 */

declare(strict_types=1);

require __DIR__ . '/../tests/autoload.php';

use Tessero\Optimize\LeastSquares;
use Tessero\Optimize\Minimize;
use Tessero\Random\Generator;
use Tessero\Spatial\KDTree;

$ext = ($argv[1] ?? 'ffi') === 'ext';
$part = $argv[2] ?? 'calls';
$c = static fn (string $name): string => $ext ? "Tessero\\Ext\\{$name}" : "Tessero\\{$name}";
$N = $c('NDArray');
$M = $c('Math');
$SL = $c('ScipyLinalg');
$SP = $c('Special');
$ST = $c('Stats');
$SPA = $c('Sparse');
$DI = $c('Distance');
$SVD = $c('SparseLinalg');
$G = $ext ? \Tessero\Ext\Random\Generator::class : Generator::class;
// an FFI array as the backend's array (bytes copied: the same values)
$as = static fn (\Tessero\NDArray $a): object => $ext ? \Tessero\Ext\NDArray::fromBytes($a->toBytes(), 'float64', $a->shape()) : $a;

/** best of $rounds runs of $inner calls, in microseconds per call */
$time = static function (callable $f, int $inner, int $rounds): float {
    $best = INF;
    for ($r = 0; $r < $rounds; $r++) {
        $t = hrtime(true);
        for ($i = 0; $i < $inner; $i++) {
            $f();
        }
        $best = min($best, (hrtime(true) - $t) / $inner / 1e3);
    }

    return $best;
};

if ($part === 'calls') {
    $x = $N::array([0.5, -1.25, 2.0, 3.75, -0.5, 1.5, 4.25, 0.125]);
    $m = $N::array([[2.0, 0.5, 0.25], [0.5, 3.0, 0.125], [0.25, 0.125, 4.0]]);
    $b = $N::array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0], [4.0, 5.0, 0.0]]);
    $pts = $N::array([[0.1, 0.2], [0.4, 0.9], [0.8, 0.3], [0.5, 0.5], [0.9, 0.9], [0.2, 0.7], [0.6, 0.1], [0.3, 0.4]]);
    $norm = $ST::norm();
    $g = $G::defaultRng(12345);
    $cases = [
        'numpy.matmul' => static fn () => $m->matmul($m),
        'numpy.std' => static fn () => $x->std(),
        'numpy.var' => static fn () => $x->var(),
        'scipy.special.expit' => static fn () => $SP::expit($x),
        'scipy.sparse.hstack' => static fn () => $SPA::hstack($b, $b),
        'scipy.spatial.cKDTree' => static fn () => new KDTree($pts),
        'scipy.optimize.minimize' => static fn () => Minimize::minimize(static fn (array $v): float => ($v[0] - 1.0) ** 2 + 10.0 * ($v[1] - 2.0) ** 2, [0.0, 0.0], null, 'L-BFGS-B'),
        'scipy.optimize.least_squares' => static fn () => LeastSquares::leastSquares(static fn (array $v): array => [$v[0] - 1.0, 10.0 * ($v[1] - 2.0), $v[0] * $v[1] - 2.0], [0.0, 0.0]),
        'scipy.stats.norm.pdf' => static fn () => $norm->pdf($x),
        'scipy.stats.norm.logpdf' => static fn () => $norm->logpdf($x),
        'scipy.stats.norm.rvs' => static fn () => $norm->rvs(8, $g),
        'Generator.normal' => static fn () => $g->normal(0.0, 1.0, 8),
        'Generator.uniform' => static fn () => $g->uniform(0.0, 1.0, 8),
        'Generator.choice' => static fn () => $g->choice(8, 4),
        'Generator.permutation' => static fn () => $g->permutation(8),
    ];
    $out = [];
    foreach ($cases as $name => $f) {
        try {
            $f();
        } catch (\Throwable $e) {
            fwrite(STDERR, "{$name}: " . get_class($e) . ': ' . $e->getMessage() . "\n");
            continue;
        }
        $slow = str_starts_with($name, 'scipy.optimize');
        $out[$name] = round($time($f, $slow ? 50 : 2000, 7), 4);
    }
    echo json_encode(['backend' => $ext ? 'ext' : 'ffi', 'unit' => 'us', 'calls' => $out], JSON_PRETTY_PRINT), "\n";
    exit;
}

// realistic sizes; inputs from Generator::defaultRng streams that bench/ml_bench.py draws identically
$rng = Generator::defaultRng(2026);
$A2k = $as($rng->random([2000, 2000]));
$B2k = $as($rng->random([2000, 2000]));
$A1kF = $rng->random([1000, 1000]);
$A1k = $as($A1kF);
$S1k = $as($A1kF->add($A1kF->transpose()));                                      // symmetric
$P1k = $as($A1kF->matmul($A1kF->transpose())->add(\Tessero\NDArray::eye(1000)->mul(1000.0)));   // positive definite
$b1k = $as($rng->random([1000]));
$Xl = $as($rng->random([10000, 50]));
$yl = $as($rng->random([10000]));
$Xp = $as($rng->random([2000, 50]));
$Xa = $as($rng->random([1000, 50]));
$Xb = $as($rng->random([1000, 50]));
$Xs = $as($rng->random([2000, 500]));
$pts = $as($rng->random([100000, 3]));
$qpts = $as($rng->random([1000, 3]));
$v1m = $as($rng->standardNormal([1000000]));
// logistic regression data: features, labels in {-1, 1} from a fixed weight vector
$Xlr = $rng->standardNormal([10000, 50]);
$wtrue = $rng->standardNormal([50]);
$ylr = $as($Xlr->matmul($wtrue)->gt(0.0)->astype('float64')->mul(2.0)->sub(1.0));
$Xlr = $as($Xlr);
$XlrT = $Xlr->transpose();
// loss = mean(log(1 + exp(-y z))), z = X w; gradient = X^T (-y expit(-y z)) / n
$lrFun = static function (array $w) use ($N, $M, $Xlr, $ylr): float {
    $nyz = $Xlr->matmul($N::array($w))->mul($ylr)->neg();
    return (float) $M::log1p($M::exp($nyz))->mean();
};
$lrJac = static function (array $w) use ($N, $SP, $Xlr, $XlrT, $ylr): array {
    $nyz = $Xlr->matmul($N::array($w))->mul($ylr)->neg();
    return $XlrT->matmul($SP::expit($nyz)->mul($ylr)->neg())->div(10000.0)->toList();
};
$w0 = array_fill(0, 50, 0.0);
$tree = new KDTree($pts);

$cases = [
    'numpy.matmul 2000x2000' => static fn () => $A2k->matmul($B2k),
    'scipy.linalg.svd 1000x1000 (full)' => static fn () => $SL::svd($A1k),
    'scipy.linalg.eigh 1000x1000' => static fn () => $SL::eigh($S1k),
    'scipy.linalg.solve 1000x1000' => static fn () => $SL::solve($A1k, $b1k),
    'scipy.linalg.cholesky 1000x1000' => static fn () => $SL::cholesky($P1k),
    'scipy.linalg.lstsq 10,000 x 50' => static fn () => $SL::lstsq($Xl, $yl),
    'scipy.sparse.linalg.svds k=10, 2,000 x 500 (dense input)' => static fn () => $SVD::svds($Xs, 10),
    'scipy.spatial.distance.pdist 2,000 x 50 (euclidean)' => static fn () => $DI::pdist($Xp),
    'scipy.spatial.distance.cdist 1,000 x 1,000 x 50 (euclidean)' => static fn () => $DI::cdist($Xa, $Xb),
    'scipy.spatial.cKDTree build 100,000 x 3' => static fn () => new KDTree($pts),
    'scipy.spatial.cKDTree query 1,000 points, k=5' => static fn () => $tree->query($qpts, 5),
    'logistic regression, L-BFGS-B, 10,000 x 50' => static fn () => Minimize::minimize($lrFun, $w0, $lrJac, 'L-BFGS-B'),
    'logistic regression, Newton-CG, 10,000 x 50' => static fn () => Minimize::minimize($lrFun, $w0, $lrJac, 'Newton-CG'),
    'logistic regression, TNC, 10,000 x 50' => static fn () => Minimize::minimize($lrFun, $w0, $lrJac, 'TNC'),
    'scipy.special.expit 1M' => static fn () => $SP::expit($v1m),
    'scipy.special.logsumexp 1M' => static fn () => $SP::logsumexp($v1m),
    'scipy.stats.norm.logpdf 1M' => static fn () => $ST::norm()->logpdf($v1m),
];
$out = [];
foreach ($cases as $name => $f) {
    try {
        $t0 = hrtime(true);
        $r = $f();
        $first = (hrtime(true) - $t0) / 1e6;
        if (str_starts_with($name, 'logistic')) {
            fwrite(STDERR, sprintf("%s: nit %s nfev %s fun %.12g\n", $name, $r->nit ?? '?', $r->nfev ?? '?', $r->fun ?? NAN));
        }
    } catch (\Throwable $e) {
        fwrite(STDERR, "{$name}: " . get_class($e) . ': ' . $e->getMessage() . "\n");
        continue;
    }
    $rounds = $first > 2000 ? 2 : ($first > 200 ? 3 : 5);
    $inner = $first < 2 ? 20 : 1;
    $out[$name] = round($time($f, $inner, $rounds) / 1e3, 4);
}
echo json_encode(['backend' => $ext ? 'ext' : 'ffi', 'unit' => 'ms', 'cases' => $out], JSON_PRETTY_PRINT), "\n";
