<?php

/**
 * Tessero vs SciPy: the SciPy-equivalent cases of docs/project/benchmarks.md, timed as a PHP app sees them.
 * bench/scipy_bench.py times the same cases with the same inputs (same seeds: the generators match NumPy).
 *
 *   OPENBLAS_NUM_THREADS=1 php -d ffi.enable=1 bench/scipy_bench.php > tessero.json
 */

declare(strict_types=1);
require __DIR__ . '/../tests/autoload.php';
use Tessero\NDArray; use Tessero\Random\Generator as G; use Tessero\ScipyLinalg as SL; use Tessero\Special as SP; use Tessero\Stats as ST;
use Tessero\Signal as SG; use Tessero\Distance as D; use Tessero\Ndimage as ND;
use Tessero\Optimize\LinearProgramming as LP; use Tessero\Optimize\Minimize as MZ; use Tessero\Optimize\Root as RT;

function med(callable $f, int $n = 11, int $inner = 1): float
{
    $f();
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

$r = fn (int $seed, array|int $size) => G::defaultRng($seed)->random(size: $size);
$A = $r(1, [600, 600])->add(NDArray::eye(600)->mul(600.0));
$b = $r(2, 600);
$M3 = $r(3, [300, 300]);
$S3 = $M3->matmul($M3->transpose())->add(NDArray::eye(300)->mul(300.0));
$M4 = $r(4, [300, 300]);
$E = $r(5, [200, 200])->mul(0.01);
$x = $r(10, 1000000)->mul(10.0);
$y = $r(11, 1000000);
$ta = $r(12, 100000);
$tb = $r(13, 100000)->add(0.01);
$sig = $r(14, 100000);
$ker = $r(15, 1001);
$s10 = $r(14, 10000);
$k101 = $r(15, 101);
$XA = $r(16, [1000, 3]);
$XB = $r(17, [1000, 3]);
$X2 = $r(18, [2000, 3]);
$img = $r(19, [1000, 1000]);
$g = G::defaultRng(20);
$lpA = $g->uniform(0.1, 1.0, [200, 200])->toArray();
$lpB = $g->uniform(50, 100, 200)->toList();
$lpC = $g->uniform(-1, 0, 200)->toList();
$xm5 = $x->sub(5.0);
$xp = $x->add(0.1);
$norm = ST::norm();
$rosen = function (array $v): float {
    $s = 0.0;
    for ($i = 0, $n = count($v) - 1; $i < $n; $i++) {
        $s += 100.0 * ($v[$i + 1] - $v[$i] ** 2) ** 2 + (1 - $v[$i]) ** 2;
    }

    return $s;
};

$cases = [
    'scipy.linalg.solve 600x600' => fn () => SL::solve($A, $b),
    'scipy.linalg.lu_factor 600x600' => fn () => SL::luFactor($A),
    'scipy.linalg.inv 600x600' => fn () => SL::inv($A),
    'scipy.linalg.det 600x600' => fn () => SL::det($A),
    'scipy.linalg.eigh 300x300' => fn () => SL::eigh($S3),
    'scipy.linalg.svd 300x300' => fn () => SL::svd($M4),
    'scipy.linalg.expm 200x200' => fn () => SL::expm($E),
    'scipy.special.erf 1M' => fn () => SP::erf($x),
    'scipy.special.gamma 1M' => fn () => SP::gamma($xp),
    'scipy.special.jv(1.5, x) 1M' => fn () => SP::jv(1.5, $x),
    'scipy.special.expit 1M' => fn () => SP::expit($xm5),
    'scipy.special.logsumexp 1M' => fn () => SP::logsumexp($x),
    'scipy.stats.norm.cdf 1M' => fn () => $norm->cdf($xm5),
    'scipy.stats.describe 1M' => fn () => ST::describe($x),
    'scipy.stats.zscore 1M' => fn () => ST::zscore($x),
    'scipy.stats.pearsonr 1M' => fn () => ST::pearsonr($x, $y),
    'scipy.stats.ttest_ind 100k vs 100k' => fn () => ST::ttestInd($ta, $tb),
    'scipy.signal.fftconvolve 100k * 1001' => fn () => SG::fftconvolve($sig, $ker),
    'scipy.signal.convolve 10k * 101' => fn () => SG::convolve($s10, $k101),
    'scipy.signal.lfilter order 2, 1M' => fn () => SG::lfilter([0.0675, 0.1349, 0.0675], [1.0, -1.143, 0.4128], $x),
    'scipy.spatial.distance.cdist 1000x1000x3' => fn () => D::cdist($XA, $XB, 'euclidean'),
    'scipy.spatial.distance.pdist 2000x3' => fn () => D::pdist($X2, 'euclidean'),
    'scipy.ndimage.gaussian_filter 1000x1000' => fn () => ND::gaussianFilter($img, 2.0),
    'scipy.ndimage.uniform_filter 1000x1000' => fn () => ND::uniformFilter($img, 5),
    'scipy.optimize.linprog 200x200 dense' => fn () => LP::linprog($lpC, $lpA, $lpB),
    'scipy.optimize.minimize rosen BFGS n=5' => fn () => MZ::minimize($rosen, [1.3, 0.7, 0.8, 1.9, 1.2]),
    'scipy.optimize.minimize_scalar brent' => fn () => MZ::minimizeScalar(fn (float $t): float => ($t - 2.0) ** 2),
    'scipy.optimize.brentq' => fn () => RT::brentq(fn (float $t): float => $t ** 3 - 2 * $t - 5, 2.0, 3.0),
];

$out = [];
foreach ($cases as $k => $f) {
    try {
        $small = str_contains($k, 'brent') || str_contains($k, 'minimize');
        $out[$k] = med($f, 11, $small ? 200 : 1);
    } catch (Throwable $e) {
        $out[$k] = 'ERR ' . get_class($e) . ': ' . substr($e->getMessage(), 0, 120);
    }
}
echo json_encode($out, JSON_PRETTY_PRINT), "\n";
