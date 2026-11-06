<?php

/**
 * Tessero vs NumPy/SciPy, the broad tier of the speed sweep: about 80 heavier calls across NumPy, FFT, signal,
 * ndimage, interpolate, integrate, optimize, stats, linalg, sparse, csgraph, spatial and cluster, timed as a PHP
 * app sees them. bench/tier2_bench.py times the same calls on the same inputs (same seeds: the generators match
 * NumPy's PCG64 streams).
 *
 *   OPENBLAS_NUM_THREADS=1 php -d ffi.enable=1 -d memory_limit=-1 bench/tier2_bench.php > tessero.json
 */

declare(strict_types=1);
require __DIR__ . '/../tests/autoload.php';
use Tessero\Cluster as CL; use Tessero\Csgraph as CG; use Tessero\Fft\Fft; use Tessero\Hierarchy as HI;
use Tessero\Integrate\Quad; use Tessero\Integrate\SolveIvp; use Tessero\Interpolate as IP; use Tessero\Linalg as LA;
use Tessero\NDArray; use Tessero\Ndimage as ND; use Tessero\Np; use Tessero\Optimize\LeastSquares as LS;
use Tessero\Optimize\Minimize as MZ; use Tessero\Optimize\Root as RT; use Tessero\Random\Generator as G;
use Tessero\ScipyFft as SF; use Tessero\ScipyLinalg as SL; use Tessero\Signal as SG; use Tessero\Sparse as SP;
use Tessero\SparseLinalg as SPL; use Tessero\Spatial\ConvexHull; use Tessero\Spatial\Delaunay; use Tessero\Spatial\KDTree;
use Tessero\Stats as ST;

/** median of repeated runs in ms; fast calls are repeated in an inner loop of about 10 ms */
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

$r = fn (int $seed, array|int $size) => G::defaultRng($seed)->random(size: $size);
// inputs (seed, shape) mirror tier2_bench.py
$x = $r(10, 1000000)->mul(10.0);
$y = $r(11, 1000000);
$xi = $r(12, 1000000)->mul(1000.0)->floor();
$s100k = $r(13, 100000)->sort();
$t100k = $r(14, 100000);
$y100k = $r(15, 100000);
$k101 = $r(16, 101);
$s10k = $r(17, 10000);
$xp = NDArray::linspace(0.0, 1.0, 1000);
$fp = $r(18, 1000);
$C = $r(19, [50, 100000]);
$A = $r(20, [600, 600])->add(NDArray::eye(600)->mul(600.0));
$M3 = $r(21, [300, 300]);
$M4 = $r(22, [300, 300]);
$img1024 = $r(23, [1024, 1024]);
$img1k = $r(24, [1000, 1000]);
$img512 = $r(25, [512, 512]);
$img500 = $r(26, [500, 500]);
$k5x5 = $r(27, [5, 5]);
$bin1k = $img1k->gt(0.7);
$bin512 = $img512->gt(0.3);
$coords = $r(28, [2, 100000])->mul(999.0);
$xk = NDArray::linspace(0.0, 100.0, 10000);
$yk = $xk->sin();
$xnew = $r(29, 1000000)->mul(100.0);
$rbfY = $r(30, [500, 2]);
$rbfD = $rbfY->slice(':', 0)->mul(3.0)->sin();
$rbfX = $r(31, [10000, 2]);
$b8 = SG::butter(8, 0.1);
$sos = SG::tf2sos($b8['b'], $b8['a']);     // both sides: tf2sos(butter(8, 0.1)) (Tessero's butter returns b, a only)
$ba4 = SG::butter(4, 0.1);
[$bb, $ba] = [$ba4['b'], $ba4['a']];
$kde = $r(32, 1000);
$kpts = NDArray::linspace(-0.5, 1.5, 1000);
$ta = $r(33, 100000);
$tb = $r(34, 100000)->add(0.01);
$k10a = $r(35, 10000);
$k10b = $r(36, 10000);
$norm = ST::norm();
$Q = $r(37, [500, 500]);
$L1 = $r(38, [1000, 100]);
$L2 = $r(39, 1000);
$S1k = $r(40, [1000, 1000]);
$SPD = $S1k->matmul($S1k->transpose())->add(NDArray::eye(1000)->mul(1000.0));
$U = Np::triu($r(41, [1000, 1000])->add(NDArray::eye(1000)->mul(10.0)));
$ub = $r(42, 1000);
$n = 100000;
$tri = SP::diags([NDArray::full($n - 1, -1.0), NDArray::full($n, 4.0), NDArray::full($n - 1, -1.0)], [-1, 0, 1], [$n, $n]);
$tb100k = $r(43, $n);
// 2-D Poisson on a 100 x 100 grid: five diagonals, the +-1 ones cut at the grid rows
$off1 = NDArray::fromFlat(array_map(fn ($i) => ($i + 1) % 100 === 0 ? 0.0 : -1.0, range(0, 9998)), [9999]);
$poisson = SP::diags([NDArray::full(9900, -1.0), $off1, NDArray::full(10000, 4.0), $off1, NDArray::full(9900, -1.0)], [-100, -1, 0, 1, 100], [10000, 10000]);
$pb = $r(44, 10000);
$gw = $r(45, [500, 500]);
$graph = $gw->mul($gw->gt(0.9));
$pts3 = $r(46, [100000, 3])->toArray();
$qpts = $r(47, [100, 3])->toArray();
$tree = new KDTree($pts3);
$pts2 = $r(48, [10000, 2])->toArray();
$km = $r(49, [10000, 3]);
$kinit = $r(50, [8, 3]);
$lk = $r(51, [2000, 3]);
$rosen = function (array $v): float {
    $s = 0.0;
    for ($i = 0, $m = count($v) - 1; $i < $m; $i++) {
        $s += 100.0 * ($v[$i + 1] - $v[$i] ** 2) ** 2 + (1 - $v[$i]) ** 2;
    }

    return $s;
};
$x0_5 = [1.3, 0.7, 0.8, 1.9, 1.2];
$x0_50 = array_fill(0, 50, 0.5);
$cfx = NDArray::linspace(0.0, 4.0, 1000)->toList();
$cfy = array_map(fn ($t) => 2.5 * exp(-1.3 * $t) + 0.5 + 0.01 * sin(37.0 * $t), $cfx);
$teval = NDArray::linspace(0.0, 15.0, 100)->toList();

$cases = [
    'numpy cumsum 1M' => fn () => $x->cumsum(),
    'numpy argsort 1M' => fn () => $x->argsort(),
    'numpy unique 1M (1000 values)' => fn () => Np::unique($xi),
    'numpy searchsorted 1M into 100k' => fn () => Np::searchsorted($s100k, $y),
    'numpy percentile 1M' => fn () => Np::percentile($x, 90),
    'numpy median 1M' => fn () => Np::median($x),
    'numpy histogram 1M, 100 bins' => fn () => Np::histogram($x, 100),
    'numpy einsum ij,jk->ik 300x300' => fn () => Np::einsum('ij,jk->ik', $M3, $M4),
    'numpy polyfit deg 5, 100k' => fn () => Np::polyfit($t100k, $y100k, 5),
    'numpy convolve 100k * 101' => fn () => Np::convolve($t100k, $k101),
    'numpy interp 1M into 1000' => fn () => Np::interp($y, $xp, $fp),
    'numpy diff 1M' => fn () => $x->diff(),
    'numpy clip 1M' => fn () => $x->clip(2.0, 8.0),
    'numpy cov 50 x 100k' => fn () => Np::cov($C),
    'numpy linalg slogdet 600x600' => fn () => LA::slogdet($A),
    'numpy fft2 1024x1024' => fn () => Fft::fft2($img1024),
    'numpy rfft2 1024x1024' => fn () => Fft::rfft2($img1024),
    'scipy.fft dct 1M' => fn () => SF::dct($x),
    'scipy.signal sosfilt order 8, 1M' => fn () => SG::sosfilt($sos, $x),
    'scipy.signal filtfilt order 4, 1M' => fn () => SG::filtfilt($bb, $ba, $x),
    'scipy.signal welch 1M, nperseg 1024' => fn () => SG::welch($x, nperseg: 1024),
    'scipy.signal resample 100k -> 50k' => fn () => SG::resample($t100k, 50000),
    'scipy.signal resample_poly 1M, 3/2' => fn () => SG::resamplePoly($x, 3, 2),
    'scipy.signal medfilt 100k, k=5' => fn () => SG::medfilt($t100k, 5),
    'scipy.signal savgol_filter 1M (31, 3)' => fn () => SG::savgolFilter($x, 31, 3),
    'scipy.signal correlate 10k * 101' => fn () => SG::correlate($s10k, $k101),
    'scipy.signal hilbert 1M' => fn () => SG::hilbert($x),
    'scipy.signal find_peaks 1M' => fn () => SG::findPeaks($y),
    'scipy.signal detrend 1M' => fn () => SG::detrend($x),
    'scipy.signal decimate 1M by 10' => fn () => SG::decimate($x, 10),
    'scipy.signal convolve2d 500x500 * 5x5' => fn () => SG::convolve2d($img500, $k5x5),
    'scipy.ndimage median_filter 512x512, 3' => fn () => ND::medianFilter($img512, 3),
    'scipy.ndimage sobel 1000x1000' => fn () => ND::sobel($img1k),
    'scipy.ndimage laplace 1000x1000' => fn () => ND::laplace($img1k),
    'scipy.ndimage maximum_filter 1000x1000, 5' => fn () => ND::maximumFilter($img1k, 5),
    'scipy.ndimage label 1000x1000' => fn () => ND::label($bin1k),
    'scipy.ndimage binary_erosion 1000x1000' => fn () => ND::binaryErosion($bin1k),
    'scipy.ndimage distance_transform_edt 512x512' => fn () => ND::distanceTransformEdt($bin512),
    'scipy.ndimage zoom 500x500 x2' => fn () => ND::zoom($img500, 2.0),
    'scipy.ndimage rotate 512x512, 30 deg' => fn () => ND::rotate($img512, 30.0),
    'scipy.ndimage map_coordinates 1000x1000, 100k points' => fn () => ND::mapCoordinates($img1k, $coords),
    'scipy.interpolate CubicSpline 10k knots, 1M points' => fn () => IP::CubicSpline($xk, $yk, $xnew),
    'scipy.interpolate make_interp_spline k=3, 1M points' => fn () => IP::makeInterpSpline($xk, $yk, 3, $xnew),
    'scipy.interpolate PchipInterpolator 10k knots, 1M points' => fn () => IP::PchipInterpolator($xk, $yk, $xnew),
    'scipy.interpolate RBFInterpolator 500 -> 10k (2-D)' => fn () => IP::RBFInterpolator($rbfY, $rbfD, $rbfX),
    'scipy.integrate quad sin(x) exp(-x) on [0, 10]' => fn () => Quad::quad(fn (float $t): float => sin($t) * exp(-$t), 0.0, 10.0),
    'scipy.integrate solve_ivp Lotka-Volterra RK45' => fn () => SolveIvp::solve(
        fn (float $t, array $v): array => [1.5 * $v[0] - $v[0] * $v[1], -3.0 * $v[1] + $v[0] * $v[1]], [0.0, 15.0], [10.0, 5.0], $teval),
    'numpy trapezoid 1M' => fn () => Np::trapezoid($x),
    'scipy.optimize least_squares Rosenbrock residuals' => fn () => LS::leastSquares(fn (array $v): array => [10.0 * ($v[1] - $v[0] ** 2), 1.0 - $v[0]], [-1.2, 1.0]),
    'scipy.optimize curve_fit exp decay, 1000 points' => fn () => LS::curveFit(fn (float $t, float $a, float $b, float $c): float => $a * exp(-$b * $t) + $c, $cfx, $cfy, [1.0, 1.0, 1.0]),
    'scipy.optimize minimize Nelder-Mead rosen n=5' => fn () => MZ::minimize($rosen, $x0_5, method: 'Nelder-Mead'),
    'scipy.optimize minimize L-BFGS-B rosen n=50' => fn () => MZ::minimize($rosen, $x0_50, method: 'L-BFGS-B'),
    'scipy.optimize root (2 equations)' => fn () => RT::root(fn (array $v): array => [$v[0] + 0.5 * ($v[0] - $v[1]) ** 3 - 1.0, 0.5 * ($v[1] - $v[0]) ** 3 + $v[1]], [0.0, 0.0]),
    'scipy.stats spearmanr 100k' => fn () => ST::spearmanr($ta, $tb),
    'scipy.stats kendalltau 10k' => fn () => ST::kendalltau($k10a, $k10b),
    'scipy.stats mannwhitneyu 100k vs 100k' => fn () => ST::mannwhitneyu($ta, $tb),
    'scipy.stats gaussian_kde 1000 points, 1000 evaluations' => fn () => ST::gaussianKde($kde, $kpts),
    'scipy.stats rankdata 1M' => fn () => ST::rankdata($x),
    'scipy.stats ks_2samp 100k vs 100k' => fn () => ST::ks2samp($ta, $tb),
    'scipy.stats skew 1M' => fn () => ST::skew($x),
    'scipy.stats kurtosis 1M' => fn () => ST::kurtosis($x),
    'scipy.stats linregress 1M' => fn () => ST::linregress($x, $y),
    'scipy.stats binned_statistic 1M, 100 bins' => fn () => ST::binnedStatistic($x, $y, 'mean', 100),
    'scipy.stats norm.ppf 1M' => fn () => $norm->ppf($y),
    'scipy.linalg qr 500x500' => fn () => SL::qr($Q),
    'scipy.linalg lstsq 1000x100' => fn () => SL::lstsq($L1, $L2),
    'scipy.linalg cholesky 1000x1000' => fn () => SL::cholesky($SPD),
    'scipy.linalg eig 300x300 (general)' => fn () => SL::eig($M3),
    'scipy.linalg schur 300x300' => fn () => SL::schur($M3),
    'scipy.linalg pinv 300x300' => fn () => SL::pinv($M4),
    'scipy.linalg solve_triangular 1000x1000' => fn () => SL::solveTriangular($U, $ub),
    'scipy.linalg svdvals 300x300' => fn () => SL::svdvals($M4),
    'scipy.sparse.linalg spsolve tridiagonal 100k' => fn () => SPL::spsolve($tri, $tb100k),
    'scipy.sparse.linalg cg 2-D Poisson 100x100' => fn () => SPL::cg($poisson, $pb),
    'scipy.sparse.linalg eigsh k=6, 2-D Poisson 100x100' => fn () => SPL::eigsh($poisson, 6),
    'scipy.sparse.csgraph shortest_path 500 nodes (Dijkstra)' => fn () => CG::shortestPath($graph, 'D'),
    'scipy.sparse.csgraph minimum_spanning_tree 500 nodes' => fn () => CG::minimumSpanningTree($graph),
    'scipy.sparse.csgraph connected_components 500 nodes' => fn () => CG::connectedComponents($graph),
    'scipy.spatial KDTree build 100k (3-D)' => fn () => new KDTree($pts3),
    'scipy.spatial KDTree query 100 points, k=5 (Tessero: exhaustive search)' => fn () => $tree->query($qpts, 5),
    'scipy.spatial ConvexHull 10k (2-D)' => fn () => new ConvexHull($pts2),
    'scipy.spatial Delaunay 10k (2-D)' => fn () => new Delaunay($pts2),
    'scipy.cluster.vq kmeans2 10k x 3, k=8' => fn () => CL::kmeans2($km, $kinit),
    'scipy.cluster.hierarchy linkage ward 2000x3' => fn () => HI::linkage($lk, 'ward'),
];

$only = $argv[1] ?? null;
$out = [];
foreach ($cases as $k => $f) {
    if ($only !== null && ! str_contains($k, $only)) {
        continue;
    }
    $t0 = hrtime(true);
    fwrite(STDERR, "start {$k}
");
    try {
        $out[$k] = getenv('DRY') ? ($f() === null ? 0.0 : 0.0) : med($f);   /* DRY=1: call once, report errors only */
    } catch (Throwable $e) {
        $out[$k] = 'ERR ' . get_class($e) . ': ' . substr($e->getMessage(), 0, 160);
    }
    fwrite(STDERR, sprintf("  %.1f s
", (hrtime(true) - $t0) / 1e9));
}
echo json_encode($out, JSON_PRETTY_PRINT), "\n";
