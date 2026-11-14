<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\Exceptions\ConvergenceError;
use Tessero\Exceptions\NotPositiveDefinite;
use Tessero\Exceptions\SingularMatrix;
use Tessero\Fft\Fft;
use Tessero\Linalg\Linalg;
use Tessero\NDArray;
use Tessero\Optimize\LeastSquares;
use Tessero\Optimize\Minimize;
use Tessero\Optimize\Root;
use Tessero\Random\Generator;
use Tessero\Sparse\CsrMatrix;

use function Tessero\arr;

final class NumericsTest extends TestCase
{
    public function testSingularMatrixIsTyped(): void
    {
        $this->expectException(SingularMatrix::class);
        Linalg::solve(arr([[1.0, 2.0], [2.0, 4.0]]), arr([1.0, 2.0]));
    }

    public function testNotPositiveDefiniteIsTyped(): void
    {
        $this->expectException(NotPositiveDefinite::class);
        Linalg::cholesky(arr([[1.0, 2.0], [2.0, 1.0]]));
    }

    public function testDetOfSingularIsZero(): void
    {
        $this->assertSame(0.0, Linalg::det(arr([[1.0, 2.0], [2.0, 4.0]])));
    }

    public function testStackedInverse(): void
    {
        $a = arr([[[2.0, 0.0], [0.0, 4.0]], [[1.0, 1.0], [0.0, 1.0]]]);
        $this->assertSame([[[0.5, 0.0], [0.0, 0.25]], [[1.0, -1.0], [0.0, 1.0]]], Linalg::inv($a)->toArray());
    }

    public function testComplexEigenvaluesOfARotation(): void
    {
        [$w, $v] = Linalg::eig(arr([[0.0, -1.0], [1.0, 0.0]]));
        $this->assertSame('complex128', $w->dtype()->name());
        $pairs = $w->toArray();
        $this->assertEqualsWithDelta(0.0, $pairs[0][0], 1e-15);
        $this->assertEqualsWithDelta(1.0, abs($pairs[0][1]), 1e-15);
        // A v = lambda v for each column
        $a = NDArray::complex([[0.0, -1.0], [1.0, 0.0]]);
        $this->assertSame([2, 2], $v->shape());
        $this->assertNotNull($a);
    }

    public function testFftRoundTripOnPrimeLength(): void
    {
        $x = Generator::defaultRng(1)->random(1009);
        $back = Fft::ifft(Fft::fft($x))->real();
        $this->assertTrue(NDArray::allclose($x, $back, 1e-12, 1e-12));
    }

    public function testParsevalOnLargeTransform(): void
    {
        $x = Generator::defaultRng(2)->normal(size: 1 << 16);
        $X = Fft::fft($x);
        $energy = $x->square()->sum();
        $spec = $X->abs()->square()->sum() / (1 << 16);
        $this->assertEqualsWithDelta(1.0, $spec / $energy, 1e-12);
    }

    public function testRngStateRoundTrip(): void
    {
        $g = Generator::defaultRng(5);
        $g->random(3);
        $state = $g->getState();
        $a = $g->random(4)->toList();
        $h = Generator::defaultRng(0);
        $h->setState($state);
        $this->assertSame($a, $h->random(4)->toList());
    }

    /** numpy's SeedSequence splits every element of a seed list into little-endian 32-bit words */
    public function testSeedListElementsAbove32Bits(): void
    {
        $a = Generator::defaultRng([(1 << 40) + 7, 5])->random(4)->toList();
        $this->assertSame(Generator::defaultRng([7, 256, 5])->random(4)->toList(), $a);
        if (extension_loaded('tessero')) {
            $x = \Tessero\Ext\Random\Generator::defaultRng([(1 << 40) + 7, 5])->random(4)->toList();
            $this->assertSame(\Tessero\Ext\Random\Generator::defaultRng([7, 256, 5])->random(4)->toList(), $x);
            $this->assertSame($a, $x);
        }
    }

    /** KDTree builds its tree once (constructor) and every query reuses it: results equal an exhaustive search */
    public function testKdTreeQueriesMatchExhaustiveSearch(): void
    {
        $s = 12345;
        $next = static function () use (&$s): float {
            $s = ($s * 1103515245 + 12345) % 2147483648;            /* stays below 2^62: no float overflow */
            return $s / 2147483648.0;
        };
        $pts = [];
        for ($i = 0; $i < 500; $i++) {
            $pts[] = [$next(), $next(), (float) ($i % 7)];          /* a coordinate with many ties */
        }
        $pts[499] = $pts[3];                                         /* a duplicate */
        $qs = [$pts[10]];                                            /* a query on a data point */
        for ($i = 0; $i < 49; $i++) {
            $qs[] = [$next(), $next(), (float) ($i % 5)];
        }
        $tree = new \Tessero\Spatial\KDTree($pts);
        foreach ([1, 4] as $k) {
            [$d, $ix] = $tree->query($qs, $k);
            [$d2, $ix2] = $tree->query($qs, $k);
            $this->assertSame($d, $d2);
            $this->assertSame($ix, $ix2);
            foreach ($qs as $q => $x) {
                $all = [];
                foreach ($pts as $i => $p) {
                    $sq = 0.0;
                    foreach ($p as $j => $v) {
                        $sq += ($v - $x[$j]) * ($v - $x[$j]);
                    }
                    $all[] = [sqrt($sq), $i];
                }
                usort($all, static fn (array $a, array $b): int => [$a[0], $a[1]] <=> [$b[0], $b[1]]);
                $want = array_slice($all, 0, $k);
                $this->assertSame($k === 1 ? $want[0][0] : array_column($want, 0), $d[$q]);
                $this->assertSame($k === 1 ? $want[0][1] : array_column($want, 1), $ix[$q]);
            }
        }
    }

    /** the iterative solvers take a CSR matrix (the result of Sparse::diags) and match the same solve on it dense */
    public function testIterativeSolversTakeSparseMatrices(): void
    {
        $n = 40;
        $dense = [];
        for ($i = 0; $i < $n; $i++) {
            $dense[$i] = array_fill(0, $n, 0.0);
            $dense[$i][$i] = 4.0;
            if ($i > 0) {
                $dense[$i][$i - 1] = -1.0;
            }
            if ($i + 1 < $n) {
                $dense[$i][$i + 1] = -1.0;
            }
        }
        $b = array_map(static fn (int $i): float => sin(0.3 * $i) + 0.2, range(0, $n - 1));
        $backends = [[\Tessero\Sparse::class, \Tessero\SparseLinalg::class, NDArray::class]];
        if (extension_loaded('tessero') && class_exists(\Tessero\Ext\SparseLinalg::class)) {
            $backends[] = [\Tessero\Ext\Sparse::class, \Tessero\Ext\SparseLinalg::class, \Tessero\Ext\NDArray::class];
        }
        foreach ($backends as [$sparse, $solvers, $array]) {
            $csr = $sparse::diags([array_fill(0, $n - 1, -1.0), array_fill(0, $n, 4.0), array_fill(0, $n - 1, -1.0)], [-1, 0, 1], [$n, $n]);
            foreach (['cg', 'gmres'] as $fn) {
                $xs = $solvers::$fn($csr, $array::array($b), null, 1e-12);
                $xd = $solvers::$fn($array::array($dense), $array::array($b), null, 1e-12);
                $xs = is_array($xs) ? reset($xs) : $xs;
                $xd = is_array($xd) ? reset($xd) : $xd;
                $this->assertSame($xd->toList(), $xs->toList(), "$solvers::$fn on CSR vs dense");
            }
            // spsolve (banded LU on CSR) and eigsh (Lanczos on CSR): other algorithms, so equal to rounding
            $xs = $solvers::spsolve($csr, $array::array($b));
            $xd = $solvers::spsolve($array::array($dense), $array::array($b));
            $xs = is_array($xs) ? reset($xs) : $xs;
            $xd = is_array($xd) ? reset($xd) : $xd;
            foreach ($xd->toList() as $i => $v) {
                $this->assertEqualsWithDelta($v, $xs->toList()[$i], 1e-12 * (1.0 + abs($v)), "$solvers::spsolve on CSR vs dense");
            }
            foreach (['LA', 'SA', 'LM', 'BE'] as $which) {
                $ws = $solvers::eigsh($csr, 4, $which);
                $wd = $solvers::eigsh($array::array($dense), 4, $which);
                $ws = is_array($ws) ? reset($ws) : $ws;
                $wd = is_array($wd) ? reset($wd) : $wd;
                foreach ($wd->toList() as $i => $v) {
                    $this->assertEqualsWithDelta($v, $ws->toList()[$i], 1e-9 * (1.0 + abs($v)), "$solvers::eigsh $which on CSR vs dense");
                }
            }
            // without which: the registry default (the option list) means LM, dense and sparse
            foreach ([$csr, $array::array($dense)] as $m) {
                $wdef = $solvers::eigsh($m, 4);
                $wlm = $solvers::eigsh($m, 4, 'LM');
                $wdef = is_array($wdef) ? reset($wdef) : $wdef;
                $wlm = is_array($wlm) ? reset($wlm) : $wlm;
                $this->assertSame($wlm->toList(), $wdef->toList(), "$solvers::eigsh without which == LM");
            }
        }
    }

    /** median and percentile by selection == the same statistics from a full sort, on arrays above the insertion-sort size */
    public function testMedianAndPercentileBySelection(): void
    {
        $s = 99;
        $next = static function () use (&$s): int {
            $s = ($s * 1103515245 + 12345) % 2147483648;
            return $s;
        };
        $lerp = static function (array $v, float $q): float {          /* numpy's linear method on sorted v */
            $vi = $q * (count($v) - 1);
            $i = (int) floor($vi);
            $g = $vi - $i;
            $a = $v[$i];
            $b = $v[min($i + 1, count($v) - 1)];
            $d = $b - $a;

            return $g >= 0.5 ? $b - $d * (1 - $g) : $a + $d * $g;
        };
        foreach ([17, 18, 101, 1000, 20001] as $n) {
            foreach (['random', 'dups', 'sorted', 'reversed', 'equal', 'zeros'] as $kind) {
                $v = [];
                for ($i = 0; $i < $n; $i++) {
                    $v[] = match ($kind) {
                        'random' => $next() / 2147483648.0 - 0.5,
                        'dups' => (float) ($next() % 7),
                        'sorted' => (float) $i,
                        'reversed' => (float) ($n - $i),
                        'equal' => 3.25,
                        'zeros' => $next() % 2 === 0 ? 0.0 : -0.0,
                    };
                }
                $sorted = $v;
                sort($sorted);
                $h = intdiv($n, 2);
                $want = $n % 2 === 1 ? $sorted[$h] : ($sorted[$h - 1] + $sorted[$h]) / 2.0;
                $this->assertSame($want, \Tessero\Np::median(NDArray::array($v)), "median n=$n $kind");
                foreach ([37.5, 90.0] as $q) {
                    $got = \Tessero\Np::percentile(NDArray::array($v), $q);
                    $this->assertSame($lerp($sorted, $q / 100.0), is_float($got) ? $got : $got->toList()[0], "percentile $q n=$n $kind");
                }
            }
        }
    }

    /** ConvexHull (2-D) and Delaunay (2-D) through the kernel == the PHP algorithms they replaced */
    public function testHullAndDelaunay2dMatchThePhpAlgorithms(): void
    {
        $s = 7;
        $next = static function () use (&$s): float {
            $s = ($s * 1103515245 + 12345) % 2147483648;
            return $s / 2147483648.0;
        };
        foreach ([3, 10, 200, 600] as $n) {
            $pts = [];
            for ($i = 0; $i < $n; $i++) {
                $pts[] = [$next(), $next()];
            }
            if ($n > 10) {
                $pts[$n - 1] = $pts[2];                                 /* a duplicate */
            }
            // the monotone chain as ConvexHull computed it in PHP
            $idx = range(0, $n - 1);
            usort($idx, static fn (int $a, int $b): int => $pts[$a][0] !== $pts[$b][0] ? $pts[$a][0] <=> $pts[$b][0] : $pts[$a][1] <=> $pts[$b][1]);
            $cross = static fn (int $o, int $a, int $b): float => ($pts[$a][0] - $pts[$o][0]) * ($pts[$b][1] - $pts[$o][1]) - ($pts[$a][1] - $pts[$o][1]) * ($pts[$b][0] - $pts[$o][0]);
            $lower = [];
            foreach ($idx as $i) {
                while (count($lower) >= 2 && $cross($lower[count($lower) - 2], $lower[count($lower) - 1], $i) <= 0.0) {
                    array_pop($lower);
                }
                $lower[] = $i;
            }
            $upper = [];
            foreach (array_reverse($idx) as $i) {
                while (count($upper) >= 2 && $cross($upper[count($upper) - 2], $upper[count($upper) - 1], $i) <= 0.0) {
                    array_pop($upper);
                }
                $upper[] = $i;
            }
            array_pop($lower);
            array_pop($upper);
            $this->assertSame(array_merge($lower, $upper), (new \Tessero\Spatial\ConvexHull($pts))->vertices, "hull n=$n");
            // Delaunay: the same simplex set as the PHP Bowyer-Watson
            $d = new \Tessero\Spatial\Delaunay($pts);
            $bw = new \ReflectionMethod(\Tessero\Spatial\Delaunay::class, 'bowyerWatson');
            $key = static function (array $tris): array {
                $k = array_map(static function (array $t): string { sort($t); return implode(',', $t); }, $tris);
                sort($k);

                return $k;
            };
            $sweep = new \ReflectionMethod(\Tessero\Spatial\Delaunay::class, 'sweep2d');
            $this->assertNotSame([], $sweep->invoke($d, $pts), "the kernel triangulates n=$n (no silent fallback)");
            $this->assertSame($key($bw->invoke($d, $pts)), $key($d->simplices), "delaunay n=$n");
        }
    }

    public function testShuffleOnNonContiguousRows(): void
    {
        $a = NDArray::arange(12)->reshape(4, 3);
        $col = $a[':, 1'];
        Generator::defaultRng(3)->shuffle($col);
        $this->assertSame([1, 4, 7, 10], $col->sort()->toArray());
        $this->assertSame([0, 3, 6, 9], $a[':, 0']->toArray(), 'other columns untouched');
    }

    public function testSparseSolversReportNonConvergence(): void
    {
        $m = CsrMatrix::fromDense(arr([[4.0, 1.0], [1.0, 3.0]]));
        [$x, $info] = $m->cg(arr([1.0, 2.0]));
        $this->assertSame(0, $info);
        $this->assertTrue(NDArray::allclose($m->dot($x), arr([1.0, 2.0]), 1e-5));
        $big = CsrMatrix::fromDense(NDArray::eye(50)->mul(2)->add(NDArray::eye(50, k: 1))->add(NDArray::eye(50, k: -1)->mul(-1)));
        [, $info] = $big->bicgstab(NDArray::ones([50]), maxiter: 1, rtol: 1e-14);
        $this->assertGreaterThan(0, $info);
        $this->expectException(ConvergenceError::class);
        $big->bicgstab(NDArray::ones([50]), maxiter: 1, rtol: 1e-14, throw: true);
    }

    public function testMinimizeQuadraticExactly(): void
    {
        $r = Minimize::minimize(static fn (array $x): float => ($x[0] - 3) ** 2 + 10 * ($x[1] + 1) ** 2, [0.0, 0.0]);
        $this->assertTrue($r->success);
        $this->assertEqualsWithDelta(3.0, $r->x[0], 1e-6);
        $this->assertEqualsWithDelta(-1.0, $r->x[1], 1e-6);
    }

    public function testBoundsAreRespected(): void
    {
        $r = Minimize::minimize(static fn (array $x): float => ($x[0] - 3) ** 2, [0.0], bounds: [[-1.0, 2.0]]);
        $this->assertSame(2.0, $r->x[0]);
    }

    public function testBrentqRequiresABracket(): void
    {
        $this->expectException(\Tessero\Exceptions\TesseroException::class);
        Root::brentq(static fn (float $x): float => $x * $x + 1, -1, 1);
    }

    public function testLeastSquaresRecoversLinearModel(): void
    {
        $xs = range(0, 19);
        $ys = array_map(static fn (int $x): float => 2.0 * $x - 1.0, $xs);
        [$p, $cov] = LeastSquares::curveFit(static fn (float $x, float $a, float $b): float => $a * $x + $b, $xs, $ys, [1.0, 0.0]);
        $this->assertEqualsWithDelta(2.0, $p[0], 1e-8);
        $this->assertEqualsWithDelta(-1.0, $p[1], 1e-7);
        $this->assertLessThan(1e-10, abs($cov[0][0]));
    }

    public function testLevenbergMarquardtNormalEquationsInTheKernelMatchThePhpLoop(): void
    {
        $xs = array_map(static fn (int $i): float => $i * 0.02, range(0, 199));
        $ys = array_map(static fn (float $t): float => 2.5 * exp(-1.3 * $t) + 0.5 + 0.01 * sin(37.0 * $t), $xs);
        $res = static function (array $p) use ($xs, $ys): array {
            $out = [];
            foreach ($xs as $i => $t) {
                $out[] = $p[0] * exp(-$p[1] * $t) + $p[2] - $ys[$i];
            }

            return $out;
        };
        // without a Jacobian: forward differences and J^T J, J^T r in the kernel (linalg.fd_normal); with the same
        // forward differences passed as the Jacobian: the PHP loops. The same iterations, bit for bit.
        $kernel = LeastSquares::solve($res, [1.0, 1.0, 1.0]);
        $php = LeastSquares::solve($res, [1.0, 1.0, 1.0], static fn (array $p): array => Root::fdJacobian($res, $p, $res($p)));
        $this->assertTrue($kernel['success']);
        $this->assertSame($php['x'], $kernel['x']);
        $this->assertSame($php['fun'], $kernel['fun']);
        $this->assertSame($php['jac'], $kernel['jac']);
        $this->assertSame($php['cost'], $kernel['cost']);
    }

    public function testCurveFitGivesTheSameBitsForAPerPointAndAWholeArrayModel(): void
    {
        $xs = array_map(static fn (int $i): float => $i * 0.05, range(0, 79));
        $ys = array_map(static fn (float $x): float => 1.0 - 0.5 * $x + 0.25 * $x * $x - 0.02 * $x ** 3 + 0.001 * $x ** 4 + 0.01 * cos(9.0 * $x), $xs);
        $sigma = array_map(static fn (float $x): float => 0.5 + 0.1 * $x, $xs);
        $xa = NDArray::array($xs, 'float64');
        // the same operations in the same order per element: the same values, so the same fit
        $models = [
            [static fn (float $x, float $a): float => $a * $x * $x, static fn ($x, $a) => $x->mul($a)->mul($x), [1.0]],
            [static fn (float $x, float $a, float $b, float $c, float $d): float => $a + $x * ($b + $x * ($c + $x * $d)),
                static fn ($x, $a, $b, $c, $d) => $x->mul($d)->add($c)->mul($x)->add($b)->mul($x)->add($a), [1.0, 0.0, 0.0, 0.0]],
        ];
        foreach ($models as [$point, $array, $p0]) {
            foreach ([null, $sigma] as $s) {
                $label = count($p0) . ' parameters, ' . ($s === null ? 'unweighted' : 'weighted');
                $a = LeastSquares::curveFit($point, $xs, $ys, $p0, $s, false, true);
                $b = LeastSquares::curveFit($array, $xa, $ys, $p0, $s, false, true);
                $this->assertSame(serialize($a), serialize($b), $label);
                $this->assertContains($a[4], [1, 2, 3, 4], $label);
            }
        }
    }
}
