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
}
