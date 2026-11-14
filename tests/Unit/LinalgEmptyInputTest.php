<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;
use Tessero\Exceptions\ShapeError;
use Tessero\Linalg\Linalg as LegacyLinalg;
use Tessero\NDArray;

/**
 * Empty inputs NumPy and SciPy answer without an error: inv of 0 x 0 matrices (alone or stacked) is empty, the
 * complete QR of an m x 0 matrix has Q = the m x m identity and an empty R, and a stack holding no matrices gives
 * empty results of the right shape.
 */
final class LinalgEmptyInputTest extends TestCase
{
    /** @return list<array{class-string, class-string, class-string}> [NDArray, Linalg, ScipyLinalg] of each backend */
    private static function backends(): array
    {
        $b = [[\Tessero\NDArray::class, \Tessero\Linalg::class, \Tessero\ScipyLinalg::class]];
        if (extension_loaded('tessero') && class_exists(\Tessero\Ext\Linalg::class)) {
            $b[] = [\Tessero\Ext\NDArray::class, \Tessero\Ext\Linalg::class, \Tessero\Ext\ScipyLinalg::class];
        }

        return $b;
    }

    public function testInvOfEmptyMatricesIsEmpty(): void
    {
        foreach (self::backends() as [$array, $linalg, $scipy]) {
            $this->assertSame([0, 0], $linalg::inv($array::zeros([0, 0]))->shape(), "$linalg 0x0");
            $this->assertSame([0, 0], $scipy::inv($array::zeros([0, 0]))->shape(), "$scipy 0x0");
            $this->assertSame([2, 0, 0], $linalg::inv($array::zeros([2, 0, 0]))->shape(), "$linalg 2x0x0");
            $this->assertSame([0, 4, 4], $linalg::inv($array::zeros([0, 4, 4]))->shape(), "$linalg 0x4x4");
        }
    }

    public function testCompleteQrOfAnMBy0MatrixHasTheIdentityAsQ(): void
    {
        $eye = pack('e*', 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        foreach (self::backends() as [$array, $linalg, $scipy]) {
            foreach ([[$linalg, 'complete'], [$scipy, 'full']] as [$cls, $mode]) {
                $qr = array_values($cls::qr($array::zeros([3, 0]), $mode));
                $this->assertSame([3, 3], $qr[0]->shape(), "$cls $mode: Q shape");
                $this->assertSame($eye, $qr[0]->toBytes(), "$cls $mode: Q");
                $this->assertSame([3, 0], $qr[1]->shape(), "$cls $mode: R shape");
            }
            $qr = array_values($linalg::qr($array::zeros([3, 0])));
            $this->assertSame([3, 0], $qr[0]->shape(), "$linalg reduced: Q shape");
            $this->assertSame([0, 0], $qr[1]->shape(), "$linalg reduced: R shape");
        }
    }

    public function testTheLinalgClassTakesAStackOfNoMatrices(): void
    {
        // Tessero\Linalg\Linalg (PHP over LAPACKE): NumPy's shapes for a stack with no matrices in it
        $A = NDArray::zeros([0, 3, 3]);
        $this->assertSame([0, 3, 3], LegacyLinalg::inv($A)->shape());
        $this->assertSame([0, 3, 3], LegacyLinalg::cholesky($A)->shape());
        $this->assertSame([0, 3], LegacyLinalg::solve($A, NDArray::zeros([3]))->shape());
        $this->assertSame([0, 3, 2], LegacyLinalg::solve($A, NDArray::zeros([3, 2]))->shape());
        $this->assertSame([0, 3, 2], LegacyLinalg::solve($A, NDArray::zeros([0, 3, 2]))->shape());
        [$sign, $logdet] = LegacyLinalg::slogdet($A);
        $this->assertSame([0], $sign->shape());
        $this->assertSame([0], $logdet->shape());
        $this->assertSame([0], LegacyLinalg::det($A)->shape());
        [$w, $v] = LegacyLinalg::eigh($A);
        $this->assertSame([0, 3], $w->shape());
        $this->assertSame([0, 3, 3], $v->shape());
        // b with the wrong number of rows is refused even when there is nothing to solve
        $this->expectException(ShapeError::class);
        LegacyLinalg::solve($A, NDArray::zeros([4]));
    }
}
