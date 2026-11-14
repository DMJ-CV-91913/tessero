<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;

/**
 * Empty inputs NumPy and SciPy answer without an error: inv of 0 x 0 matrices (alone or stacked) is empty, and the
 * complete QR of an m x 0 matrix has Q = the m x m identity and an empty R.
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
}
