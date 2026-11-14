<?php

declare(strict_types=1);

namespace Tessero\Tests\Unit;

use PHPUnit\Framework\TestCase;

/**
 * lu_factor and cho_factor return LAPACK's factor in column (Fortran) order, as SciPy does, so lu_solve and cho_solve
 * hand it to LAPACK without a copy. The values are those of the row-major factor; the solves give the same bytes
 * whether the factor comes in that order, as a C-ordered copy, or as float32.
 */
final class LinalgFactorLayoutTest extends TestCase
{
    /** @return list<array{class-string, class-string}> [NDArray, ScipyLinalg] of each backend */
    private static function backends(): array
    {
        $b = [[\Tessero\NDArray::class, \Tessero\ScipyLinalg::class]];
        if (extension_loaded('tessero') && class_exists(\Tessero\Ext\ScipyLinalg::class)) {
            $b[] = [\Tessero\Ext\NDArray::class, \Tessero\Ext\ScipyLinalg::class];
        }

        return $b;
    }

    /** an n x n matrix with a dominant diagonal, from + - * / only (the same values everywhere) */
    private static function matrix(string $array, int $n, bool $spd): object
    {
        $a = [];
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $n; $j++) {
                $v = (($i * 7 + $j * 3) % 11) / 10.0 - 0.5;
                $a[] = $i === $j ? $v + $n : ($spd ? (($i + $j) % 5) / 10.0 : $v);
            }
        }

        return $array::fromBytes(pack('e*', ...$a), 'float64', [$n, $n]);
    }

    public function testLuFactorIsColumnOrderedAndLuSolveReadsEveryLayout(): void
    {
        foreach (self::backends() as [$array, $scipy]) {
            foreach ([1, 2, 33, 70] as $n) {
                [$lu, $piv] = array_values($scipy::luFactor(self::matrix($array, $n, false)));
                $this->assertSame([$n, $n], $lu->shape(), "$scipy n=$n");
                if ($n > 1) {
                    $this->assertSame([8, 8 * $n], $lu->strides(), "$scipy n=$n: column order");
                }
                $c = $lu->copy();
                $this->assertSame($c->toBytes(), $lu->toBytes(), "$scipy n=$n: the values of the row-major factor");
                $b = $array::fromBytes(pack('e*', ...array_map(static fn (int $i): float => $i / 4.0 - 1.0, range(0, 3 * $n - 1))), 'float64', [$n, 3]);
                $x = $scipy::luSolve($lu, $piv, $b)->toBytes();
                $this->assertSame($x, $scipy::luSolve($c, $piv, $b)->toBytes(), "$scipy n=$n: a C-ordered factor");
                foreach ([1, 2] as $t) {
                    $this->assertSame($scipy::luSolve($c, $piv, $b, $t)->toBytes(), $scipy::luSolve($lu, $piv, $b, $t)->toBytes(), "$scipy n=$n trans=$t");
                }
            }
        }
    }

    public function testChoFactorIsColumnOrderedAndChoSolveReadsEveryLayout(): void
    {
        foreach (self::backends() as [$array, $scipy]) {
            foreach ([1, 2, 33, 70] as $n) {
                $A = self::matrix($array, $n, true);
                foreach ([false, true] as $lower) {
                    $c = array_values($scipy::choFactor($A, $lower))[0];
                    if ($n > 1) {
                        $this->assertSame([8, 8 * $n], $c->strides(), "$scipy n=$n: column order");
                    }
                    // the other triangle is the input's, as scipy leaves it
                    $cl = $c->toList();
                    $al = $A->toList();
                    for ($i = 0; $i < $n; $i++) {
                        for ($j = 0; $j < $n; $j++) {
                            if ($lower ? $j > $i : $j < $i) {
                                $this->assertSame($al[$i * $n + $j], $cl[$i * $n + $j], "$scipy n=$n: input kept at ($i, $j)");
                            }
                        }
                    }
                    $b = $array::fromBytes(pack('e*', ...array_map(static fn (int $i): float => $i / 8.0, range(0, $n - 1))), 'float64', [$n]);
                    $this->assertSame($scipy::choSolve($c->copy(), $lower, $b)->toBytes(), $scipy::choSolve($c, $lower, $b)->toBytes(), "$scipy n=$n: a C-ordered factor");
                }
            }
        }
    }
}
