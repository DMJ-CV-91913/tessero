<?php

declare(strict_types=1);

namespace Tessero\Optimize;

use Tessero\Exceptions\SingularMatrix;
use Tessero\Linalg\Linalg;
use Tessero\NDArray;
use Tessero\Native\Blas;

/** @internal small dense helpers for the optimisers (PHP arrays in, PHP arrays out) */
final class Dense
{
    /**
     * Solve A x = b; LAPACK when a BLAS is loaded, otherwise Gaussian elimination with partial pivoting.
     *
     * @param list<list<float>> $A
     * @param list<float> $b
     * @return list<float>
     */
    public static function solve(array $A, array $b): array
    {
        $n = count($b);
        if ($n > 8 && Blas::available()) {
            return Linalg::solve(NDArray::array($A, 'float64'), NDArray::array($b, 'float64'))->toList();
        }
        for ($i = 0; $i < $n; $i++) {
            $A[$i][] = $b[$i];
        }
        for ($c = 0; $c < $n; $c++) {
            $p = $c;
            for ($r = $c + 1; $r < $n; $r++) {
                if (abs($A[$r][$c]) > abs($A[$p][$c])) {
                    $p = $r;
                }
            }
            if ($A[$p][$c] == 0.0) {
                throw new SingularMatrix('Singular Jacobian / normal matrix.');
            }
            if ($p !== $c) {
                [$A[$p], $A[$c]] = [$A[$c], $A[$p]];
            }
            $pivot = $A[$c][$c];
            for ($r = $c + 1; $r < $n; $r++) {
                $factor = $A[$r][$c] / $pivot;
                if ($factor == 0.0) {
                    continue;
                }
                $row = $A[$r];
                $src = $A[$c];
                for ($k = $c; $k <= $n; $k++) {
                    $row[$k] = $row[$k] - $factor * $src[$k];
                }
                $A[$r] = $row;
            }
        }
        $x = array_fill(0, $n, 0.0);
        for ($i = $n - 1; $i >= 0; $i--) {
            $s = $A[$i][$n];
            for ($k = $i + 1; $k < $n; $k++) {
                $s -= $A[$i][$k] * $x[$k];
            }
            $x[$i] = $s / $A[$i][$i];
        }

        return $x;
    }

    /** @param list<list<float>> $A @return list<list<float>> */
    public static function inverse(array $A): array
    {
        $n = count($A);
        $cols = [];
        for ($j = 0; $j < $n; $j++) {
            $e = array_fill(0, $n, 0.0);
            $e[$j] = 1.0;
            $cols[] = self::solve($A, $e);
        }
        $inv = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $n; $j++) {
                $row[] = $cols[$j][$i];
            }
            $inv[] = $row;
        }

        return $inv;
    }
}
