<?php

declare(strict_types=1);

namespace Tessero;

use Tessero\Exceptions\TesseroException;

/**
 * scipy.special sequence/derivative helpers that return a whole family of values at once:
 * the Legendre polynomial with its derivatives ({@see legendreP}, {@see legendrePAll}), the
 * Legendre function of the second kind and its derivative ({@see lqn}), and the Riccati-Bessel
 * functions and their derivatives ({@see riccatiJn}, {@see riccatiYn}).
 *
 * Each is a pure-PHP recurrence (stable: the regular Riccati-Bessel uses downward/Miller recurrence,
 * the rest use their natural stable direction), reproducing scipy to ~1e-10 on both backends with
 * no kernel.
 */
final class SpecialSeq
{
    /**
     * Legendre polynomial P_n(x) and its derivatives up to order $diffN (0, 1 or 2).
     *
     * @return list<float> [P_n, P_n', ...] of length $diffN + 1
     */
    public static function legendreP(int $n, float $x, int $diffN = 0): array
    {
        if ($n < 0) {
            throw new TesseroException('n must be nonnegative.');
        }

        return self::legDerivs($n, $x, $diffN);
    }

    /**
     * P_k(x) and derivatives for every degree k = 0..n.
     *
     * @return list<list<float>> rows indexed by derivative order 0..$diffN, columns by degree 0..$n
     */
    public static function legendrePAll(int $n, float $x, int $diffN = 0): array
    {
        if ($n < 0) {
            throw new TesseroException('n must be nonnegative.');
        }
        $perDegree = [];
        for ($k = 0; $k <= $n; $k++) {
            $perDegree[$k] = self::legDerivs($k, $x, $diffN);
        }
        $rows = [];
        for ($d = 0; $d <= $diffN; $d++) {
            $row = [];
            for ($k = 0; $k <= $n; $k++) {
                $row[$k] = $perDegree[$k][$d];
            }
            $rows[$d] = $row;
        }

        return $rows;
    }

    /**
     * Legendre function of the second kind Q_k(x) and its derivative for k = 0..n (|x| < 1).
     *
     * @return array{0: list<float>, 1: list<float>} [Q_0..Q_n, Q'_0..Q'_n]
     */
    public static function lqn(int $n, float $x): array
    {
        if ($n < 0) {
            throw new TesseroException('n must be nonnegative.');
        }
        $Q = array_fill(0, $n + 1, 0.0);
        $D = array_fill(0, $n + 1, 0.0);
        $Q[0] = atanh($x);
        if ($n >= 1) {
            $Q[1] = $x * atanh($x) - 1.0;
        }
        for ($k = 1; $k < $n; $k++) {
            $Q[$k + 1] = ((2 * $k + 1) * $x * $Q[$k] - $k * $Q[$k - 1]) / ($k + 1);
        }
        $D[0] = 1.0 / (1.0 - $x * $x);
        for ($k = 1; $k <= $n; $k++) {
            $D[$k] = $k * ($x * $Q[$k] - $Q[$k - 1]) / ($x * $x - 1.0);
        }

        return [$Q, $D];
    }

    /**
     * Riccati-Bessel function of the first kind psi_k(x) = x * j_k(x) and its derivative, k = 0..n.
     *
     * @return array{0: list<float>, 1: list<float>}
     */
    public static function riccatiJn(int $n, float $x): array
    {
        if ($n < 0) {
            throw new TesseroException('n must be nonnegative.');
        }
        $S = array_fill(0, $n + 1, 0.0);
        $S[0] = sin($x);
        if ($n >= 1) {
            // Miller's downward recurrence for the regular solution, then normalise by psi_0 = sin(x).
            $m = $n + max(15, (int) round(2.0 * sqrt($n * abs($x))) + 10);
            $f = array_fill(0, $m + 2, 0.0);
            $f[$m] = 1e-30;
            for ($k = $m; $k >= 1; $k--) {
                $f[$k - 1] = (2 * $k + 1) / $x * $f[$k] - $f[$k + 1];
            }
            $scale = sin($x) / $f[0];
            for ($k = 1; $k <= $n; $k++) {
                $S[$k] = $f[$k] * $scale;
            }
        }
        $D = array_fill(0, $n + 1, 0.0);
        $D[0] = cos($x);
        for ($k = 1; $k <= $n; $k++) {
            $D[$k] = $S[$k - 1] - $k / $x * $S[$k];
        }

        return [$S, $D];
    }

    /**
     * Riccati-Bessel function of the second kind chi_k(x) = -x * y_k(x) and its derivative, k = 0..n.
     *
     * @return array{0: list<float>, 1: list<float>}
     */
    public static function riccatiYn(int $n, float $x): array
    {
        if ($n < 0) {
            throw new TesseroException('n must be nonnegative.');
        }
        $C = array_fill(0, $n + 1, 0.0);
        $C[0] = -cos($x);
        if ($n >= 1) {
            $C[1] = -cos($x) / $x - sin($x);
        }
        for ($k = 1; $k < $n; $k++) {
            $C[$k + 1] = (2 * $k + 1) / $x * $C[$k] - $C[$k - 1];
        }
        $D = array_fill(0, $n + 1, 0.0);
        $D[0] = sin($x);
        for ($k = 1; $k <= $n; $k++) {
            $D[$k] = $C[$k - 1] - $k / $x * $C[$k];
        }

        return [$C, $D];
    }

    /**
     * P_n(x) and its derivatives up to order $diffN via the Legendre recurrence plus the
     * differentiated Legendre ODE.
     *
     * @return list<float>
     */
    private static function legDerivs(int $n, float $x, int $diffN): array
    {
        $pn = self::legP($n, $x);
        $out = [$pn];
        if ($diffN >= 1) {
            $pnm1 = $n >= 1 ? self::legP($n - 1, $x) : 0.0;
            $out[1] = $n >= 1 ? $n * ($x * $pn - $pnm1) / ($x * $x - 1.0) : 0.0;
        }
        // (1 - x^2) y^(k+2) - 2x(k+1) y^(k+1) + (n(n+1) - k(k+1)) y^(k) = 0
        for ($k = 0; $k < $diffN - 1; $k++) {
            $out[$k + 2] = (2 * $x * ($k + 1) * $out[$k + 1] - ($n * ($n + 1) - $k * ($k + 1)) * $out[$k]) / (1.0 - $x * $x);
        }

        return $out;
    }

    private static function legP(int $n, float $x): float
    {
        if ($n === 0) {
            return 1.0;
        }
        $p0 = 1.0;
        $p1 = $x;
        for ($k = 1; $k < $n; $k++) {
            $p2 = ((2 * $k + 1) * $x * $p1 - $k * $p0) / ($k + 1);
            $p0 = $p1;
            $p1 = $p2;
        }

        return $p1;
    }
}
