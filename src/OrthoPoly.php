<?php

declare(strict_types=1);

namespace Tessero;

use Tessero\Exceptions\TesseroException;

/**
 * The scipy.special orthogonal-polynomial factories (legendre, chebyt, jacobi, ...), each
 * returning an {@see OrthoPoly1d} — the power-basis polynomial as a callable object.
 *
 * The coefficients are built from each family's three-term recurrence in the power basis
 * (not from the Gauss nodes), which reproduces scipy's `orthopoly1d.c` to machine precision
 * while staying pure PHP (no kernel, identical on both backends). The derived families
 * (chebyc, chebys, the shifted polynomials, ...) are obtained by composing/scaling a base
 * family exactly as scipy does.
 */
final class OrthoPoly
{
    // ---- polynomial helpers (coefficients highest-first) --------------------------------

    /**
     * @param list<float> $a
     * @param list<float> $b
     * @return list<float>
     */
    private static function padd(array $a, array $b): array
    {
        $n = max(count($a), count($b));
        $a = array_merge(array_fill(0, $n - count($a), 0.0), $a);
        $b = array_merge(array_fill(0, $n - count($b), 0.0), $b);
        $r = [];
        for ($i = 0; $i < $n; $i++) {
            $r[$i] = $a[$i] + $b[$i];
        }

        return $r;
    }

    /**
     * @param list<float> $a
     * @return list<float>
     */
    private static function sxp(array $a, float $s): array
    {
        return array_map(static fn ($x): float => $s * $x, $a);
    }

    /**
     * Multiply a polynomial by x (shift coefficients up one degree).
     *
     * @param list<float> $a
     * @return list<float>
     */
    private static function mulx(array $a): array
    {
        $a[] = 0.0;

        return $a;
    }

    /**
     * Compose a(m*x + c) and return the resulting coefficients.
     *
     * @param list<float> $a
     * @return list<float>
     */
    private static function compose(array $a, float $m, float $c): array
    {
        $res = [$a[0]];
        $n = count($a);
        for ($i = 1; $i < $n; $i++) {
            $r = array_fill(0, count($res) + 1, 0.0);
            foreach ($res as $j => $rv) {
                $r[$j] += $rv * $m;
                $r[$j + 1] += $rv * $c;
            }
            $r[count($r) - 1] += $a[$i];
            $res = $r;
        }

        return $res;
    }

    /**
     * Scale a polynomial so its leading coefficient is 1 (monic).
     *
     * @param list<float> $a
     * @return list<float>
     */
    private static function leadnorm(array $a): array
    {
        return self::sxp($a, 1.0 / $a[0]);
    }

    private static function nonneg(int $n): void
    {
        if ($n < 0) {
            throw new TesseroException('n must be nonnegative.');
        }
    }

    // ---- base families (private coefficient builders) -----------------------------------

    /** @return list<float> */
    private static function legendreC(int $n): array
    {
        $Pm1 = [1.0];
        $Pk = [1.0, 0.0];
        if ($n === 0) {
            return $Pm1;
        }
        for ($k = 1; $k < $n; $k++) {
            $Pk1 = self::sxp(self::padd(self::sxp(self::mulx($Pk), 2 * $k + 1), self::sxp($Pm1, -$k)), 1.0 / ($k + 1));
            $Pm1 = $Pk;
            $Pk = $Pk1;
        }

        return $Pk;
    }

    /** @return list<float> */
    private static function chebytC(int $n): array
    {
        $Pm1 = [1.0];
        $Pk = [1.0, 0.0];
        if ($n === 0) {
            return $Pm1;
        }
        for ($k = 1; $k < $n; $k++) {
            $Pk1 = self::padd(self::sxp(self::mulx($Pk), 2.0), self::sxp($Pm1, -1.0));
            $Pm1 = $Pk;
            $Pk = $Pk1;
        }

        return $Pk;
    }

    /** @return list<float> */
    private static function chebyuC(int $n): array
    {
        $Pm1 = [1.0];
        $Pk = [2.0, 0.0];
        if ($n === 0) {
            return $Pm1;
        }
        for ($k = 1; $k < $n; $k++) {
            $Pk1 = self::padd(self::sxp(self::mulx($Pk), 2.0), self::sxp($Pm1, -1.0));
            $Pm1 = $Pk;
            $Pk = $Pk1;
        }

        return $Pk;
    }

    /** @return list<float> */
    private static function hermiteC(int $n): array
    {
        $Pm1 = [1.0];
        $Pk = [2.0, 0.0];
        if ($n === 0) {
            return $Pm1;
        }
        for ($k = 1; $k < $n; $k++) {
            $Pk1 = self::padd(self::sxp(self::mulx($Pk), 2.0), self::sxp($Pm1, -2.0 * $k));
            $Pm1 = $Pk;
            $Pk = $Pk1;
        }

        return $Pk;
    }

    /** @return list<float> */
    private static function hermitenormC(int $n): array
    {
        $Pm1 = [1.0];
        $Pk = [1.0, 0.0];
        if ($n === 0) {
            return $Pm1;
        }
        for ($k = 1; $k < $n; $k++) {
            $Pk1 = self::padd(self::mulx($Pk), self::sxp($Pm1, -1.0 * $k));
            $Pm1 = $Pk;
            $Pk = $Pk1;
        }

        return $Pk;
    }

    /** @return list<float> */
    private static function laguerreC(int $n): array
    {
        $Pm1 = [1.0];
        $Pk = [-1.0, 1.0];
        if ($n === 0) {
            return $Pm1;
        }
        for ($k = 1; $k < $n; $k++) {
            $Pk1 = self::sxp(self::padd(self::padd(self::sxp(self::mulx($Pk), -1.0), self::sxp($Pk, 2 * $k + 1)), self::sxp($Pm1, -$k)), 1.0 / ($k + 1));
            $Pm1 = $Pk;
            $Pk = $Pk1;
        }

        return $Pk;
    }

    /** @return list<float> */
    private static function genlaguerreC(int $n, float $al): array
    {
        $Pm1 = [1.0];
        $Pk = [-1.0, 1.0 + $al];
        if ($n === 0) {
            return $Pm1;
        }
        for ($k = 1; $k < $n; $k++) {
            $Pk1 = self::sxp(self::padd(self::padd(self::sxp(self::mulx($Pk), -1.0), self::sxp($Pk, 2 * $k + 1 + $al)), self::sxp($Pm1, -($k + $al))), 1.0 / ($k + 1));
            $Pm1 = $Pk;
            $Pk = $Pk1;
        }

        return $Pk;
    }

    /** @return list<float> */
    private static function jacobiC(int $n, float $a, float $b): array
    {
        if ($n === 0) {
            return [1.0];
        }
        $P1 = [0.5 * ($a + $b + 2.0), 0.5 * ($a - $b)];
        if ($n === 1) {
            return $P1;
        }
        $Pm1 = [1.0];
        $Pk = $P1;
        for ($k = 1; $k < $n; $k++) {
            $m = $k + 1;
            $c1 = 2 * $m * ($m + $a + $b) * (2 * $m + $a + $b - 2.0);
            $c2x = (2 * $m + $a + $b - 1.0) * (2 * $m + $a + $b) * (2 * $m + $a + $b - 2.0);
            $c2c = (2 * $m + $a + $b - 1.0) * ($a * $a - $b * $b);
            $c3 = 2 * ($m + $a - 1.0) * ($m + $b - 1.0) * (2 * $m + $a + $b);
            $Pk1 = self::sxp(self::padd(self::padd(self::sxp(self::mulx($Pk), $c2x), self::sxp($Pk, $c2c)), self::sxp($Pm1, -$c3)), 1.0 / $c1);
            $Pm1 = $Pk;
            $Pk = $Pk1;
        }

        return $Pk;
    }

    /** @return list<float> */
    private static function gegenbauerC(int $n, float $al): array
    {
        if ($n === 0) {
            return [1.0];
        }
        $P1 = [2.0 * $al, 0.0];
        if ($n === 1) {
            return $P1;
        }
        $Pm1 = [1.0];
        $Pk = $P1;
        for ($k = 1; $k < $n; $k++) {
            $m = $k + 1;
            $Pk1 = self::sxp(self::padd(self::sxp(self::mulx($Pk), 2.0 * ($m + $al - 1.0)), self::sxp($Pm1, -($m + 2 * $al - 2.0))), 1.0 / $m);
            $Pm1 = $Pk;
            $Pk = $Pk1;
        }

        return $Pk;
    }

    /** @return list<float> */
    private static function shJacobiC(int $n, float $p, float $q): array
    {
        if ($n === 0) {
            return [1.0];
        }

        return self::leadnorm(self::compose(self::jacobiC($n, $p - $q, $q - 1.0), 2.0, -1.0));
    }

    // ---- public factories ---------------------------------------------------------------

    public static function legendre(int $n): OrthoPoly1d
    {
        self::nonneg($n);

        return new OrthoPoly1d(self::legendreC($n));
    }

    public static function chebyt(int $n): OrthoPoly1d
    {
        self::nonneg($n);

        return new OrthoPoly1d(self::chebytC($n));
    }

    public static function chebyu(int $n): OrthoPoly1d
    {
        self::nonneg($n);

        return new OrthoPoly1d(self::chebyuC($n));
    }

    /** Chebyshev polynomial of the first kind on [-2, 2]: C_n(x) = 2 T_n(x/2). */
    public static function chebyc(int $n): OrthoPoly1d
    {
        self::nonneg($n);
        if ($n === 0) {
            return new OrthoPoly1d([2.0]);
        }

        return new OrthoPoly1d(self::sxp(self::compose(self::chebytC($n), 0.5, 0.0), 2.0));
    }

    /** Chebyshev polynomial of the second kind on [-2, 2]: S_n(x) = U_n(x/2). */
    public static function chebys(int $n): OrthoPoly1d
    {
        self::nonneg($n);

        return new OrthoPoly1d(self::compose(self::chebyuC($n), 0.5, 0.0));
    }

    public static function hermite(int $n): OrthoPoly1d
    {
        self::nonneg($n);

        return new OrthoPoly1d(self::hermiteC($n));
    }

    public static function hermitenorm(int $n): OrthoPoly1d
    {
        self::nonneg($n);

        return new OrthoPoly1d(self::hermitenormC($n));
    }

    public static function laguerre(int $n): OrthoPoly1d
    {
        self::nonneg($n);

        return new OrthoPoly1d(self::laguerreC($n));
    }

    public static function genlaguerre(int $n, float $alpha): OrthoPoly1d
    {
        self::nonneg($n);
        if ($alpha <= -1) {
            throw new TesseroException('alpha must be > -1');
        }

        return new OrthoPoly1d(self::genlaguerreC($n, $alpha));
    }

    public static function gegenbauer(int $n, float $alpha): OrthoPoly1d
    {
        self::nonneg($n);
        if (! is_finite($alpha) || $alpha <= -0.5) {
            throw new TesseroException('`alpha` must be a finite number greater than -1/2');
        }

        return new OrthoPoly1d(self::gegenbauerC($n, $alpha));
    }

    public static function jacobi(int $n, float $alpha, float $beta): OrthoPoly1d
    {
        self::nonneg($n);
        if ($alpha <= -1 || $beta <= -1) {
            throw new TesseroException('alpha and beta must be > -1');
        }

        return new OrthoPoly1d(self::jacobiC($n, $alpha, $beta));
    }

    /** Shifted Legendre polynomial on [0, 1]: P*_n(x) = P_n(2x - 1). */
    public static function shLegendre(int $n): OrthoPoly1d
    {
        self::nonneg($n);

        return new OrthoPoly1d(self::compose(self::legendreC($n), 2.0, -1.0));
    }

    /** Shifted Chebyshev polynomial of the first kind on [0, 1]. */
    public static function shChebyt(int $n): OrthoPoly1d
    {
        self::nonneg($n);
        $base = self::shJacobiC($n, 0.0, 0.5);

        return new OrthoPoly1d($n > 0 ? self::sxp($base, (4.0 ** $n) / 2.0) : $base);
    }

    /** Shifted Chebyshev polynomial of the second kind on [0, 1]. */
    public static function shChebyu(int $n): OrthoPoly1d
    {
        self::nonneg($n);

        return new OrthoPoly1d(self::sxp(self::shJacobiC($n, 2.0, 1.5), 4.0 ** $n));
    }

    /** Shifted Jacobi polynomial (standard form, monic) on [0, 1]. */
    public static function shJacobi(int $n, float $p, float $q): OrthoPoly1d
    {
        self::nonneg($n);

        return new OrthoPoly1d(self::shJacobiC($n, $p, $q));
    }
}
