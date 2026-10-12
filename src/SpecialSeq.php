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
     * Associated Legendre polynomial (Ferrers function) P_n^m(x) and its x-derivatives up to
     * $diffN (0-2), with the Condon-Shortley phase and branch_cut=2 (|x| < 1). If $norm, apply
     * the normalisation factor sqrt((2n+1)(n-m)! / (2(n+m)!)).
     *
     * @return list<float>
     */
    public static function assocLegendreP(int $n, int $m, float $x, int $diffN = 0, bool $norm = false): array
    {
        if ($n < 0) {
            throw new TesseroException('n must be nonnegative.');
        }
        $am = abs($m);
        if ($am > $n) {
            return array_fill(0, $diffN + 1, 0.0);
        }
        $res = self::assocPos($n, $am, $x, $diffN);
        if ($norm) {
            $f = sqrt((2 * $n + 1) * self::fact($n - $am) / (2.0 * self::fact($n + $am)));
            if ($m < 0 && $am % 2 === 1) {
                $f = -$f;
            }
            $res = array_map(static fn ($v): float => $f * $v, $res);
        } elseif ($m < 0) {
            $c = ($am % 2 === 0 ? 1.0 : -1.0) * self::fact($n - $am) / self::fact($n + $am);
            $res = array_map(static fn ($v): float => $c * $v, $res);
        }

        return $res;
    }

    /**
     * P_k^j(x) and derivatives for every degree k = 0..n and order j, laid out like scipy's
     * assoc_legendre_p_all: shape [$diffN + 1][n + 1][2m + 1] with the order axis wrapped
     * (index 0..m = orders 0..m, then m+1..2m = orders -m..-1).
     *
     * @return list<list<list<float>>>
     */
    public static function assocLegendrePAll(int $n, int $m, float $x, int $diffN = 0, bool $norm = false): array
    {
        return self::assembleAll($n, $m, $diffN, fn (int $nn, int $ord): array => self::assocLegendreP($nn, $ord, $x, $diffN, $norm));
    }

    /**
     * Spherical (fully normalised) Legendre function Y_n^m(theta) without the azimuthal factor,
     * and its theta-derivatives up to $diffN (0-2).
     *
     * @return list<float>
     */
    public static function sphLegendreP(int $n, int $m, float $theta, int $diffN = 0): array
    {
        if ($n < 0) {
            throw new TesseroException('n must be nonnegative.');
        }
        $x = cos($theta);
        $s = sin($theta);
        $V = self::assocLegendreP($n, $m, $x, min($diffN, 2), true);
        $k = sqrt(2.0 * M_PI);
        $V = array_map(static fn ($v): float => $v / $k, $V);
        $res = [$V[0]];
        if ($diffN >= 1) {
            $res[1] = -$s * $V[1];
        }
        if ($diffN >= 2) {
            $res[2] = $s * $s * $V[2] - $x * $V[1];
        }

        return $res;
    }

    /**
     * sph_legendre_p for every degree/order, laid out like scipy's sph_legendre_p_all.
     *
     * @return list<list<list<float>>>
     */
    public static function sphLegendrePAll(int $n, int $m, float $theta, int $diffN = 0): array
    {
        return self::assembleAll($n, $m, $diffN, fn (int $nn, int $ord): array => self::sphLegendreP($nn, $ord, $theta, $diffN));
    }

    /**
     * Assemble an "_all" result [$diffN+1][n+1][2m+1] from a per-(degree, order) family builder,
     * with the scipy order-axis wrapping (0..m then -m..-1).
     *
     * @param callable(int, int): list<float> $family
     * @return list<list<list<float>>>
     */
    private static function assembleAll(int $n, int $m, int $diffN, callable $family): array
    {
        $out = [];
        for ($d = 0; $d <= $diffN; $d++) {
            $out[$d] = [];
            for ($nn = 0; $nn <= $n; $nn++) {
                $out[$d][$nn] = array_fill(0, 2 * $m + 1, 0.0);
            }
        }
        for ($nn = 0; $nn <= $n; $nn++) {
            for ($j = 0; $j < 2 * $m + 1; $j++) {
                $ord = $j <= $m ? $j : $j - (2 * $m + 1);
                $v = $family($nn, $ord);
                for ($d = 0; $d <= $diffN; $d++) {
                    $out[$d][$nn][$j] = $v[$d];
                }
            }
        }

        return $out;
    }

    /** P_n^mm(x) and x-derivatives (mm >= 0), from the m-th derivative of P_n times (1-x^2)^(mm/2). */
    private static function assocPos(int $n, int $mm, float $x, int $diffN): array
    {
        $u = self::legDerivs($n, $x, $mm + 2);
        $um = $u[$mm];
        $um1 = $u[$mm + 1];
        $um2 = $u[$mm + 2];
        $w = 1.0 - $x * $x;
        $sgn = $mm % 2 === 0 ? 1.0 : -1.0;
        $res = [$sgn * $w ** ($mm / 2.0) * $um];
        if ($diffN >= 1) {
            $A = -$mm * $x * $um + $w * $um1;
            $res[1] = $sgn * $w ** ($mm / 2.0 - 1.0) * $A;
        }
        if ($diffN >= 2) {
            $A = -$mm * $x * $um + $w * $um1;
            $Ap = -$mm * $um - ($mm + 2) * $x * $um1 + $w * $um2;
            $res[2] = $sgn * $w ** ($mm / 2.0 - 2.0) * (-($mm - 2) * $x * $A + $w * $Ap);
        }

        return $res;
    }

    private static function fact(int $k): float
    {
        $r = 1.0;
        for ($i = 2; $i <= $k; $i++) {
            $r *= $i;
        }

        return $r;
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
