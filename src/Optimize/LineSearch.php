<?php

declare(strict_types=1);

namespace Tessero\Optimize;

/**
 * scipy.optimize.line_search and scipy.optimize.approx_fprime (pure PHP).
 *
 * scipy.optimize.line_search is line_search_wolfe2 / scalar_search_wolfe2 (strong-Wolfe,
 * Algorithm 3.5-3.6 of Nocedal & Wright). This is a line-for-line port returning scipys
 * tuple [alpha, fc, gc, new_fval, old_fval, new_slope] where new_slope is the gradient
 * array g(xk + alpha*pk) at the accepted step (scipy sets derphi_star = gval[0]).
 *
 * approx_fprime is scipy forward 2-point finite differences with the exact absolute-step
 * handling (default sqrt(eps) ~= 1.49e-8).
 */
final class LineSearch
{
    private const EPS = 2.220446049250313e-16;
    private const DEFAULT_EPS = 1.4901161193847656e-08;

    /**
     * scipy.optimize.approx_fprime(xk, f, epsilon=sqrt(eps)).
     *
     * @param list<float> $xk
     * @param float|list<float> $epsilon
     * @return list<float>
     */
    public static function approxFprime(callable $f, array $xk, float|array $epsilon = self::DEFAULT_EPS, ?float $f0 = null): array
    {
        $xk = array_map("floatval", array_values($xk));
        $n = count($xk);
        $f0 = $f0 ?? (float) $f($xk);
        $rstep = sqrt(self::EPS);
        $g = [];
        for ($i = 0; $i < $n; $i++) {
            $xi = $xk[$i];
            $eps = is_array($epsilon) ? (float) $epsilon[$i] : (float) $epsilon;
            $h = $eps;
            $dx = ($xi + $h) - $xi;
            if ($dx == 0.0) {
                $sign = $xi >= 0 ? 1.0 : -1.0;
                $h = $rstep * $sign * max(1.0, abs($xi));
            }
            $xh = $xk;
            $xh[$i] = $xi + $h;
            $dx = $xh[$i] - $xi;
            $g[$i] = ((float) $f($xh) - $f0) / $dx;
        }
        return $g;
    }

    /**
     * scipy.optimize.line_search -> [alpha, fc, gc, new_fval, old_fval, new_slope].
     *
     * @param list<float> $xk
     * @param list<float> $pk
     * @param list<float>|null $gfk
     * @param callable|null $extraCondition  fn(float $alpha, list<float> $x, float $phi, list<float> $g): bool
     * @return array{0: float|null, 1: int, 2: int, 3: float|null, 4: float|null, 5: list<float>|null}
     */
    public static function lineSearch(callable $f, callable $fprime, array $xk, array $pk, ?array $gfk = null, ?float $oldFval = null, ?float $oldOldFval = null, float $c1 = 1e-4, float $c2 = 0.9, ?float $amax = null, ?callable $extraCondition = null, int $maxiter = 10): array
    {
        $xk = array_map("floatval", array_values($xk));
        $pk = array_map("floatval", array_values($pk));
        $fc = 0;
        $gc = 0;
        $gval = null;
        $gvalAlpha = null;
        $phi = function (float $alpha) use (&$fc, $f, $xk, $pk): float {
            $fc++;
            return (float) $f(self::axpy($alpha, $pk, $xk));
        };
        $derphi = function (float $alpha) use (&$gc, &$gval, &$gvalAlpha, $fprime, $xk, $pk): float {
            $gc++;
            $gval = array_map("floatval", array_values($fprime(self::axpy($alpha, $pk, $xk))));
            $gvalAlpha = $alpha;
            return self::dot($gval, $pk);
        };
        if ($gfk === null) {
            $gfk = array_map("floatval", array_values($fprime($xk)));
        } else {
            $gfk = array_map("floatval", array_values($gfk));
        }
        $derphi0 = self::dot($gfk, $pk);
        $ec2 = null;
        if ($extraCondition !== null) {
            $ec2 = function (float $alpha, float $phiv) use (&$gval, &$gvalAlpha, $derphi, $extraCondition, $xk, $pk): bool {
                if ($gvalAlpha !== $alpha) {
                    $derphi($alpha);
                }
                $x = self::axpy($alpha, $pk, $xk);
                return (bool) $extraCondition($alpha, $x, $phiv, $gval);
            };
        }
        [$alphaStar, $phiStar, $oldFvalOut, $derphiStar] = self::scalarSearchWolfe2(
            $phi, $derphi, $oldFval, $oldOldFval, $derphi0, $c1, $c2, $amax, $ec2, $maxiter
        );
        $newSlope = $derphiStar === null ? null : $gval;
        return [$alphaStar, $fc, $gc, $phiStar, $oldFvalOut, $newSlope];
    }

    /**
     * scipy.optimize._linesearch.scalar_search_wolfe2.
     *
     * @return array{0: float|null, 1: float|null, 2: float|null, 3: float|null} [alpha_star, phi_star, phi0, derphi_star]
     */
    public static function scalarSearchWolfe2(callable $phi, callable $derphi, ?float $phi0 = null, ?float $oldPhi0 = null, ?float $derphi0 = null, float $c1 = 1e-4, float $c2 = 0.9, ?float $amax = null, ?callable $extraCondition = null, int $maxiter = 10): array
    {
        $phi0 = $phi0 ?? $phi(0.0);
        $derphi0 = $derphi0 ?? $derphi(0.0);
        $alpha0 = 0.0;
        if ($oldPhi0 !== null && $derphi0 != 0.0) {
            $alpha1 = min(1.0, 1.01 * 2 * ($phi0 - $oldPhi0) / $derphi0);
        } else {
            $alpha1 = 1.0;
        }
        if ($alpha1 < 0) {
            $alpha1 = 1.0;
        }
        if ($amax !== null) {
            $alpha1 = min($alpha1, $amax);
        }
        $phiA1 = $phi($alpha1);
        $phiA0 = $phi0;
        $derphiA0 = $derphi0;
        if ($extraCondition === null) {
            $extraCondition = static fn (float $a, float $p): bool => true;
        }
        $alphaStar = null; $phiStar = null; $derphiStar = null;
        $done = false;
        for ($i = 0; $i < $maxiter; $i++) {
            if ($alpha1 == 0.0 || ($amax !== null && $alpha0 > $amax)) {
                $alphaStar = null;
                $phiStar = $phi0;
                $phi0 = $oldPhi0;
                $derphiStar = null;
                $done = true;
                break;
            }
            $notFirst = $i > 0;
            if (($phiA1 > $phi0 + $c1 * $alpha1 * $derphi0) || (($phiA1 >= $phiA0) && $notFirst)) {
                [$alphaStar, $phiStar, $derphiStar] = self::zoom($alpha0, $alpha1, $phiA0, $phiA1, $derphiA0, $phi, $derphi, $phi0, $derphi0, $c1, $c2, $extraCondition);
                $done = true;
                break;
            }
            $derphiA1 = $derphi($alpha1);
            if (abs($derphiA1) <= -$c2 * $derphi0) {
                if ($extraCondition($alpha1, $phiA1)) {
                    $alphaStar = $alpha1;
                    $phiStar = $phiA1;
                    $derphiStar = $derphiA1;
                    $done = true;
                    break;
                }
            }
            if ($derphiA1 >= 0) {
                [$alphaStar, $phiStar, $derphiStar] = self::zoom($alpha1, $alpha0, $phiA1, $phiA0, $derphiA1, $phi, $derphi, $phi0, $derphi0, $c1, $c2, $extraCondition);
                $done = true;
                break;
            }
            $alpha2 = 2 * $alpha1;
            if ($amax !== null) {
                $alpha2 = min($alpha2, $amax);
            }
            $alpha0 = $alpha1;
            $alpha1 = $alpha2;
            $phiA0 = $phiA1;
            $phiA1 = $phi($alpha1);
            $derphiA0 = $derphiA1;
        }
        if (!$done) {
            $alphaStar = $alpha1;
            $phiStar = $phiA1;
            $derphiStar = null;
        }
        return [$alphaStar, $phiStar, $phi0, $derphiStar];
    }

    /**
     * scipy.optimize._linesearch._zoom (Algorithm 3.6).
     *
     * @return array{0: float|null, 1: float|null, 2: float|null}
     */
    private static function zoom(float $aLo, float $aHi, float $phiLo, float $phiHi, float $derphiLo, callable $phi, callable $derphi, float $phi0, float $derphi0, float $c1, float $c2, callable $extraCondition): array
    {
        $maxiter = 10;
        $i = 0;
        $delta1 = 0.2;
        $delta2 = 0.1;
        $phiRec = $phi0;
        $aRec = 0.0;
        while (true) {
            $dalpha = $aHi - $aLo;
            if ($dalpha < 0) { $a = $aHi; $b = $aLo; } else { $a = $aLo; $b = $aHi; }
            $aJ = null;
            if ($i > 0) {
                $cchk = $delta1 * $dalpha;
                $aJ = self::cubicmin($aLo, $phiLo, $derphiLo, $aHi, $phiHi, $aRec, $phiRec);
            }
            if ($i == 0 || $aJ === null || $aJ > $b - ($delta1 * $dalpha) || $aJ < $a + ($delta1 * $dalpha)) {
                $qchk = $delta2 * $dalpha;
                $aJ = self::quadmin($aLo, $phiLo, $derphiLo, $aHi, $phiHi);
                if ($aJ === null || $aJ > $b - $qchk || $aJ < $a + $qchk) {
                    $aJ = $aLo + 0.5 * $dalpha;
                }
            }
            $phiAj = $phi($aJ);
            if (($phiAj > $phi0 + $c1 * $aJ * $derphi0) || ($phiAj >= $phiLo)) {
                $phiRec = $phiHi;
                $aRec = $aHi;
                $aHi = $aJ;
                $phiHi = $phiAj;
            } else {
                $derphiAj = $derphi($aJ);
                if (abs($derphiAj) <= -$c2 * $derphi0 && $extraCondition($aJ, $phiAj)) {
                    return [$aJ, $phiAj, $derphiAj];
                }
                if ($derphiAj * ($aHi - $aLo) >= 0) {
                    $phiRec = $phiHi;
                    $aRec = $aHi;
                    $aHi = $aLo;
                    $phiHi = $phiLo;
                } else {
                    $phiRec = $phiLo;
                    $aRec = $aLo;
                }
                $aLo = $aJ;
                $phiLo = $phiAj;
                $derphiLo = $derphiAj;
            }
            $i++;
            if ($i > $maxiter) {
                return [null, null, null];
            }
        }
    }

    /** scipy.optimize._linesearch._cubicmin. */
    private static function cubicmin(float $a, float $fa, float $fpa, float $b, float $fb, float $c, float $fc): ?float
    {
        $C = $fpa;
        $db = $b - $a;
        $dc = $c - $a;
        $denom = ($db * $dc) ** 2 * ($db - $dc);
        if ($denom == 0.0 || !is_finite($denom)) { return null; }
        $r1 = $fb - $fa - $C * $db;
        $r2 = $fc - $fa - $C * $dc;
        $A = ($dc ** 2 * $r1 - $db ** 2 * $r2) / $denom;
        $B = (-($dc ** 3) * $r1 + $db ** 3 * $r2) / $denom;
        if ($A == 0.0) { return null; }
        $radical = $B * $B - 3 * $A * $C;
        if ($radical < 0) { return null; }
        $xmin = $a + (-$B + sqrt($radical)) / (3 * $A);
        return is_finite($xmin) ? $xmin : null;
    }

    /** scipy.optimize._linesearch._quadmin. */
    private static function quadmin(float $a, float $fa, float $fpa, float $b, float $fb): ?float
    {
        $D = $fa;
        $C = $fpa;
        $db = $b - $a * 1.0;
        if ($db == 0.0) { return null; }
        $B = ($fb - $D - $C * $db) / ($db * $db);
        if ($B == 0.0) { return null; }
        $xmin = $a - $C / (2.0 * $B);
        return is_finite($xmin) ? $xmin : null;
    }

    /** @param list<float> $x @param list<float> $y @return list<float> */
    private static function axpy(float $a, array $x, array $y): array
    {
        $out = [];
        foreach ($y as $i => $yi) {
            $out[$i] = $yi + $a * $x[$i];
        }
        return $out;
    }

    /** @param list<float> $a @param list<float> $b */
    private static function dot(array $a, array $b): float
    {
        $s = 0.0;
        foreach ($a as $i => $ai) {
            $s += $ai * $b[$i];
        }
        return $s;
    }
}
