<?php

declare(strict_types=1);

namespace Tessero\Integrate;

/**
 * Convergent series summation (scipy.integrate.nsum).
 *
 * Finite limits: evaluates sum_{k=0}^{n} f(a + k*step) with n = round((b - a) / step), matching scipy
 * `_direct` term-by-term evaluation to machine precision (NumPy reduces with pairwise summation while PHP
 * sums sequentially, so very long sums may differ in the last ~1-2 ULP).
 *
 * Infinite limits (b = INF / null): mirrors scipy `_integral_bound`. The tail beyond a cut index k is
 * replaced by a tanh-sinh integral:
 *     S = left + right/step + f(k)/2 + f(INF)/2 ,   f(INF) = 0 ,
 * where `left` is the direct partial sum a..k (exclusive) and `right` is the tanh-sinh integral of f over
 * the half-infinite interval [k, INF). The cut k = a + n_steps*step is chosen with scipy power-of-two
 * heuristic: n_steps runs over [2^0, 2^1, ..., 2^floor(log2(maxterms)), maxterms] and the smallest value at
 * which f(k) <= tol and f is still decreasing is used; tol = atol + rtol*lb where lb is a lower-bound
 * tanh-sinh integral of f over [a, INF). The half-infinite integral uses the substitution x = 1/t - 1 + k
 * (so f-tail over [k, INF) becomes f(1/t-1+k)/t^2 over the finite [0, 1]) fed to {@see Tanhsinh}. The sum
 * matches scipy.integrate.nsum(f, a, inf).sum to scipy default tolerance. Pure PHP, host-independent.
 *
 * The integrand must be a smooth, positive, unimodal, eventually-decreasing function of a single real arg.
 */
final class Nsum
{
    private const EPS = 2.220446049250313e-16;

    /**
     * @param callable(float): float $f     term function f(k)
     * @param float                  $a     first index
     * @param float|null             $b     last index (null/INF = infinite sum)
     * @param float                  $step  index step (default 1)
     * @return array{sum: float, error: float, nfev: int, success: bool, status: int}
     */
    public static function nsum(callable $f, float $a, ?float $b = null, float $step = 1.0): array
    {
        if ($step <= 0.0) {
            throw new \RuntimeException('Tessero\Integrate\Nsum::nsum: step must be positive');
        }
        if ($b === null || is_infinite($b)) {
            return self::nsumInfinite($f, $a, $step);
        }
        if ($b < $a) {
            return ['sum' => 0.0, 'error' => 0.0, 'nfev' => 0, 'success' => true, 'status' => 0];
        }
        $n = (int) round(($b - $a) / $step);
        $s = 0.0;
        $maxAbs = 0.0;
        for ($k = 0; $k <= $n; $k++) {
            $term = (float) $f($a + $k * $step);
            $s += $term;
            $at = abs($term);
            if ($at > $maxAbs) { $maxAbs = $at; }
        }
        return ['sum' => $s, 'error' => self::EPS * abs($s), 'nfev' => $n + 1, 'success' => true, 'status' => 0];
    }

    /**
     * Infinite-sum path (b = INF). See class docblock; follows scipy `_integral_bound`.
     *
     * @param callable(float): float $f
     * @return array{sum: float, error: float, nfev: int, success: bool, status: int}
     */
    private static function nsumInfinite(callable $f, float $a, float $step): array
    {
        $rtol = sqrt(self::EPS);         // scipy nsum default rtol = eps**0.5
        $atol = 0.0;
        $maxterms = 2.0 ** 20;           // scipy default maxterms

        // Lower-bound integral over [a, INF) gives the effective tolerance.
        $lb = self::tanhsinhRight($f, $a);
        $tol = $atol + $rtol * $lb['integral'];

        // n_steps candidates: powers of two up to floor(log2(maxterms)), then maxterms itself.
        $log2mt = $maxterms > 0 ? (int) floor(log($maxterms, 2)) : 0;
        $nSteps = [];
        for ($i = 0; $i < $log2mt; $i++) { $nSteps[] = (float) (2 ** $i); }
        $nSteps[] = $maxterms;

        $nfev = count($nSteps) * 2;
        $nt = null;
        foreach ($nSteps as $idx => $ns) {
            $kk = $a + $ns * $step;
            $fk = (float) $f($kk);
            $fkp1 = (float) $f($kk + $step);
            $insufficient = ($fk > $tol) || ($fkp1 > $fk);
            if (! $insufficient) { $nt = $idx; break; }
        }
        $fkInsufficient = ($nt === null);
        if ($nt === null) { $nt = count($nSteps) - 1; }
        $nStepsSel = $nSteps[$nt];

        $k = $a + $nStepsSel * $step;

        // left: direct partial sum a..k, EXCLUSIVE of k (scipy _direct inclusive=False).
        $m = (int) round(($k - $a) / $step);
        $left = 0.0;
        for ($j = 0; $j < $m; $j++) { $left += (float) $f($a + $j * $step); }
        $nfev += $m;

        $fk = (float) $f($k);
        $nfev += 1;

        $right = self::tanhsinhRight($f, $k);
        $nfev += $right['nfev'] + $lb['nfev'];

        // f(INF) = 0 for a convergent sum.
        $S = $left + $right['integral'] / $step + $fk / 2.0;
        $E = $lb['error'] + $right['error'] / $step + $fk / 2.0;

        $status = $right['status'];
        if ($status === 0 && $fkInsufficient) { $status = -4; }

        return [
            'sum' => $S,
            'error' => $E,
            'nfev' => $nfev,
            'success' => ($status === 0),
            'status' => $status,
        ];
    }

    /**
     * Tanh-sinh integral of f over the half-infinite interval [k, INF). Uses the substitution
     * x = 1/t - 1 + k, which maps [k, INF) onto the finite (0, 1]; the transformed integrand is
     * g(t) = f(1/t - 1 + k) / t^2 (and g(0) := 0 for the convergent tail). Matches scipy
     * tanhsinh(f, k, inf).integral.
     *
     * @param callable(float): float $f
     * @return array{integral: float, error: float, success: bool, status: int, maxlevel: int, nfev: int}
     */
    private static function tanhsinhRight(callable $f, float $k): array
    {
        $g = static function (float $t) use ($f, $k): float {
            $t2 = $t * $t;
            if ($t2 === 0.0) { return 0.0; }            // t underflow: convergent tail contributes 0
            $x = 1.0 / $t - 1.0 + $k;
            return (float) $f($x) / $t2;
        };
        return Tanhsinh::tanhsinh($g, 0.0, 1.0);
    }
}
