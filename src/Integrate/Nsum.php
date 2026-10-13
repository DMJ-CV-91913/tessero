<?php

declare(strict_types=1);

namespace Tessero\Integrate;

/**
 * Convergent series summation (scipy.integrate.nsum).
 *
 * For finite limits this evaluates sum_{k=0}^{n} f(a + k*step) with n = round((b - a) / step), matching
 * scipy's `_direct` term-by-term evaluation to machine precision (the only departure is that NumPy reduces
 * with pairwise summation while PHP sums sequentially, so very long sums may differ in the last ~1-2 ULP).
 *
 * The integrand must be a smooth, positive, unimodal function of a single real argument.
 *
 * NOTE: the infinite-limit path (b = INF) is a follow-up. scipy approximates it as
 *     S = left + right.integral/step + f(k)/2 + f(b)/2
 * where `left` is the direct partial sum a..k (exclusive), `k = a + n_steps*step` with n_steps the
 * smallest power of two (<= maxterms) at which f(k) <= tol and f is decreasing, `right.integral` is the
 * tanh-sinh integral of f over the HALF-INFINITE interval [k, INF), and f(b)=0 for b=INF. Implementing it
 * requires extending {@see Tanhsinh} to half-infinite intervals plus scipy's exact n_steps selection; it is
 * deliberately left unimplemented here (throws) so the finite path can land verified on its own.
 */
final class Nsum
{
    /**
     * @param callable(float): float $f     term function f(k)
     * @param float                  $a     first index
     * @param float|null             $b     last index (null/INF = infinite sum; not yet supported)
     * @param float                  $step  index step (default 1)
     * @return array{sum: float, error: float, nfev: int, success: bool, status: int}
     */
    public static function nsum(callable $f, float $a, ?float $b = null, float $step = 1.0): array
    {
        if ($b === null || is_infinite($b)) {
            throw new \RuntimeException('Tessero\Integrate\Nsum::nsum: infinite summation not yet implemented (finite limits only)');
        }
        if ($step <= 0.0) {
            throw new \RuntimeException('Tessero\Integrate\Nsum::nsum: step must be positive');
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
        $eps = 2.220446049250313e-16;
        return ['sum' => $s, 'error' => $eps * abs($s), 'nfev' => $n + 1, 'success' => true, 'status' => 0];
    }
}
