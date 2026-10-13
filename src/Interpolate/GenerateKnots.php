<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

/**
 * Knot vector for a spline of degree k through data x with smoothing factor s = 0
 * (scipy.interpolate.generate_knots, the s=0 / interpolation branch).
 *
 * For s = 0 (non-periodic) SciPy's generate_knots yields exactly one knot vector, the not-a-knot vector
 * produced by _not_a_knot(x, k) (scipy/interpolate/_bsplines.py). That construction is a closed-form, purely
 * numeric formula -- no FITPACK, no spline solve -- so this is a self-contained backend-neutral pure-PHP
 * routine valid for every degree k >= 1:
 *
 *   if k is odd:  k2 = (k+1)/2 ; t = x
 *   if k is even: k2 = k/2     ; t = (x[1:] + x[:-1]) / 2   (Greville midpoints)
 *   t = t[k2 : len(t)-k2]
 *   knots = [x[0]]*(k+1) ++ t ++ [x[-1]]*(k+1)
 *
 * SCOPE: s = 0 only. s > 0 is the Dierckx iterative knot-insertion generator (_generate_knots_impl), which
 * Tessero does not implement, and is rejected. x must be strictly increasing with len(x) >= k+1.
 */
final class GenerateKnots
{
    /**
     * Return the not-a-knot knot vector (the single vector generate_knots(x, y, k=k, s=0) yields).
     *
     * @param list<float|int> $x data abscissae, strictly increasing (length >= k+1)
     * @param int             $k spline degree (>= 1)
     * @param float           $s smoothing factor; only s = 0 is supported
     * @return list<float> the knot vector
     */
    public static function knots(array $x, int $k = 3, float $s = 0.0): array
    {
        if ($s != 0.0) {
            throw new \ValueError(
                "GenerateKnots: only s=0 (interpolation) is supported; s={$s} needs the Dierckx iterative "
                . 'knot-insertion generator, which Tessero does not implement.'
            );
        }
        if ($k < 1) {
            throw new \ValueError("GenerateKnots: degree k must be >= 1, got k={$k}");
        }
        $xv = array_map('floatval', array_values($x));
        $n = count($xv);
        if ($n < $k + 1) {
            throw new \ValueError("GenerateKnots: need at least k+1 (={$k}+1) data points");
        }
        for ($i = 1; $i < $n; $i++) {
            if ($xv[$i] <= $xv[$i - 1]) {
                throw new \ValueError('GenerateKnots: x must be strictly increasing');
            }
        }

        if ($k % 2 === 1) {
            $k2 = intdiv($k + 1, 2);
            $t = $xv;                              // t = x.copy()
        } else {
            $k2 = intdiv($k, 2);
            $t = [];                               // t = (x[1:] + x[:-1]) / 2
            for ($i = 1; $i < $n; $i++) {
                $t[] = ($xv[$i] + $xv[$i - 1]) / 2.0;
            }
        }
        // t = t[k2 : len(t)-k2]
        $t = array_slice($t, $k2, count($t) - 2 * $k2);

        $first = $xv[0];
        $last = $xv[$n - 1];
        $knots = [];
        for ($i = 0; $i <= $k; $i++) {             // (x[0],)*(k+1)
            $knots[] = $first;
        }
        foreach ($t as $v) {
            $knots[] = $v;
        }
        for ($i = 0; $i <= $k; $i++) {             // (x[-1],)*(k+1)
            $knots[] = $last;
        }
        return $knots;
    }
}
