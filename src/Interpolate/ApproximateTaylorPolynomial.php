<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

/**
 * Estimate the Taylor polynomial of a function f at a point x by polynomial interpolation
 * (scipy.interpolate.approximate_taylor_polynomial).
 *
 * Faithful reimplementation of SciPy's algorithm (scipy/interpolate/_interpolate.py):
 *   order = degree if order is None
 *   n  = order + 1
 *   xs = scale * cos(linspace(0, pi, n, endpoint=False)) + x        # Chebyshev-like nodes, one at x
 *   P  = KroghInterpolator(xs, f(xs))                               # unique degree n-1 interpolant
 *   d  = P.derivatives(x, der=degree+1)                             # f^(0..degree) of the interpolant at x
 *   return poly1d( (d / factorial(0..degree))[::-1] )               # Taylor coeffs, highest power first
 *
 * Because the n nodes are distinct, the Krogh interpolant is just the unique Newton divided-difference
 * polynomial. This routine builds that polynomial's Newton divided differences and re-expands it about x into
 * the shifted power basis t = (X - x); the coefficient of t^d is f^(d)(x)/d!, which is exactly
 * (d / factorial)[::-1] reversed. No kernel code, no RNG -- fully deterministic and backend-neutral.
 * Verified machine-precision vs SciPy 1.17.1 (VERIFY_approximate_taylor_polynomial.py / VERIFY_atp2.py).
 *
 * f is supplied as a PHP callable taking float -> float and is evaluated at the n node abscissae.
 *
 * @return list<float> polynomial coefficients, highest power first (poly1d order), length degree+1
 */
final class ApproximateTaylorPolynomial
{
    /**
     * @param callable(float):float $f      function to approximate (evaluated at the node abscissae)
     * @param float                 $x      expansion point
     * @param int                   $degree degree of the returned Taylor polynomial (>= 0)
     * @param float                 $scale  width of the fitting interval
     * @param int|null              $order  fitting polynomial order (defaults to degree)
     * @return list<float> coefficients highest-power-first, length degree+1
     */
    public static function compute(callable $f, float $x, int $degree, float $scale, ?int $order = null): array
    {
        if ($degree < 0) {
            throw new \ValueError("ApproximateTaylorPolynomial: degree must be >= 0, got {$degree}");
        }
        $order = $order ?? $degree;
        if ($order < $degree) {
            throw new \ValueError('ApproximateTaylorPolynomial: order must be >= degree');
        }
        $n = $order + 1;

        // xs = scale*cos(linspace(0,pi,n,endpoint=False)) + x   (numpy linspace with endpoint=False)
        $xs = [];
        $ys = [];
        for ($i = 0; $i < $n; $i++) {
            $theta = ($n > 1) ? (M_PI * $i / $n) : 0.0;   // step = pi/n for endpoint=False
            $xi = $scale * cos($theta) + $x;
            $xs[$i] = $xi;
            $ys[$i] = (float) $f($xi);
        }

        // Newton divided differences c[k] of the data (xs, ys).
        $c = $ys;
        for ($j = 1; $j < $n; $j++) {
            for ($i = $n - 1; $i >= $j; $i--) {
                $c[$i] = ($c[$i] - $c[$i - 1]) / ($xs[$i] - $xs[$i - $j]);
            }
        }

        // Re-expand the Newton polynomial about x into the power basis t=(X-x):
        //   p(X) = sum_k c[k] * prod_{i<k} (X - xs[i])
        // Maintain the running product in the t-basis and accumulate Taylor coefficients tc[d].
        $prod = array_fill(0, $n, 0.0);
        $prod[0] = 1.0;
        $tc = array_fill(0, $n, 0.0);
        for ($k = 0; $k < $n; $k++) {
            $ck = $c[$k];
            for ($d = 0; $d < $n; $d++) {
                $tc[$d] += $ck * $prod[$d];
            }
            if ($k < $n - 1) {
                // multiply prod by (t + (x - xs[k]))
                $a = $x - $xs[$k];
                $newp = array_fill(0, $n, 0.0);
                for ($d = 0; $d < $n; $d++) {
                    $newp[$d] += $a * $prod[$d];          // a * prod
                    if ($d + 1 < $n) {
                        $newp[$d + 1] += $prod[$d];       // t * prod (shift up)
                    }
                }
                $prod = $newp;
            }
        }

        // tc[d] = f^(d)(x)/d! ; poly1d coeffs = tc[degree], tc[degree-1], ..., tc[0]  (highest power first).
        $coeffs = [];
        for ($d = $degree; $d >= 0; $d--) {
            $coeffs[] = $tc[$d];
        }
        return $coeffs;
    }
}
