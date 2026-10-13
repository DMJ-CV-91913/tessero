<?php

declare(strict_types=1);

namespace Tessero\Integrate;

/**
 * Tanh-sinh (double-exponential) quadrature over a finite interval (scipy.integrate.tanhsinh, scalar, log=False).
 *
 * The change of variable x = (a+b)/2 + (b-a)/2 * tanh(pi/2 * sinh(t)) maps t in (-inf, inf) onto (a, b); the
 * integrand's endpoint behaviour is smoothed (nodes crowd towards the endpoints double-exponentially), so this
 * rule is accurate even for endpoint-singular integrands such as 1/sqrt(x) or log(x). The trapezoid rule in t is
 * evaluated at step h and the step is halved (level-doubling) until a scipy-style error estimate falls below the
 * tolerance. Pure PHP: calls the user's callable, no kernel. The returned ->integral matches scipy's value to the
 * requested tolerance; the error/status fields are Tessero's own and track scipy's semantics but are not compared.
 *
 * Mirrors scipy's abscissa-complement formulation (Takahasi-Mori / van Engelen): rather than forming the node x
 * directly, the *complement* xjc = 1 - tanh(u) = 1/(exp(u)cosh(u)) is carried so the two endpoints stay accurate
 * to full precision, and endpoints are mapped as b - alpha*xjc (right) and a + alpha*xjc (left).
 */
final class Tanhsinh
{
    private const N_BASE_STEPS = 8;

    /**
     * Integrate $f over the finite interval [$a, $b].
     *
     * @param callable(float): float $f
     * @return array{integral: float, error: float, success: bool, status: int, maxlevel: int, nfev: int}
     */
    public static function tanhsinh(
        callable $f,
        float $a,
        float $b,
        int $maxlevel = 10,
        int $minlevel = 2,
        float $atol = 0.0,
        ?float $rtol = null
    ): array {
        $eps = 2.220446049250313e-16;                 // double precision machine epsilon (2^-52)
        if ($rtol === null) {
            $rtol = $eps ** 0.75;                      // scipy's default relative tolerance
        }

        if ($a === $b) {
            return ['integral' => 0.0, 'error' => 0.0, 'success' => true, 'status' => 0, 'maxlevel' => -1, 'nfev' => 1];
        }

        $h0 = self::baseStep();
        $alpha = ($b - $a) / 2.0;

        // Incumbent extreme-abscissa trackers (for the d4 error term and singular-node replacement).
        $xr0 = -INF; $fr0 = NAN; $wr0 = 0.0;           // right side: track the largest valid abscissa
        $xl0 = INF;  $fl0 = NAN; $wl0 = 0.0;           // left side:  track the smallest valid abscissa

        $cumFjwj = [];           // every fj*wj product seen so far (for the d3 term: eps * max|fj wj|)
        $maxAbsFjwj = 0.0;
        $sk = [];                // integral estimate at each completed level (level-doubling history)
        $sn = 0.0;
        $aerr = NAN;
        $rerr = NAN;
        $nfev = 1;
        $status = -2;            // -2: in progress / not converged, 0: converged, -3: non-finite
        $converged = false;
        $finalLevel = -1;

        for ($n = $minlevel; $n <= $maxlevel; $n++) {
            $h = $h0 / (2 ** $n);
            $inclusive = ($n === $minlevel);
            // On the first (jump-started) level evaluate every sub-level 0..n; afterwards only the new odd nodes.
            $levels = $inclusive ? range(0, $n) : [$n];

            $sumLevel = 0.0;      // sum of fj*wj for the nodes evaluated at this level (both sides)
            $d4 = 0.0;

            foreach ($levels as $k) {
                [$js, $jstep, $jmax] = self::levelIndices($k);
                $hk = $h0 / (2 ** $k);
                for ($j = $js; $j <= $jmax; $j += $jstep) {
                    $jh = $j * $hk;
                    $u1 = (M_PI / 2.0) * cosh($jh);
                    $u2 = (M_PI / 2.0) * sinh($jh);
                    $chU2 = cosh($u2);
                    $wj = $u1 / ($chU2 * $chU2);
                    // xjc = 1 - tanh(u2) = 1/(exp(u2) cosh(u2)); the abscissa complement, accurate near the endpoint.
                    $xjc = 1.0 / (exp($u2) * $chU2);
                    if ($k === 0 && $j === 0) {
                        $wj *= 0.5;                    // the central node is shared between the two half-grids
                    }
                    $wjt = $wj * $alpha;

                    // Right node:  b - alpha*xjc  (approaches b as xjc -> 0)
                    $xr = -$alpha * $xjc + $b;
                    // Left node:   a + alpha*xjc  (approaches a as xjc -> 0)
                    $xl = $alpha * $xjc + $a;

                    $wr = ($xr <= $a || $xr >= $b) ? 0.0 : $wjt;
                    $wl = ($xl <= $a || $xl >= $b) ? 0.0 : $wjt;

                    $fr = ($wr !== 0.0) ? (float) $f($xr) : NAN;
                    $fl = ($wl !== 0.0) ? (float) $f($xl) : NAN;
                    $nfev += 1;

                    // Track the extreme valid abscissa on each side (nearest the endpoint); used for d4 and to
                    // replace a non-finite interior sample at a singular endpoint with the nearest valid value.
                    $invalidR = ($wr === 0.0) || ! is_finite($fr);
                    if (! $invalidR && $xr > $xr0) {
                        $xr0 = $xr; $fr0 = $fr; $wr0 = $wr;
                    }
                    $invalidL = ($wl === 0.0) || ! is_finite($fl);
                    if (! $invalidL && $xl < $xl0) {
                        $xl0 = $xl; $fl0 = $fl; $wl0 = $wl;
                    }

                    // Replace invalid interior values with the nearest-extreme incumbent (singularity handling).
                    $frUse = $invalidR ? $fr0 : $fr;
                    $flUse = $invalidL ? $fl0 : $fl;

                    $fjwjR = ($wr === 0.0) ? 0.0 : $frUse * $wr;
                    $fjwjL = ($wl === 0.0) ? 0.0 : $flUse * $wl;

                    if (is_finite($fjwjR)) {
                        $cumFjwj[] = $fjwjR;
                        $aR = abs($fjwjR);
                        if ($aR > $maxAbsFjwj) { $maxAbsFjwj = $aR; }
                        $sumLevel += $fjwjR;
                    }
                    if (is_finite($fjwjL)) {
                        $cumFjwj[] = $fjwjL;
                        $aL = abs($fjwjL);
                        if ($aL > $maxAbsFjwj) { $maxAbsFjwj = $aL; }
                        $sumLevel += $fjwjL;
                    }
                }
            }

            $frwr0 = (is_finite($fr0) ? $fr0 : 0.0) * $wr0;
            $flwl0 = (is_finite($fl0) ? $fl0 : 0.0) * $wl0;
            $d4 = max(abs($flwl0), abs($frwr0));

            $snLevel = $sumLevel * $h;
            if ($sk !== []) {
                // Level-doubling recurrence: halve the previous estimate and add the fresh odd-node contribution.
                $sn = end($sk) / 2.0 + $snLevel;
            } else {
                $sn = $snLevel;
            }
            $sk[] = $sn;

            // scipy-style error estimate once at least three level estimates exist.
            if (count($sk) >= 3) {
                $snm1 = $sk[count($sk) - 2];
                $snm2 = $sk[count($sk) - 3];
                $d1 = abs($sn - $snm1);
                $d2 = abs($sn - $snm2);
                $d3 = $eps * $maxAbsFjwj;
                $d5 = $eps * abs($sn);
                $temp = ($d1 > 0.0 && $d2 > 0.0) ? $d1 ** (log($d1) / log($d2)) : 0.0;
                $aerr = max($temp, $d1 * $d1, $d3, $d4);
                $aerr = min(max($aerr, $d5), $d1);
                $rerr = ($sn !== 0.0) ? $aerr / abs($sn) : INF;
            }

            $finalLevel = $n;
            if (! is_finite($sn)) {
                $status = -3;
                break;
            }
            if (! is_nan($rerr) && ($rerr < $rtol || $aerr < $atol)) {
                $status = 0;
                $converged = true;
                break;
            }
        }

        if (! $converged && $status !== -3) {
            $status = is_finite($sn) ? -2 : -3;
        }

        return [
            'integral' => $sn,
            'error' => $aerr,
            'success' => ($status === 0),
            'status' => $status,
            'maxlevel' => $finalLevel,
            'nfev' => $nfev,
        ];
    }

    /**
     * Base step h0 in t. Chosen (scipy / Takahasi-Mori Eq.13) so the most extreme abscissa complement is just
     * above the smallest representable positive value: tmax = asinh(log(2/fmin - 1) / pi), h0 = tmax / N_BASE_STEPS,
     * where fmin = 4 * smallest normal double.
     */
    private static function baseStep(): float
    {
        $fmin = 4.0 * 2.2250738585072014e-308;        // 4 * DBL_MIN (smallest normal double)
        $tmax = asinh(log(2.0 / $fmin - 1.0) / M_PI);
        return $tmax / self::N_BASE_STEPS;
    }

    /**
     * Node index range for sub-level $k. Level 0 uses j = 0,1,...,mx (every node); deeper levels contribute only
     * the new odd nodes j = 1,3,...,mx (the even ones were already summed at the coarser level).
     *
     * @return array{0: int, 1: int, 2: int}  [start, step, max]
     */
    private static function levelIndices(int $k): array
    {
        $mx = self::N_BASE_STEPS * (2 ** $k);
        if ($k === 0) {
            return [0, 1, $mx];
        }
        return [1, 2, $mx];
    }
}
