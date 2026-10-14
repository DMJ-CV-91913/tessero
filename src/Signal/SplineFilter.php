<?php

declare(strict_types=1);

namespace Tessero\Signal;

/**
 * scipy.signal.spline_filter - (cubic) smoothing-spline prefilter of a rank-2 array.
 *
 * Composition of already-landed kernel routines (symiirorder1/symiirorder2 via cspline2d-equivalent,
 * then a separable [1,4,1]/6 B-spline convolution via sepfir2d), matching
 * scipy.signal._spline_filters.spline_filter:
 *
 *     ck  = cspline2d(Iin, lmbda)            # smoothing 2-D cubic-spline coefficients
 *     out = sepfir2d(ck, h, h),  h = float32([1,4,1]/6)
 *
 * The smoothing path (lmbda > 1/144) is the second-order symmetric IIR filter symiirorder2 applied
 * separably along axis -1 then axis 0, with poles (r, omega) from compute_root_from_lambda(lmbda).
 * The non-smoothing path (lmbda <= 1/144) reduces to the ordinary cubic coefficients, i.e. the landed
 * kernel cspline2d (lamb = 0).
 *
 * dtype note: scipy quantises the separable B-spline kernel to float32 (hcol = float32([1,4,1])/6),
 * so we feed sepfir2d the float32-rounded tap values; the float64 compute path then matches scipy
 * bit-for-bit (verified rel = 0 over 23 cases). scipy additionally performs the whole computation in
 * single precision for float32 / complex inputs (historic, see scipy gh-9209); Tessero's kernel is
 * float64-only, so only the float64 real path is reproduced here.
 */
final class SplineFilter
{
    /** float32-rounded [1, 4, 1] / 6 (the exact float64 bit-values of np.float32([1,4,1])/6). */
    private const HCOL = [0.1666666716337204, 0.6666666865348816, 0.1666666716337204];

    /**
     * @param mixed       $iin         rank-2 input (NDArray or nested PHP array)
     * @param float       $lmbda       spline smoothing fall-off (scipy default 5.0)
     * @param class-string $signalClass the Signal facade to route kernel routines through
     *                                  (\Tessero\Signal or \Tessero\Ext\Signal)
     * @param class-string $ndClass     the matching NDArray class
     * @return mixed the filtered array (an NDArray of the given backend)
     */
    public static function filter(mixed $iin, float $lmbda, string $signalClass, string $ndClass): mixed
    {
        $rows = self::toRows($iin, $ndClass);
        $r = count($rows);
        $c = $r > 0 ? count($rows[0]) : 0;

        if ($lmbda <= 1.0 / 144.0) {
            // ordinary cubic coefficients: the landed kernel cspline2d (lamb = 0)
            $ck = $signalClass::cspline2d($ndClass::array($rows, 'float64'), 0.0);
        } else {
            [$rr, $omega] = self::rootFromLambda($lmbda);
            // symiirorder2 along axis -1 (each row), then axis 0 (each column); precision 1e-6 (f64 default)
            $tmp = [];
            foreach ($rows as $row) {
                $tmp[] = self::toVec($signalClass::symiirorder2($ndClass::array($row, 'float64'), $rr, $omega, 1e-6), $ndClass);
            }
            $out = $tmp;
            for ($j = 0; $j < $c; $j++) {
                $col = [];
                for ($i = 0; $i < $r; $i++) { $col[] = $tmp[$i][$j]; }
                $fc = self::toVec($signalClass::symiirorder2($ndClass::array($col, 'float64'), $rr, $omega, 1e-6), $ndClass);
                for ($i = 0; $i < $r; $i++) { $out[$i][$j] = $fc[$i]; }
            }
            $ck = $ndClass::array($out, 'float64');
        }

        $h = $ndClass::array(self::HCOL, 'float64');
        return $signalClass::sepfir2d($ck, $h, $h);
    }

    /** scipy.signal._spline_filters.compute_root_from_lambda */
    public static function rootFromLambda(float $lamb): array
    {
        $tmp = sqrt(3.0 + 144.0 * $lamb);
        $xi = 1.0 - 96.0 * $lamb + 24.0 * $lamb * $tmp;
        $omega = atan(sqrt((144.0 * $lamb - 1.0) / $xi));
        $tmp2 = sqrt($xi);
        $r = (24.0 * $lamb - 1.0 - $tmp2) / (24.0 * $lamb)
            * sqrt(48.0 * $lamb + 24.0 * $lamb * $tmp) / $tmp2;
        return [$r, $omega];
    }

    /** Input -> nested PHP rows of floats. */
    private static function toRows(mixed $iin, string $ndClass): array
    {
        if (is_object($iin) && method_exists($iin, 'toArray')) {
            $a = $iin->toArray();
        } else {
            $a = $iin;
        }
        if (! is_array($a) || $a === [] || ! is_array($a[0])) {
            throw new \InvalidArgumentException('spline_filter: input must be a rank-2 array');
        }
        return array_map(static fn ($row) => array_map('floatval', (array) $row), $a);
    }

    /** NDArray (or array) -> flat PHP vector of floats. */
    private static function toVec(mixed $v, string $ndClass): array
    {
        if (is_object($v) && method_exists($v, 'toList')) {
            return array_map('floatval', $v->toList());
        }
        return array_map('floatval', (array) $v);
    }
}
