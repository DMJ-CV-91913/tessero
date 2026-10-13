<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate as InterpolateFacade;

/**
 * 1-D interpolating spline through every data point (scipy.interpolate.InterpolatedUnivariateSpline).
 *
 * With the smoothing parameter s = 0 (the defining property of an InterpolatedUnivariateSpline), SciPy's FITPACK
 * interpolating spline is the same spline as make_interp_spline(x, y, k) with the default (interpolating) knot
 * pattern and boundary conditions: the curve passes through every (x[i], y[i]) and, for k=3, satisfies the
 * not-a-knot end conditions. This class therefore reuses the EXISTING Tessero\Interpolate::makeInterpSpline
 * kernel routine (interpolate.make_interp_spline) to both construct and evaluate the spline, so it is a thin,
 * backend-neutral pure-PHP facade (no new kernel code).
 *
 * SCOPE: only k = 1 (piecewise linear) and k = 3 (not-a-knot cubic) are supported, because the underlying
 * Tessero make_interp_spline kernel caps there (see csrc/src/scipy_interpolate.c r_make_interp_spline:
 * "only k=1 and k=3 are supported"). Degrees k = 2, 4, 5 require the general B-spline collocation solve
 * (banded system over the full FITPACK IUS knot vector) that Tessero does not yet expose, so they are OUT OF
 * SCOPE and rejected with a clear error rather than silently approximated.
 *
 * The data x must be strictly increasing, matching SciPy (which requires x strictly ascending for an IUS).
 */
final class InterpolatedUnivariateSpline
{
    /** @var list<float> strictly increasing sample abscissae */
    private array $x;

    /** @var list<float> sample ordinates, one per x */
    private array $y;

    private int $k;

    /**
     * @param list<float|int> $x data abscissae, strictly increasing (length >= k+1)
     * @param list<float|int> $y data ordinates, same length as $x
     * @param int             $k spline degree: 1 (linear) or 3 (cubic). k in {2,4,5} is out of scope.
     */
    public function __construct(array $x, array $y, int $k = 3)
    {
        if ($k !== 1 && $k !== 3) {
            throw new \ValueError(
                "InterpolatedUnivariateSpline: only k=1 and k=3 are supported (Tessero make_interp_spline caps "
                . "there); k={$k} needs the general B-spline collocation solve and is out of scope."
            );
        }
        $xv = array_map('floatval', array_values($x));
        $yv = array_map('floatval', array_values($y));
        $n = count($xv);
        if ($n !== count($yv)) {
            throw new \ValueError('InterpolatedUnivariateSpline: x and y must have the same length');
        }
        if ($n < $k + 1) {
            throw new \ValueError("InterpolatedUnivariateSpline: need at least k+1 (={$k}+1) data points");
        }
        for ($i = 1; $i < $n; $i++) {
            if ($xv[$i] <= $xv[$i - 1]) {
                throw new \ValueError('InterpolatedUnivariateSpline: x must be strictly increasing');
            }
        }
        $this->x = $xv;
        $this->y = $yv;
        $this->k = $k;
    }

    /**
     * Evaluate the interpolating spline at the query point(s).
     *
     * @param list<float|int> $xq query abscissae
     * @return list<float> spline value at each query point
     */
    public function evaluate(array $xq): array
    {
        $q = array_map('floatval', array_values($xq));
        if (count($q) === 0) {
            return [];
        }
        $res = InterpolateFacade::makeInterpSpline($this->x, $this->y, $this->k, $q);
        // makeInterpSpline returns a 1-D Tessero\NDArray; normalise to a flat list<float>.
        $out = is_object($res) && method_exists($res, 'toArray') ? $res->toArray() : $res;
        $flat = [];
        $arr = is_array($out) ? $out : [$out];
        array_walk_recursive($arr, static function ($v) use (&$flat): void {
            $flat[] = (float) $v;
        });
        return $flat;
    }

    /**
     * Callable form, matching scipy: InterpolatedUnivariateSpline(x, y, k)(xq).
     *
     * @param list<float|int> $xq
     * @return list<float>
     */
    public function __invoke(array $xq): array
    {
        return $this->evaluate($xq);
    }
}
