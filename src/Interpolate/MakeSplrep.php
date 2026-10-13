<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate as InterpolateFacade;

/**
 * Smoothing/interpolating 1-D B-spline representation with the smoothing factor s = 0
 * (scipy.interpolate.make_splrep and the legacy scipy.interpolate.splrep, s=0 branch).
 *
 * For s = 0 (interpolation only), SciPy's make_splrep returns exactly make_interp_spline(x, y, k, bc_type)
 * (see scipy/interpolate/_bsplines.py make_splrep: "if s == 0: ... return make_interp_spline(...)"), and the
 * legacy FITPACK splrep(x, y, k, s=0) produces the SAME curve: its knot vector is the not-a-knot vector and
 * its coefficients coincide with make_interp_spline's to machine precision (verified in VERIFY_make_splrep.py
 * and VERIFY_splrep_s0 against SciPy 1.17.1). This class therefore reuses the EXISTING
 * Tessero\Interpolate::makeInterpSpline kernel routine (interpolate.make_interp_spline) to construct and
 * evaluate the spline -- a thin, backend-neutral pure-PHP facade (no new kernel code).
 *
 * SCOPE: only s = 0 and k in {1 (linear), 3 (not-a-knot cubic)}. s > 0 is the FITPACK smoothing fit
 * (curfit / the Dierckx iterative knot-insertion solve) which Tessero does not implement; k in {2,4,5} need
 * the general B-spline collocation solve. Both are rejected with a clear error rather than silently
 * approximated. x must be strictly increasing (SciPy requires strictly ascending x for s=0).
 */
final class MakeSplrep
{
    /** @var list<float> strictly increasing sample abscissae */
    private array $x;

    /** @var list<float> sample ordinates, one per x */
    private array $y;

    private int $k;

    /**
     * @param list<float|int> $x data abscissae, strictly increasing (length >= k+1)
     * @param list<float|int> $y data ordinates, same length as $x
     * @param int             $k spline degree: 1 or 3
     * @param float           $s smoothing factor; only s = 0 (interpolation) is supported
     */
    public function __construct(array $x, array $y, int $k = 3, float $s = 0.0)
    {
        if ($s != 0.0) {
            throw new \ValueError(
                "MakeSplrep: only s=0 (interpolation) is supported; s={$s} needs the FITPACK smoothing fit "
                . '(curfit), which Tessero does not implement.'
            );
        }
        if ($k !== 1 && $k !== 3) {
            throw new \ValueError(
                "MakeSplrep: only k=1 and k=3 are supported (Tessero make_interp_spline caps there); "
                . "k={$k} needs the general B-spline collocation solve and is out of scope."
            );
        }
        $xv = array_map('floatval', array_values($x));
        $yv = array_map('floatval', array_values($y));
        $n = count($xv);
        if ($n !== count($yv)) {
            throw new \ValueError('MakeSplrep: x and y must have the same length');
        }
        if ($n < $k + 1) {
            throw new \ValueError("MakeSplrep: need at least k+1 (={$k}+1) data points");
        }
        for ($i = 1; $i < $n; $i++) {
            if ($xv[$i] <= $xv[$i - 1]) {
                throw new \ValueError('MakeSplrep: x must be strictly increasing');
            }
        }
        $this->x = $xv;
        $this->y = $yv;
        $this->k = $k;
    }

    /**
     * Evaluate the spline at the query point(s).
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
        $out = is_object($res) && method_exists($res, 'toArray') ? $res->toArray() : $res;
        $flat = [];
        $arr = is_array($out) ? $out : [$out];
        array_walk_recursive($arr, static function ($v) use (&$flat): void {
            $flat[] = (float) $v;
        });
        return $flat;
    }

    /**
     * Callable form, matching scipy: make_splrep(x, y, k=k, s=0)(xq).
     *
     * @param list<float|int> $xq
     * @return list<float>
     */
    public function __invoke(array $xq): array
    {
        return $this->evaluate($xq);
    }
}
