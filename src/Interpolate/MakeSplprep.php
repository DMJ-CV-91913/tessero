<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate as InterpolateFacade;

/**
 * Parametric smoothing/interpolating B-spline curve with smoothing factor s = 0
 * (scipy.interpolate.make_splprep and the legacy scipy.interpolate.splprep, s=0 branch).
 *
 * For s = 0 (interpolation only) SciPy builds the default chord-length parametrization
 *     u[0] = 0,  u[i] = cumsum( || pts[i] - pts[i-1] || ) / total,
 * then returns make_interp_spline(u, pts.T, k=k, axis=1) -- i.e. ONE interpolating spline PER COORDINATE
 * component, all sharing the knots and the parameter u (see scipy/interpolate/_bsplines.py make_splprep:
 * "if s == 0: ... return make_interp_spline(u, x.T, k=k, axis=1), u"). The legacy FITPACK splprep(..., s=0)
 * produces the identical curve. This class therefore reuses the EXISTING
 * Tessero\Interpolate::makeInterpSpline kernel routine once per component -- a thin, backend-neutral pure-PHP
 * facade (no new kernel code). Verified machine-precision vs SciPy 1.17.1 (VERIFY_make_splprep.py).
 *
 * SCOPE: s = 0 only (s > 0 is the FITPACK/Dierckx smoothing fit, not implemented) and k in {1, 3} (the
 * make_interp_spline kernel cap). The parameter samples u[i] must be strictly increasing, which for the
 * chord-length default means consecutive points must be distinct.
 */
final class MakeSplprep
{
    /** @var list<list<float>> one strictly-ascending-indexed ordinate list per coordinate component */
    private array $comps;

    /** @var list<float> chord-length parameter values, strictly increasing in [0, 1] */
    private array $u;

    private int $k;

    /**
     * @param list<list<float|int>> $x list of coordinate component arrays (each length m); the curve is the
     *                                  set of points (x[0][i], x[1][i], ...) for i in 0..m-1
     * @param int                   $k spline degree: 1 or 3
     * @param float                 $s smoothing factor; only s = 0 (interpolation) is supported
     */
    public function __construct(array $x, int $k = 3, float $s = 0.0)
    {
        if ($s != 0.0) {
            throw new \ValueError(
                "MakeSplprep: only s=0 (interpolation) is supported; s={$s} needs the FITPACK smoothing fit, "
                . 'which Tessero does not implement.'
            );
        }
        if ($k !== 1 && $k !== 3) {
            throw new \ValueError(
                "MakeSplprep: only k=1 and k=3 are supported (Tessero make_interp_spline caps there); k={$k} "
                . 'is out of scope.'
            );
        }
        $comps = [];
        foreach (array_values($x) as $comp) {
            $comps[] = array_map('floatval', array_values($comp));
        }
        $ndim = count($comps);
        if ($ndim < 1) {
            throw new \ValueError('MakeSplprep: need at least one coordinate component');
        }
        $m = count($comps[0]);
        foreach ($comps as $comp) {
            if (count($comp) !== $m) {
                throw new \ValueError('MakeSplprep: all coordinate components must have the same length');
            }
        }
        if ($m < $k + 1) {
            throw new \ValueError("MakeSplprep: need at least k+1 (={$k}+1) points");
        }

        // Default chord-length parametrization: u[0]=0, u[i]=cumsum(||p[i]-p[i-1]||), then /= u[-1].
        $u = [0.0];
        $acc = 0.0;
        for ($i = 1; $i < $m; $i++) {
            $sq = 0.0;
            foreach ($comps as $comp) {
                $d = $comp[$i] - $comp[$i - 1];
                $sq += $d * $d;
            }
            $acc += sqrt($sq);
            $u[$i] = $acc;
        }
        $total = $u[$m - 1];
        if ($total <= 0.0) {
            throw new \ValueError('MakeSplprep: degenerate curve (zero total chord length)');
        }
        for ($i = 0; $i < $m; $i++) {
            $u[$i] /= $total;
        }
        for ($i = 1; $i < $m; $i++) {
            if ($u[$i] <= $u[$i - 1]) {
                throw new \ValueError('MakeSplprep: consecutive points must be distinct (u must strictly increase)');
            }
        }

        $this->comps = $comps;
        $this->u = $u;
        $this->k = $k;
    }

    /** The chord-length parameter values (scipy's returned `u`). @return list<float> */
    public function parameters(): array
    {
        return $this->u;
    }

    /**
     * Evaluate the parametric curve at parameter value(s) uq, returning one coordinate list per component.
     *
     * @param list<float|int> $uq query parameter values (in [0, 1])
     * @return list<list<float>> component-major: [ [comp0 values...], [comp1 values...], ... ]
     */
    public function evaluate(array $uq): array
    {
        $q = array_map('floatval', array_values($uq));
        if (count($q) === 0) {
            return array_fill(0, count($this->comps), []);
        }
        $out = [];
        foreach ($this->comps as $comp) {
            $res = InterpolateFacade::makeInterpSpline($this->u, $comp, $this->k, $q);
            $vals = is_object($res) && method_exists($res, 'toArray') ? $res->toArray() : $res;
            $flat = [];
            $arr = is_array($vals) ? $vals : [$vals];
            array_walk_recursive($arr, static function ($v) use (&$flat): void {
                $flat[] = (float) $v;
            });
            $out[] = $flat;
        }
        return $out;
    }

    /**
     * @param list<float|int> $uq
     * @return list<list<float>>
     */
    public function __invoke(array $uq): array
    {
        return $this->evaluate($uq);
    }
}
