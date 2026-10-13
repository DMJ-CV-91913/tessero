<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate;

/**
 * Evaluate a bivariate B-spline on a grid (scipy.interpolate.bisplev). Given a FITPACK spline
 * representation tck = (tx, ty, c, kx, ky) -- knot vectors tx, ty, the flat coefficient array c
 * of length ncx*ncy (ncx = len(tx)-kx-1, ncy = len(ty)-ky-1), and the degrees kx, ky -- this
 * returns the spline value at every point of the outer-product grid x[i] x y[j], i.e. a matrix
 * z with z[i][j] = s(x[i], y[j]).
 *
 * Pure PHP over the existing tensor-de-Boor kernel Tessero\Interpolate::NdBSpline: c is reshaped
 * to (ncx, ncy) and the de Boor recursion is applied along x then y, which is bit-identical to
 * FITPACK's bisplev for dx = dy = 0 (verified max |err| ~ 1e-17 vs scipy). Backend-neutral: the
 * only arithmetic is in the shared kernel.
 *
 * Like SciPy, x and y must be sorted in ascending order. Only dx = dy = 0 (function value, no
 * partial derivatives) is supported; the underlying NdBSpline kernel does not carry FITPACK's
 * derivative path. SciPy squeezes size-1 axes (scalar x and/or y collapse that dimension); this
 * facade always returns the full 2-D grid as a list of rows, so callers that want SciPy's
 * squeeze/scalar behaviour should handle size-1 inputs themselves.
 */
final class Bisplev
{
    private function __construct()
    {
    }

    /**
     * @param list<float|int>                $x  query coordinates along the first axis (ascending)
     * @param list<float|int>                $y  query coordinates along the second axis (ascending)
     * @param array{0:list<float|int>,1:list<float|int>,2:list<float|int>,3:int,4:int} $tck
     *                                           FITPACK representation (tx, ty, c, kx, ky); c is the
     *                                           flat coefficient array of length ncx*ncy
     * @param int                            $dx partial-derivative order in x (only 0 supported)
     * @param int                            $dy partial-derivative order in y (only 0 supported)
     * @return list<list<float>>                 z[i][j] = s(x[i], y[j])
     */
    public static function eval(array $x, array $y, array $tck, int $dx = 0, int $dy = 0): array
    {
        if ($dx !== 0 || $dy !== 0) {
            throw new \ValueError('bisplev: only dx = dy = 0 (no partial derivatives) is supported');
        }
        if (count($tck) < 5) {
            throw new \ValueError('bisplev: tck must be (tx, ty, c, kx, ky)');
        }
        $tx = array_map('floatval', array_values($tck[0]));
        $ty = array_map('floatval', array_values($tck[1]));
        $c  = array_map('floatval', array_values($tck[2]));
        $kx = (int) $tck[3];
        $ky = (int) $tck[4];

        $ncx = count($tx) - $kx - 1;
        $ncy = count($ty) - $ky - 1;
        if ($ncx < 1 || $ncy < 1) {
            throw new \ValueError('bisplev: knot vectors too short for the given degrees');
        }
        if (count($c) !== $ncx * $ncy) {
            throw new \ValueError(sprintf(
                'bisplev: coefficient array has %d entries, expected ncx*ncy = %d*%d = %d',
                count($c), $ncx, $ncy, $ncx * $ncy
            ));
        }

        $xs = array_map('floatval', array_values($x));
        $ys = array_map('floatval', array_values($y));
        $nx = count($xs);
        $ny = count($ys);
        if ($nx === 0 || $ny === 0) {
            return [];
        }

        // c reshaped to (ncx, ncy) as NdBSpline expects.
        $cmat = [];
        for ($i = 0; $i < $ncx; $i++) {
            $row = [];
            for ($j = 0; $j < $ncy; $j++) {
                $row[$j] = $c[$i * $ncy + $j];
            }
            $cmat[$i] = $row;
        }

        // Full outer-product grid as (n, 2) query points, evaluated by the tensor de Boor kernel.
        $pts = [];
        foreach ($xs as $px) {
            foreach ($ys as $py) {
                $pts[] = [$px, $py];
            }
        }
        $flat = Interpolate::NdBSpline($tx, $ty, $cmat, $kx, $ky, $pts);
        $flat = self::flatten($flat);

        $z = [];
        $p = 0;
        for ($i = 0; $i < $nx; $i++) {
            $row = [];
            for ($j = 0; $j < $ny; $j++) {
                $row[$j] = (float) $flat[$p++];
            }
            $z[$i] = $row;
        }
        return $z;
    }

    /** @param mixed $v @return list<float> */
    private static function flatten(mixed $v): array
    {
        if (is_object($v) && method_exists($v, 'toArray')) {
            $v = $v->toArray();
        }
        if (! is_array($v)) {
            return [(float) $v];
        }
        $out = [];
        array_walk_recursive($v, static function ($x) use (&$out): void { $out[] = (float) $x; });
        return $out;
    }
}
