<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate as Kernel;

/**
 * scipy.interpolate.UnivariateSpline — a 1-D smoothing spline fit by FITPACK curfit
 * (Dierckx), vendored f2c into the Tessero kernel (interpolate.splrep routine).
 *
 * UnivariateSpline(x, y, w=None, k=3, s=None): weighted least-squares spline with the
 * smoothing factor s. When s is None SciPy uses the default s = m - sqrt(2*m) (m = #points);
 * w defaults to all-ones. The fit (knots t, coefficients c) is computed by the exact FITPACK
 * curfit iterative knot-insertion solve, so t/c and the evaluated spline match SciPy bit-for-bit.
 * __invoke($xq) evaluates via FITPACK splev.
 */
final class UnivariateSpline
{
    /** @var list<float> */
    private array $t;
    /** @var list<float> */
    private array $c;
    private int $k;

    /**
     * @param list<float|int>      $x strictly increasing abscissae (length >= k+1)
     * @param list<float|int>      $y ordinates, same length as x
     * @param list<float|int>|null $w weights (ones if null)
     * @param int                  $k spline degree 1..5
     * @param float|null           $s smoothing factor; null => m - sqrt(2m)
     */
    public function __construct(array $x, array $y, ?array $w = null, int $k = 3, ?float $s = null)
    {
        $xv = array_map("floatval", array_values($x));
        $yv = array_map("floatval", array_values($y));
        $m = count($xv);
        if ($m !== count($yv)) {
            throw new \ValueError("UnivariateSpline: x and y must have the same length");
        }
        if ($k < 1 || $k > 5) {
            throw new \ValueError("UnivariateSpline: k must be in 1..5");
        }
        if ($m < $k + 1) {
            throw new \ValueError("UnivariateSpline: need at least k+1 points");
        }
        $wv = $w === null ? array_fill(0, $m, 1.0) : array_map("floatval", array_values($w));
        if (count($wv) !== $m) {
            throw new \ValueError("UnivariateSpline: w must have the same length as x");
        }
        $sv = $s === null ? (float) $m - sqrt(2.0 * $m) : (float) $s;

        $r = Kernel::splrep($xv, $yv, $wv, $k, $sv);
        $t = $r["t"] ?? $r[0];
        $c = $r["c"] ?? $r[1];
        $this->t = self::flat($t);
        $this->c = self::flat($c);
        $this->k = $k;
    }

    /** @param list<float|int> $xq @return list<float> */
    public function evaluate(array $xq): array
    {
        $q = array_map("floatval", array_values($xq));
        if (count($q) === 0) {
            return [];
        }
        $v = Kernel::splev($this->t, $this->c, $this->k, $q);
        return self::flat($v);
    }

    /** @param list<float|int> $xq @return list<float> */
    public function __invoke(array $xq): array
    {
        return $this->evaluate($xq);
    }

    /** @return list<float> the full knot vector */
    public function getKnots(): array
    {
        return $this->t;
    }

    /** @return list<float> the spline coefficients */
    public function getCoeffs(): array
    {
        return $this->c;
    }

    /** @param mixed $v @return list<float> */
    private static function flat($v): array
    {
        $out = is_object($v) && method_exists($v, "toArray") ? $v->toArray() : $v;
        $arr = is_array($out) ? $out : [$out];
        $flat = [];
        array_walk_recursive($arr, static function ($z) use (&$flat): void {
            $flat[] = (float) $z;
        });
        return $flat;
    }
}
