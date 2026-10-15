<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate as Kernel;

/**
 * scipy.interpolate.RectSphereBivariateSpline — bivariate smoothing spline over a rectangular
 * mesh on the surface of a sphere, fit by FITPACK spgrid (Dierckx), vendored f2c into the Tessero
 * kernel (interpolate.spgrid). u (colatitude) is a strictly increasing grid in (0, pi); v
 * (longitude) a strictly increasing grid with v[-1] <= v[0] + 2*pi; r the (len(u) x len(v)) grid of
 * values. Smoothing factor s >= 0 (s = 0 interpolates). Poles are left free (SciPy's default
 * iopt=[0,0,0], ider=[-1,0,-1,0]); knots/coefficients match SciPy bit-for-bit.
 */
final class RectSphereBivariateSpline
{
    /** @var list<float> */
    private array $tu;
    /** @var list<float> */
    private array $tv;
    /** @var list<float> */
    private array $c;

    /**
     * @param list<float|int>                   $u colatitude grid in (0, pi), strictly increasing, length mu
     * @param list<float|int>                   $v longitude grid, strictly increasing, length mv
     * @param list<list<float|int>>|list<float> $r (mu x mv) grid of values (2-D) or its C-order ravel
     * @param float                             $s smoothing factor (>= 0)
     */
    public function __construct(array $u, array $v, array $r, float $s = 0.0)
    {
        $uv = array_map('floatval', array_values($u));
        $vv = array_map('floatval', array_values($v));
        $mu = count($uv);
        $mv = count($vv);
        if ($s < 0.0) {
            throw new \ValueError('RectSphereBivariateSpline: s must be non-negative');
        }
        $rflat = self::flat($r);
        if (count($rflat) !== $mu * $mv) {
            throw new \ValueError('RectSphereBivariateSpline: r must have len(u)*len(v) elements');
        }
        $res = Kernel::spgrid($uv, $vv, $rflat, $s);
        $this->tu = self::flat($res['tu'] ?? $res[0]);
        $this->tv = self::flat($res['tv'] ?? $res[1]);
        $this->c = self::flat($res['c'] ?? $res[2]);
    }

    /** @return array{0: list<float>, 1: list<float>} the (tu, tv) knot vectors */
    public function getKnots(): array
    {
        return [$this->tu, $this->tv];
    }

    /** @return list<float> */
    public function getCoeffs(): array
    {
        return $this->c;
    }

    /** @param mixed $v @return list<float> */
    private static function flat($v): array
    {
        if (is_object($v) && method_exists($v, 'toList')) {
            $v = $v->toList();
        } elseif (is_object($v) && method_exists($v, 'toArray')) {
            $v = $v->toArray();
        }
        $out = [];
        array_walk_recursive($v, static function ($x) use (&$out): void {
            $out[] = (float) $x;
        });
        return $out;
    }
}
