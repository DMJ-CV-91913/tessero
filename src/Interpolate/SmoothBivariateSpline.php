<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

/**
 * scipy.interpolate.bisplrep + SmoothBivariateSpline — scattered-data bivariate smoothing
 * spline fit by FITPACK surfit (Dierckx), vendored f2c into the Tessero kernel
 * (interpolate.bisplrep routine). The knots (tx, ty) and coefficients c are the exact
 * FITPACK surfit iterative knot-insertion solve, matching SciPy bit-for-bit; construction,
 * evaluation (.ev) and get_knots/get_coeffs are inherited from BivariateSpline. This class
 * adds the bisplrep() helper returning the (tx, ty, c, kx, ky) tck tuple.
 */
final class SmoothBivariateSpline extends BivariateSpline
{
    /**
     * scipy.interpolate.SmoothBivariateSpline(x, y, z, w=None, kx=3, ky=3, s=None) — fit the
     * scattered-data smoothing spline (FITPACK surfit). Construction/evaluation live on the
     * BivariateSpline base; this override exists so the smoothing constructor is the class's
     * own public entry point.
     *
     * @param list<float|int>      $x
     * @param list<float|int>      $y
     * @param list<float|int>      $z
     * @param list<float|int>|null $w
     */
    public function __construct(array $x, array $y, array $z, ?array $w = null, int $kx = 3, int $ky = 3, ?float $s = null)
    {
        parent::__construct($x, $y, $z, $w, $kx, $ky, $s);
    }

    /**
     * scipy.interpolate.bisplrep(x, y, z, w=None, kx=3, ky=3, s=None) -> [tx, ty, c, kx, ky].
     *
     * @param list<float|int>      $x
     * @param list<float|int>      $y
     * @param list<float|int>      $z
     * @param list<float|int>|null $w
     * @return array{0: list<float>, 1: list<float>, 2: list<float>, 3: int, 4: int}
     */
    public static function bisplrep(array $x, array $y, array $z, ?array $w = null, int $kx = 3, int $ky = 3, ?float $s = null): array
    {
        $sp = new self($x, $y, $z, $w, $kx, $ky, $s);
        [$tx, $ty] = $sp->getKnots();
        return [$tx, $ty, $sp->getCoeffs(), $kx, $ky];
    }
}
