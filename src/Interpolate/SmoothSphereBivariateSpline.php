<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate as Kernel;

/**
 * scipy.interpolate.SmoothSphereBivariateSpline — scattered-data smoothing spline on the
 * surface of a sphere, fit by FITPACK sphere (Dierckx), vendored f2c into the Tessero kernel
 * (interpolate.spherfit routine). The latitude/longitude knots (tt, tp) and coefficients c are
 * the exact FITPACK sphere iterative knot-insertion solve, matching SciPy bit-for-bit.
 *
 * theta (colatitude) in [0, pi], phi (longitude) in [0, 2*pi]; smoothing factor s >= 0.
 * Construction mirrors SciPy: get_knots() returns (tt, tp), get_coeffs() the raveled block.
 */
final class SmoothSphereBivariateSpline
{
    /** @var list<float> */
    private array $tt;
    /** @var list<float> */
    private array $tp;
    /** @var list<float> */
    private array $c;

    /**
     * @param list<float|int>      $theta colatitude in [0, pi], length m
     * @param list<float|int>      $phi   longitude in [0, 2*pi], length m
     * @param list<float|int>      $r     data values, length m
     * @param list<float|int>|null $w     weights (ones if null)
     * @param float                $s     smoothing factor (>= 0)
     * @param float                $eps   threshold in (0, 1) for rank determination
     */
    public function __construct(array $theta, array $phi, array $r, ?array $w = null, float $s = 0.0, float $eps = 1e-16)
    {
        $tv = array_map('floatval', array_values($theta));
        $pv = array_map('floatval', array_values($phi));
        $rv = array_map('floatval', array_values($r));
        $m = count($tv);
        if ($m !== count($pv) || $m !== count($rv)) {
            throw new \ValueError('SmoothSphereBivariateSpline: theta, phi, r must have the same length');
        }
        if ($s < 0.0) {
            throw new \ValueError('SmoothSphereBivariateSpline: s must be non-negative');
        }
        if (! ($eps > 0.0 && $eps < 1.0)) {
            throw new \ValueError('SmoothSphereBivariateSpline: eps must be in (0, 1)');
        }
        $wv = $w === null ? array_fill(0, $m, 1.0) : array_map('floatval', array_values($w));
        if (count($wv) !== $m) {
            throw new \ValueError('SmoothSphereBivariateSpline: w must have the same length as theta');
        }
        $res = Kernel::spherfit($tv, $pv, $rv, $wv, $s, $eps);
        $this->tt = self::flat($res['tt'] ?? $res[0]);
        $this->tp = self::flat($res['tp'] ?? $res[1]);
        $this->c = self::flat($res['c'] ?? $res[2]);
    }

    /** @return array{0: list<float>, 1: list<float>} the (tt, tp) knot vectors */
    public function getKnots(): array
    {
        return [$this->tt, $this->tp];
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
