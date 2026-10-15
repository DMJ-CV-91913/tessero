<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate as Kernel;

/**
 * scipy.interpolate.LSQSphereBivariateSpline — weighted least-squares bivariate spline on the
 * surface of a sphere with GIVEN knots, fit by FITPACK sphere (iopt=-1), vendored f2c into the
 * Tessero kernel (interpolate.spherfit_lsq). theta (colatitude) in [0, pi], phi (longitude) in
 * [0, 2*pi]; tt, tp are strictly ordered interior knots (0 < tt[i] < pi, 0 < tp[i] < 2*pi). The
 * knots are padded to FITPACK form exactly as SciPy does, so the coefficients match bit-for-bit.
 */
final class LSQSphereBivariateSpline
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
     * @param list<float|int>      $tt    interior theta knots (0 < tt < pi), strictly increasing
     * @param list<float|int>      $tp    interior phi knots (0 < tp < 2*pi), strictly increasing
     * @param list<float|int>|null $w     weights (ones if null)
     * @param float                $eps   threshold in (0, 1) for rank determination
     */
    public function __construct(array $theta, array $phi, array $r, array $tt, array $tp, ?array $w = null, float $eps = 1e-16)
    {
        $tv = array_map('floatval', array_values($theta));
        $pv = array_map('floatval', array_values($phi));
        $rv = array_map('floatval', array_values($r));
        $m = count($tv);
        if ($m !== count($pv) || $m !== count($rv)) {
            throw new \ValueError('LSQSphereBivariateSpline: theta, phi, r must have the same length');
        }
        if (! ($eps > 0.0 && $eps < 1.0)) {
            throw new \ValueError('LSQSphereBivariateSpline: eps must be in (0, 1)');
        }
        $wv = $w === null ? array_fill(0, $m, 1.0) : array_map('floatval', array_values($w));
        if (count($wv) !== $m) {
            throw new \ValueError('LSQSphereBivariateSpline: w must have the same length as theta');
        }
        $ttPad = self::pad(array_map('floatval', array_values($tt)), M_PI);
        $tpPad = self::pad(array_map('floatval', array_values($tp)), 2.0 * M_PI);
        $res = Kernel::spherfitLsq($tv, $pv, $rv, $wv, $ttPad, $tpPad, $eps);
        $this->tt = self::flat($res['tt'] ?? $res[0]);
        $this->tp = self::flat($res['tp'] ?? $res[1]);
        $this->c = self::flat($res['c'] ?? $res[2]);
    }

    /**
     * Pad interior knots to the FITPACK vector [0,0,0,0, interior..., end,end,end,end] (SciPy's
     * nt_ = 8 + len(interior), tt_[4:-4] = interior, tt_[-4:] = end).
     *
     * @param list<float> $interior
     * @return list<float>
     */
    private static function pad(array $interior, float $end): array
    {
        $out = [0.0, 0.0, 0.0, 0.0];
        foreach ($interior as $k) {
            $out[] = $k;
        }
        $out[] = $end;
        $out[] = $end;
        $out[] = $end;
        $out[] = $end;
        return $out;
    }

    /** @return array{0: list<float>, 1: list<float>} the padded (tt, tp) knot vectors */
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
