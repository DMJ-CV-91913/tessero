<?php

declare(strict_types=1);

namespace Tessero\Interpolate;

use Tessero\Interpolate as Kernel;

/**
 * scipy.interpolate.BivariateSpline — base class for bivariate splines in the FITPACK
 * (tx, ty, c, kx, ky) tensor-product representation. A scattered-data smoothing fit is
 * performed by the f2c-vendored Dierckx surfit_ routine (interpolate.bisplrep), so the
 * knots and coefficients match SciPy bit-for-bit; evaluation (.ev / get_knots / get_coeffs)
 * delegates to the tensor de Boor kernel (NdBSpline). Concrete fits such as
 * SmoothBivariateSpline extend this class.
 */
class BivariateSpline
{
    /** @var list<float> */
    protected array $tx;
    /** @var list<float> */
    protected array $ty;
    /** @var list<float> */
    protected array $c;
    protected int $kx;
    protected int $ky;

    /**
     * @param list<float|int>      $x
     * @param list<float|int>      $y
     * @param list<float|int>      $z
     * @param list<float|int>|null $w weights (ones if null)
     */
    public function __construct(array $x, array $y, array $z, ?array $w = null, int $kx = 3, int $ky = 3, ?float $s = null)
    {
        $xv = array_map('floatval', array_values($x));
        $yv = array_map('floatval', array_values($y));
        $zv = array_map('floatval', array_values($z));
        $m = count($xv);
        if ($m !== count($yv) || $m !== count($zv)) {
            throw new \ValueError('BivariateSpline: x, y, z must have the same length');
        }
        if ($kx < 1 || $kx > 5 || $ky < 1 || $ky > 5) {
            throw new \ValueError('BivariateSpline: kx, ky must be in 1..5');
        }
        $wv = $w === null ? array_fill(0, $m, 1.0) : array_map('floatval', array_values($w));
        if (count($wv) !== $m) {
            throw new \ValueError('BivariateSpline: w must have the same length as x');
        }
        $sv = $s === null ? (float) $m - sqrt(2.0 * $m) : (float) $s;
        $r = Kernel::bisplrep($xv, $yv, $zv, $wv, $kx, $ky, $sv, 1e-16);
        $this->tx = self::flat($r['tx'] ?? $r[0]);
        $this->ty = self::flat($r['ty'] ?? $r[1]);
        $this->c = self::flat($r['c'] ?? $r[2]);
        $this->kx = $kx;
        $this->ky = $ky;
    }

    /**
     * Evaluate at the paired points (xi[k], yi[k]) — scipy.interpolate.BivariateSpline.ev(xi, yi).
     *
     * @param list<float|int> $xi
     * @param list<float|int> $yi
     * @return list<float>
     */
    public function ev(array $xi, array $yi): array
    {
        $xa = array_map('floatval', array_values($xi));
        $ya = array_map('floatval', array_values($yi));
        if (count($xa) !== count($ya)) {
            throw new \ValueError('ev: xi and yi must have the same length');
        }
        if (count($xa) === 0) {
            return [];
        }
        $pts = [];
        foreach ($xa as $k => $px) {
            $pts[] = [$px, $ya[$k]];
        }
        // NdBSpline wants the coefficient block as (ncx, ncy); the kernel surfit returns it raveled.
        $ncx = count($this->tx) - $this->kx - 1;
        $ncy = count($this->ty) - $this->ky - 1;
        $c2d = [];
        for ($i = 0; $i < $ncx; $i++) {
            $row = [];
            for ($j = 0; $j < $ncy; $j++) {
                $row[$j] = $this->c[$i * $ncy + $j];
            }
            $c2d[$i] = $row;
        }
        $res = Kernel::NdBSpline($this->tx, $this->ty, $c2d, $this->kx, $this->ky, $pts);
        return self::flat($res);
    }

    /** @return array{0: list<float>, 1: list<float>} the (tx, ty) knot vectors */
    public function getKnots(): array
    {
        return [$this->tx, $this->ty];
    }

    /** @return list<float> */
    public function getCoeffs(): array
    {
        return $this->c;
    }

    /** @param mixed $v @return list<float> */
    protected static function flat($v): array
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
